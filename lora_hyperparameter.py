import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from src.preprocessor import LLMTIMEPreprocessor
from qwen import load_qwen
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)




def process_sequences(texts, tokenizer, max_length=512, stride=256):
    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)


def evaluate_model_batched(model, val_loader, tokenizer, context_ratio=0.8, batch_size=8):
    """
    Evaluate the model on validation data using batched processing.
    
    Parameters:
        - model: The model to evaluate
        - val_loader: DataLoader containing validation data
        - tokenizer: Tokenizer used to decode predicted token IDs
        - context_ratio: Ratio of sequence to use as context
        - batch_size: Size of batches for generation
    
    Returns:
        - result: Dictionary containing evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    print("Evaluating model with batched processing...")
    scaling_factor = json.load(open("results/scaling_factor.json"))['scaling_factor']

    with torch.no_grad():
        for batch_idx, (input_batch,) in enumerate(tqdm(val_loader)):
            if input_batch.numel() == 0:
                continue
            
            # Step 1: Preprocess data to extract contexts and targets
            batch_contexts = []
            batch_targets = []
            batch_context_lengths = []
            valid_indices = []
            
            # Process each example to extract context and target
            for i, text in enumerate(input_batch):
                token_ids = text.cpu().numpy().tolist()
                original_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                
                # Standardize the text format
                pairs = original_text.split(';')
                standardized_pairs = []
                
                for pair in pairs:
                    if not pair.strip():
                        continue
                    if ',' in pair:
                        values = pair.split(',')
                        if len(values) == 2:
                            x_val = values[0].strip()
                            y_val = values[1].strip()
                            
                            if x_val.startswith('.'):
                                x_val = '0' + x_val
                            if y_val.startswith('.'):
                                y_val = '0' + y_val
                                
                            standardized_pairs.append(f"{x_val},{y_val}")
                
                if not standardized_pairs:
                    continue
                
                # Split into context and target
                split_index = int(len(standardized_pairs) * context_ratio)
                if split_index == 0:
                    continue
                
                context_pairs = standardized_pairs[:split_index]
                target_pairs = standardized_pairs[split_index:]
                
                context_text = ';'.join(context_pairs)
                target_text = ';'.join(target_pairs)
                
                if not context_text or not target_text:
                    continue
                
                # Store the valid example
                batch_contexts.append(context_text)
                batch_targets.append(target_text)
                valid_indices.append(i)
            
            if not batch_contexts:
                continue
            
            # Step 2: Tokenize all contexts in a batch
            encoded_contexts = tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).to(model.device)
            
            input_ids = encoded_contexts["input_ids"]
            attention_mask = encoded_contexts["attention_mask"]
            
            # Keep track of original context lengths for extracting generations
            context_lengths = attention_mask.sum(dim=1).tolist()
            
            # Step 3: Generate completions for the whole batch at once
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 200,  # Allow reasonable generation length
                num_return_sequences=1,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,    # Enable KV caching for faster generation
                return_dict_in_generate=True,
                output_scores=False
            )
            
            generated_sequences = outputs.sequences
            
            # Step 4: Process each generated sequence and corresponding target
            for i, (context_length, target_text) in enumerate(zip(context_lengths, batch_targets)):
                # Extract only the newly generated tokens
                generated_ids = generated_sequences[i, context_length:]
                
                if generated_ids.numel() == 0:
                    continue
                
                predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Parse target values
                target_values_list = []
                for timestep in target_text.split(';'):
                    if not timestep.strip():
                        continue
                    try:
                        parts = timestep.split(',')
                        if len(parts) == 2:
                            x = float(parts[0].strip()) * scaling_factor
                            y = float(parts[1].strip()) * scaling_factor
                            target_values_list.append([x, y])
                    except:
                        continue
                
                # Parse predicted values
                predicted_values_list = []
                for timestep in predicted_text.split(';'):
                    if not timestep.strip():
                        continue
                    try:
                        parts = timestep.split(',')
                        if len(parts) == 2:
                            x = float(parts[0].strip()) * scaling_factor
                            y = float(parts[1].strip()) * scaling_factor
                            predicted_values_list.append([x, y])
                    except:
                        continue
                
                if not target_values_list or not predicted_values_list:
                    continue
                
                # Convert to numpy arrays
                target_values = np.array(target_values_list)
                predicted_values = np.array(predicted_values_list)
                
                # Match sequence lengths
                min_length = min(len(target_values), len(predicted_values))
                target_values = target_values[:min_length]
                predicted_values = predicted_values[:min_length]
                
                all_targets.append(target_values)
                all_predictions.append(predicted_values)

    if not all_targets or not all_predictions:
        print("No valid predictions were made. Cannot calculate metrics.")
        return {"MSE": None, "MAE": None, "R²": None}
        
    # Calculate metrics on all data at once
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    mse = np.mean((all_targets - all_predictions) ** 2)
    mae = np.mean(np.abs(all_targets - all_predictions))
    
    # Calculate R²
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    result = {
        "MSE": float(mse),
        "MAE": float(mae),
        "R²": float(r2)
    }
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer = load_qwen()
model = model.to(device)
print(f"Model is now on: {next(model.parameters()).device}")
# Define Hyperparameter Search Space
lora_ranks = [2, 4, 8]
learning_rates = [1e-5, 5e-5, 1e-4]
context_lengths = [128, 512, 768]
num_folds = 3

best_hyperparams = None
best_val_score = float("inf")
results = {}

# Load and tokenize dataset
all_texts = json.load(open("results/processed_all_traj.json"))
max_ctx_length = 256
all_input_ids = process_sequences(all_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)

# Create train/validation/test splits with proportions 0.6/0.2/0.2
train_input_ids, temp_input_ids = train_test_split(all_input_ids, test_size=0.4, random_state=42)
val_input_ids, test_input_ids = train_test_split(temp_input_ids, test_size=0.5, random_state=42)

train_dataset = TensorDataset(train_input_ids)
val_dataset = TensorDataset(val_input_ids)
test_dataset = TensorDataset(test_input_ids)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model_performance = {}
for lora_rank in lora_ranks:
    for learning_rate in learning_rates:
        print(f"\n=== Training LoRA Rank {lora_rank}, LR {learning_rate} ===")

        model, tokenizer = load_qwen()
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False


        # Apply LoRA modifications
        for layer in model.model.layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)


        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        # Use Accelerator for multi-GPU support
        accelerator = Accelerator()
        model, optimizer, prepared_train_loader = accelerator.prepare(model, optimizer, train_loader)

            # Training Loop
        model.train()
        steps = 0
        max_steps = 10000  # Adjust as needed
        pbar = tqdm(total=max_steps)

        while steps < max_steps:
            progress_bar = tqdm(prepared_train_loader, desc=f"Steps {steps}")
            for (batch,) in progress_bar:
                optimizer.zero_grad()
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                steps += 1

                progress_bar.set_postfix(loss=loss.item())
                if steps >= max_steps:
                    break

                pbar.update(1)

    
        # val_results = evaluate_model(model, val_loader, tokenizer)
        val_results = evaluate_model_batched(model, val_loader, tokenizer)
        print(f"Validation Results: {val_results}")

        # Evaluate Model on Test Set
        # test_results = evaluate_model(model, test_loader, tokenizer)
        test_results = evaluate_model_batched(model, test_loader, tokenizer)
        print(f"Test Results: {test_results}")


        model_key = f"lora_r{lora_rank}_lr{learning_rate}"
        model_performance[model_key] = {
            "Validation Metrics": val_results,
            "Test Metrics": test_results,
            "LoRA Rank": lora_rank,
            "Learning Rate": learning_rate,
        }
        
        if val_results["MSE"] < best_val_score:
            best_val_score = val_results["MSE"]
            best_hyperparams = (lora_rank, learning_rate)
            save_path = "results/lora_model_hyper.pth"
            torch.save(model.state_dict(), save_path)
            json.dump({"best_hyperparams": best_hyperparams}, open("results/best_hyperparams.json", "w"), indent=4)
            print(f"Model saved to {save_path}")

saved_path = "results/lora_test_competition.json"
with open(saved_path, "w") as f:
    json.dump(model_performance, f, indent=4)

# Find Best Hyperparameters
sorted_models = sorted(model_performance.items(), key=lambda x: x[1]["Test Metrics"]["MSE"])
best_model_key, best_model_metrics = sorted_models[0]
best_hyperparams = (best_model_metrics["LoRA Rank"], best_model_metrics["Learning Rate"])


best_lora_rank, best_learning_rate = best_hyperparams
results["best_hyperparams"] = best_hyperparams
print(f"\nBest Hyperparameters: Rank={best_lora_rank}, LR={best_learning_rate}")
max_steps_context = 2000

# Re-train Model with Best Hyperparameters
for context_length in context_lengths:
    print(f"\n=== Testing Context Length {context_length} ===")
    model, tokenizer = load_qwen()
    model = model.to(device)
    

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=best_lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=best_lora_rank)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=best_learning_rate)
    accelerator = Accelerator()
    model, optimizer, prepared_train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    steps = 0
    pbar = tqdm(total=max_steps_context)
   

    while steps < max_steps_context:
        progress_bar = tqdm(prepared_train_loader, desc=f"Steps {steps}")
        for (batch,) in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1

            if steps >= max_steps_context:
                break
            pbar.update(1)



    # context_metrics = evaluate_model(model, test_loader, tokenizer)
    context_metrics = evaluate_model_batched(model, test_loader, tokenizer)
    # context_metrics["FLOPs"] = total_flops

    model_performance[f"context_len{context_length}"] = context_metrics

# **Save Results**
saved_path = "results/lora_hyperparam_search.json"
with open(saved_path, "w") as f:
    json.dump(model_performance, f, indent=4)
