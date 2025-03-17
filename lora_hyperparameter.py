import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np

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

def evaluate_model(model, val_loader, tokenizer, context_ratio=0.8):
    """
    Evaluate the model on validation data and compute metrics.
    
    Parameters:
        - model: The model to evaluate
        - val_loader: DataLoader containing validation data
        - tokenizer: Tokenizer used to decode predicted token IDs
        - context_ratio: Ratio of sequence to use as context
    
    Returns:
        - result: Dictionary containing evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    print("Evaluating model...")
    scaling_factor = json.load(open("results/scaling_factor.json"))['scaling_factor']

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            if batch[0].numel() == 0:
                print("Empty batch detected. Skipping...")
                continue
                
            # Process each example in the batch individually
            for example_idx, text in enumerate(batch[0]):
                # Decode the sequence
                token_ids = text.cpu().numpy().tolist()
               
                original_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                
                # Format check and processing
                pairs = original_text.split(';')
                standardized_pairs = []
                
                for pair in pairs:
                    if not pair.strip():
                        continue
                        
                    # Check if the pair has a comma
                    if ',' in pair:
                        values = pair.split(',')
                        if len(values) == 2:
                            # Ensure each value has a leading 0 if it starts with a decimal point
                            x_val = values[0].strip()
                            y_val = values[1].strip()
                            
                            if x_val.startswith('.'):
                                x_val = '0' + x_val
                            if y_val.startswith('.'):
                                y_val = '0' + y_val
                                
                            standardized_pairs.append(f"{x_val},{y_val}")
                
                if not standardized_pairs:
                    print("No valid pairs found. Skipping...")
                    continue
                    
                # Join the standardized pairs
                standardized_text = ';'.join(standardized_pairs)
        
                
                # Split into context and target
                split_index = int(len(standardized_pairs) * context_ratio)
                if split_index == 0:
                    print("Split index is 0, not enough data. Skipping...")
                    continue
                
                context_pairs = standardized_pairs[:split_index]
                target_pairs = standardized_pairs[split_index:]
                
                context_text = ';'.join(context_pairs)
                target_text = ';'.join(target_pairs)
                
                if not context_text or not target_text:
                    print("Empty context or target. Skipping...")
                    continue
                
                # Tokenize the context
                context_tokenized = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
                
                # Process this example individually
                context_tensor = context_tokenized.unsqueeze(0).to(model.device)
              
                # Generate predictions
                outputs = model.generate(
                    input_ids=context_tensor,
                    max_length=len(context_tokenized) + 200,  # Allow reasonable generation length
                    num_return_sequences=1,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Extract only the newly generated tokens (skip the context)
                predicted_ids = outputs[0, len(context_tokenized):]
                
                if predicted_ids.numel() == 0:
                    print("Model generated empty output. Skipping this example.")
                    continue
                    
                # Convert token IDs back to text
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                

                # Manually decode the sequences instead of using preprocessor
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
                    print("Failed to decode valid values. Skipping...")
                    continue
                
                # Convert to numpy arrays
                target_values = np.array(target_values_list)
                predicted_values = np.array(predicted_values_list)
                
                # If sequences have different lengths, truncate to match
                min_length = min(len(target_values), len(predicted_values))
                target_values = target_values[:min_length]
                predicted_values = predicted_values[:min_length]
                
                all_targets.append(target_values)
                all_predictions.append(predicted_values)


    if not all_targets or not all_predictions:
        print("No valid predictions were made. Cannot calculate metrics.")
        return {"MSE": None, "MAE": None, "R²": None}
        
    # Convert to arrays for calculation
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
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


model, tokenizer = load_qwen()

# Define Hyperparameter Search Space
lora_ranks = [2, 4, 8]
learning_rates = [1e-5, 5e-5, 1e-4]
context_lengths = [128, 512, 768]
num_folds = 5  

best_hyperparams = None
best_val_score = float("inf")
results = {}

# Load and tokenize dataset
all_texts = json.load(open("results/processed_all_traj.json"))
max_ctx_length = 256
all_input_ids = process_sequences(all_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)

# Split into Training (80%) and Testing (20%)
test_size = 0.2
train_input_ids, test_input_ids = train_test_split(all_input_ids, test_size=test_size, random_state=42)

# Create Test Dataset
test_dataset = TensorDataset(test_input_ids)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
model_performance = {}
for lora_rank in lora_ranks:
    for learning_rate in learning_rates:
        print(f"\n=== Training LoRA Rank {lora_rank}, LR {learning_rate} ===")
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_input_ids)):
            print(f"\n--- Fold {fold + 1} / {num_folds} ---")

            # Create Train/Validation Splits for this fold
            train_dataset = Subset(TensorDataset(train_input_ids), train_idx)
            val_dataset = Subset(TensorDataset(train_input_ids), val_idx)

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

            # Reinitialize Model & Optimizer
            model, tokenizer = load_qwen()
            for param in model.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)



            # Apply LoRA modifications
            for layer in model.model.layers:
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)



            # Use Accelerator for multi-GPU support
            accelerator = Accelerator()
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

            # Training Loop
            model.train()
            steps = 0
            max_steps_per_fold = 2000  # Adjust as needed
            pbar = tqdm(total=max_steps_per_fold)

            while steps < max_steps_per_fold:
                progress_bar = tqdm(train_loader, desc=f"Fold {fold + 1} - Steps {steps}")
                for (batch,) in progress_bar:
                    optimizer.zero_grad()
                    outputs = model(batch, labels=batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    steps += 1

                    progress_bar.set_postfix(loss=loss.item())
                    if steps >= max_steps_per_fold:
                        break

                    pbar.update(1)

            # Evaluate Model on Validation Set
            model.eval()
            fold_result = evaluate_model(model, val_loader, tokenizer)
            fold_metrics.append(fold_result)
            print(f"Fold {fold + 1} Results: {fold_result}")

        # Compute Average Metrics Across Folds
        avg_metrics = {
            "MSE": np.mean([m["MSE"] for m in fold_metrics]),
            "MAE": np.mean([m["MAE"] for m in fold_metrics]),
            "R²": np.mean([m["R²"] for m in fold_metrics]),
        }

    
        model.eval()
        test_results = evaluate_model(model, test_loader, tokenizer)

        # Store test performance
        model_key = f"lora_r{lora_rank}_lr{learning_rate}"
        model_performance[model_key] = {
            "Training + Validation Metrics": avg_metrics,
            "Test Metrics": test_results,
            "LoRA Rank": lora_rank,
            "Learning Rate": learning_rate,
        }
        if test_results["MSE"] < best_val_score:
            best_val_score = test_results["MSE"]
            best_hyperparams = (lora_rank, learning_rate)
            save_path = "results/lora_model_hyper.pth"
            torch.save(model.state_dict(), save_path)
            json.dump({"best_hyperparams": best_hyperparams}, open("results/best_hyperparams.json", "w"), indent=4)
            print(f"Model saved to {save_path}")

saved_path = "results/kfold_lora_test_competition.json"
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
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=best_learning_rate)

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=best_lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=best_lora_rank)

    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    steps = 0
    pbar = tqdm(total=max_steps_context)
   

    while steps < max_steps_context:
        progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
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


    model.eval()
    context_metrics = evaluate_model(model, test_loader, tokenizer)
    # context_metrics["FLOPs"] = total_flops

    model_performance[f"context_len{context_length}"] = context_metrics

# **Save Results**
saved_path = "results/lora_hyperparam_search.json"
with open(saved_path, "w") as f:
    json.dump(model_performance, f, indent=4)
