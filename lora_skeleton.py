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


# LoRA implementation
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


model, tokenizer = load_qwen()
lora_rank = 4
for param in model.parameters():
    param.requires_grad = False

# Actually apply LoRA to the model:
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
# ^These are the parts that will actually be trained!

# Process the data into sequences of text
train_percentage = 0.8
all_texts = json.load(open("results/processed_all_traj.json"))
train_texts = all_texts[: int(len(all_texts) * train_percentage)]
val_texts = all_texts[int(len(all_texts) * train_percentage) :]

# ^Each of these is a `list[str]` representing contiguous parts of the time series,
#  in text form (using the LLMTIME scheme).


# Modified tokenization with chunking
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

def evaluate_model(model, val_loader, tokenizer):
    """
    Evaluate the model on validation data and compute metrics.
    
    Parameters:
        - model: The model to evaluate
        - val_loader: DataLoader containing validation data
        - tokenizer: Tokenizer used to decode predicted token IDs
    
    
    Returns:
        - result: Dictionary containing evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    print("Evaluating model...")
    scaling_factor = json.load(open("results/scaling_factor.json"))
    preprocessor = LLMTIMEPreprocessor(scaling_factor=scaling_factor)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Use first 50% of sequence as context, predict last 50%
            context_length = int(batch.shape[1] * 0.8)

            # Extract context and targets
            context_ids = batch[:, :context_length]
            target_ids = batch[:, context_length:]
            
            # Generate predictions
            outputs = model.generate(
                input_ids=context_ids,
                max_length=batch.shape[1],
                num_return_sequences=1,
                do_sample=False  # Use greedy decoding
            )
            
            # Extract only the newly generated tokens
            predicted_ids = outputs[:, context_length:]
            
            # Convert token IDs back to text for both targets and predictions
            for i in range(len(target_ids)):
                target_text = tokenizer.decode(target_ids[i], skip_special_tokens=True)
                predicted_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
                
                # Convert text to numerical values using preprocessor
                try:
                    target_values = preprocessor.decode_sequence(target_text)
                    predicted_values = preprocessor.decode_sequence(predicted_text)
                    
                    # If sequences have different lengths, truncate to match
                    min_length = min(len(target_values), len(predicted_values))
                    target_values = target_values[:min_length]
                    predicted_values = predicted_values[:min_length]
                    
                    all_targets.append(target_values)
                    all_predictions.append(predicted_values)
                except:
                    # Skip examples where parsing fails
                    continue
    
    # Convert to arrays for calculation
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    mse = np.mean((all_targets - all_predictions) ** 2)
    mae = np.mean(np.abs(all_targets - all_predictions))
    
    # Calculate R²
    # R² = 1 - (sum of squared residuals) / (total sum of squares)
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    result = {
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    }
    return result

# Defines the maximum context length
max_ctx_length = 256
train_input_ids = process_sequences(
    train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2
)
val_input_ids = process_sequences(
    val_texts, tokenizer, max_ctx_length, stride=max_ctx_length
)

batch_size = 2
learning_rate = 1e-5

optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=learning_rate
)
train_dataset = TensorDataset(train_input_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(val_input_ids)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# Prepare components with Accelerator
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

model.train()
steps = 0
pbar = tqdm(total=10000)
while steps < 10000:
    progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
    for (batch,) in progress_bar:
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        steps += 1

        progress_bar.set_postfix(loss=loss.item())
        if steps > 10000:
            break

        pbar.update(1)

# model.eval()
lora_metrics = evaluate_model(model, valid_loader, tokenizer)
print(lora_metrics)
# Save the model
saved_path = "results/lora_metrics.json"
with open(saved_path, 'w') as f:
    json.dump({"lora_metrics": lora_metrics}, f)