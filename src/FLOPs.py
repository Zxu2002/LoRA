import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from qwen import load_qwen
from untrained_Qwen import load_data
from preprocessor import LLMTIMEPreprocessor

def calculate_qwen_flops(
    seq_length,
    hidden_dim=896,
    mlp_dim=4864,
    num_layers=24,
    kv_dim=128,
    vocab_size=151936,
    is_training=True):
    """
    Calculate the approximate FLOPS for Qwen2.5-Instruct model.
    
    Parameters:
        - seq_length: Input sequence length
        - hidden_dim: Hidden dimension size (d_model)
        - mlp_dim: MLP intermediate dimension
        - num_layers: Number of transformer layers
        - kv_dim: Key/Value dimension per group (for grouped query attention)
        - vocab_size: Size of the vocabulary
        - is_training: Whether to include backward pass (training) or just forward pass (inference)
            
    Returns:
        - total_flops: Total number of floating point operations
    """
    #FLOPS for a single layer
    
    # RMSNorm (input_layernorm)
    rms_norm_flops = 4 * seq_length *  hidden_dim + 10 # +10 for square root
    
    # Query Projection: d_model -> d_model
    q_proj_flops = seq_length * hidden_dim * (2 * hidden_dim - 1)
    
    # Key/Value Projection: d_model -> d_kv
    
    k_proj_flops = seq_length * kv_dim * (2 * hidden_dim - 1)
    
    v_proj_flops = seq_length * kv_dim * (2 * hidden_dim - 1)
    
    # Rotary Position Embeddings
    rotary_flops = seq_length * 2 * hidden_dim
    
    # Attention Score Computation
    # Q*K^T: L × n_heads × (L × (2*d_kv/n_heads - 1))
    qk_flops = seq_length *7 * (seq_length * 128)
    
    scaling_flops = seq_length * 7 * seq_length
    
    softmax_flops = seq_length * 7 *(seq_length + 10)  # +10 for exponential
    
    # Attention Output
    # Attention * V: Same as QK^T
    attn_v_flops = seq_length * seq_length * 7 * 128
    
    o_proj_flops = seq_length * hidden_dim * (2 * hidden_dim - 1)
    
    post_rms_norm_flops = rms_norm_flops
    
    # MLP Components
    gate_proj_flops = seq_length *mlp_dim  * (2 * hidden_dim  - 1)
    up_proj_flops = seq_length *  mlp_dim* (2 * hidden_dim - 1) 
    
    # SwiGLU activation (SiLU + multiplication)
    swiglu_flops = seq_length * mlp_dim * 14

    # Down Projection
    down_proj_flops = seq_length * mlp_dim * (2 * hidden_dim - 1)
    
    layer_flops = (
        rms_norm_flops + 
        q_proj_flops + 
        k_proj_flops + 
        v_proj_flops + 
        rotary_flops + 
        qk_flops + 
        scaling_flops + 
        softmax_flops + 
        attn_v_flops + 
        o_proj_flops + 
        post_rms_norm_flops + 
        gate_proj_flops + 
        up_proj_flops + 
        swiglu_flops + 
        down_proj_flops
    )
    
    # Total for all layers
    total_layer_flops = num_layers * layer_flops

    # Final Layer Norm
    final_rms_norm_flops = seq_length * (2 * hidden_dim + 10)
    
    # LM Head
    lm_head_flops = seq_length * hidden_dim * (2 * vocab_size - 1)
    
    # Total forward pass FLOPS
    forward_flops = total_layer_flops + final_rms_norm_flops + lm_head_flops
    
    # Total FLOPS (including backward if training)
    if is_training:
        # Assume backward pass is 2x the forward pass
        total_flops = 3 * forward_flops
    else:
        total_flops = forward_flops
    
    return total_flops


def calculate_flops(context_length=80):
    ''' Evaluate the untrained model on the Lotka-Volterra dataset
    Parameters:
        - context_length: The length of the context segment
    Returns:
        - results: total number of flops 
        - qwen_seq_length: The length of the qwen sequence
    '''
    qwen, tokenizer = load_qwen()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen = qwen.to(device)
    qwen.eval()


    trajectories, time_points = load_data()
    traj = trajectories[0]
    input_segment = traj[:context_length]
    preprocessor = LLMTIMEPreprocessor()
    encoded_input = preprocessor.encode_sequence(input_segment)
    inputs = tokenizer(encoded_input, return_tensors="pt")
    qwen_seq_length = len(inputs["input_ids"][0])
    result = calculate_qwen_flops(seq_length=len(inputs["input_ids"][0]))

    return result, qwen_seq_length

def calculate_lora_flops(
    seq_length,
    hidden_dim=896,
    lora_rank=8,
    num_layers=24,
    is_training=True,
):


    """
    Calculate the additional FLOPs introduced by LoRA fine-tuning.
    
    Parameters:
        - seq_length: Input sequence length
        - hidden_dim: Hidden dimension size (d_model)
        - lora_rank: Rank of the LoRA matrices
        - num_layers: Number of transformer layers
        - is_training: Whether to include backward pass

            
    Returns:
        - lora_flops: Additional FLOPs from LoRA
    """
    # Count how many modules have LoRA applied
    lora_modules_count = 2

    
    # LoRA operations: x → (x @ A.T) @ B.T
  
    
    # 1. First matrix multiplication: (seq_length × hidden_dim) @ (lora_rank × hidden_dim).T
    lora_a_flops = seq_length * hidden_dim * (2 * lora_rank - 1)
    
    # 2. Second matrix multiplication: (seq_length × lora_rank) @ (hidden_dim × lora_rank).T
    lora_b_flops = seq_length * lora_rank * (2 * hidden_dim - 1)
    
    # Total FLOPs for one LoRA module
    lora_module_flops = lora_a_flops + lora_b_flops
    
    # Total LoRA FLOPs for one layer
    lora_layer_flops = lora_module_flops * lora_modules_count
    
    # Total LoRA FLOPs for all layers
    lora_forward_flops = lora_layer_flops * num_layers
    
    # Total FLOPs (including backward if training)
    if is_training:
        # Assume backward pass is 2x the forward pass
        lora_total_flops = 3 * lora_forward_flops
    else:
        lora_total_flops = lora_forward_flops
    
    return lora_total_flops

def calculate_total_flops_with_lora(
    seq_length,
    hidden_dim=896,
    mlp_dim=4864,
    num_layers=24,
    kv_dim=128,
    vocab_size=151936,
    lora_rank=8,
    num_steps=10000,
    is_training=True
):
    """
    Calculate the total FLOPs for a Qwen model with LoRA fine-tuning over multiple steps.
    
    Parameters:
        - seq_length: Input sequence length
        - hidden_dim: Hidden dimension size (d_model)
        - mlp_dim: MLP intermediate dimension
        - num_layers: Number of transformer layers
        - kv_dim: Key/Value dimension per head
        - vocab_size: Size of the vocabulary
        - lora_rank: Rank of the LoRA matrices
        - num_steps: Number of training/inference steps
        - is_training: Whether to include backward pass

            
    Returns:
        - Dictionary with FLOPs information
    """
    # Calculate base model FLOPs (using the existing function)
    base_model_flops = calculate_qwen_flops(
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        kv_dim=kv_dim,
        vocab_size=vocab_size,
        is_training=False
    )
    
    # Calculate additional LoRA FLOPs
    lora_flops = calculate_lora_flops(
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        lora_rank=lora_rank,
        num_layers=num_layers,
        is_training=is_training,

    )
    
    # Total FLOPs per step
    total_flops_per_step = base_model_flops + lora_flops
    
    # Total FLOPs for all steps
    total_flops = total_flops_per_step * num_steps
    
    return {
        "base_model_flops_per_step": base_model_flops,
        "lora_flops_per_step": lora_flops,
        "total_flops_per_step": total_flops_per_step,
        "total_flops": total_flops,

    }

def compute_hyperparameter_search_flops(
    seq_length,
    best_rank = 4,
    best_ctx_len = 256, 
    lora_ranks=[2, 4, 8],
    learning_rates=[1e-5, 5e-5, 1e-4],
    steps_per_config=2000,
    num_folds=5,
    context_length_steps=2000,
    context_lengths=[128, 512, 768],
    final_training_steps=30000
):
    """
    Calculate the total FLOPs for a hyperparameter search and final training.
    
    Parameters:
        - seq_length: Base sequence length
        - lora_ranks: List of LoRA ranks to try
        - learning_rates: List of learning rates to try
        - steps_per_config: Steps per configuration in hyperparameter search
        - num_folds: Number of k-fold cross-validation folds
        - context_length_steps: Steps for each context length experiment
        - context_lengths: List of context lengths to try
        - final_training_steps: Steps for final model training
            
    Returns:
        - Dictionary with FLOPs breakdown
    """
    total_flops = 0
    flops_breakdown = {}
    
    # 1. Hyperparameter search with k-fold cross-validation
    for rank in lora_ranks:
        for lr in learning_rates:
            config_key = f"rank_{rank}_lr_{lr}_kfold"
            
            # Calculate FLOPs for this configuration with k-fold
            config_flops = calculate_total_flops_with_lora(
                seq_length=seq_length,
                lora_rank=rank,
                num_steps=steps_per_config * num_folds,
                is_training=True
            )
            
            flops_breakdown[config_key] = config_flops["total_flops"]
            total_flops += config_flops["total_flops"]
    


    
    for ctx_len in context_lengths:
        config_key = f"context_length_{ctx_len}"
        
        # Calculate FLOPs for this context length
        config_flops = calculate_total_flops_with_lora(
            seq_length=ctx_len,
            lora_rank=best_rank,
            num_steps=context_length_steps,
            is_training=True
        )
        
        flops_breakdown[config_key] = config_flops["total_flops"]
        total_flops += config_flops["total_flops"]
    
    # 3. Final training with best hyperparameters
    final_config_key = "final_training"
    
    

    final_flops = calculate_total_flops_with_lora(
        seq_length=best_ctx_len,
        lora_rank=best_rank,
        num_steps=final_training_steps,
        is_training=True
    )
    
    flops_breakdown[final_config_key] = final_flops["total_flops"]
    total_flops += final_flops["total_flops"]
    
    return {
        "total_flops": total_flops,
        "breakdown": flops_breakdown,
        "under_budget": total_flops < 1e17
    }

# Example usage
if __name__ == "__main__":
    qwen_flops, qwen_input_length = calculate_flops()
    print(calculate_qwen_flops(256))
    # Calculate FLOPs for a single step with LoRA
    #Lora skeleton flops:
    max_ctx_length = 256
    lora_skeleton_flops = calculate_total_flops_with_lora(
        seq_length=min(qwen_input_length, max_ctx_length),
        lora_rank=4,
        num_steps=1,
        is_training=True
    )
    
    print("FLOPs for a single training step with LoRA:")
    print(f"Base model: {lora_skeleton_flops['base_model_flops_per_step']:,}")
    print(f"LoRA addition: {lora_skeleton_flops['lora_flops_per_step']:,}")
    print(f"Total: {lora_skeleton_flops['total_flops_per_step']:,}")
    
    # Calculate FLOPs for a full hyperparameter search
    search_flops = compute_hyperparameter_search_flops(
        seq_length=min(qwen_input_length, max_ctx_length)
    )
    

    print("Total FLOPs for hyperparameter search:")
    print(search_flops)
