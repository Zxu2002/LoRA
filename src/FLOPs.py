import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qwen import load_qwen
from untrained_Qwen import load_data
from preprocessor import LLMTIMEPreprocessor

def calculate_qwen_flops(
    seq_length,
    hidden_dim=896,
    mlp_dim=4864,
    num_layers=24,
    num_heads=8,
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
        - num_heads: Number of attention heads
        - kv_dim: Key/Value dimension per group (for grouped query attention)
        - vocab_size: Size of the vocabulary
        - is_training: Whether to include backward pass (training) or just forward pass (inference)
            
    Returns:
        - total_flops: Total number of floating point operations
    """
    #FLOPS for a single layer
    
    # RMSNorm (input_layernorm)
    rms_norm_flops = seq_length * (2 * hidden_dim + 10)  # +10 for square root
    
    # Query Projection: d_model -> d_model
    q_proj_flops = seq_length * hidden_dim * (2 * hidden_dim - 1)
    
    # Key/Value Projection: d_model -> d_kv
    k_proj_flops = seq_length * hidden_dim * (2 * kv_dim - 1)
    
    v_proj_flops = seq_length * hidden_dim * (2 * kv_dim - 1)
    
    # Rotary Position Embeddings
    rotary_flops = seq_length * 2 * hidden_dim
    
    # Attention Score Computation
    # Q*K^T: L × n_heads × (L × (2*d_kv/n_heads - 1))
    qk_flops = seq_length * seq_length * 254  # Simplified from above formula
    
    # Scaling
    scaling_flops = seq_length * seq_length
    
    # Softmax
    softmax_flops = seq_length * (seq_length + 10)  # +10 for exponential
    
    # Attention Output
    # Attention * V: Same as QK^T
    attn_v_flops = seq_length * seq_length * 254
    
    # Output Projection
    o_proj_flops = seq_length * hidden_dim * (2 * hidden_dim - 1)
    
    # Post Attention Normalization (same as input_layernorm)
    post_rms_norm_flops = rms_norm_flops
    
    # MLP Components
    gate_proj_flops = seq_length * hidden_dim * (2 * mlp_dim - 1)
    up_proj_flops = seq_length * hidden_dim * (2 * mlp_dim - 1)
    
    # SwiGLU activation (SiLU + multiplication)
    swiglu_flops = seq_length * mlp_dim * 11  # 10 for sigmoid + 1 for multiplication
    
    # Down Projection
    down_proj_flops = seq_length * mlp_dim * (2 * hidden_dim - 1)
    
    # Sum all layer components
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
    
    # 2. Final components
    
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


def calculate_flops(context_length=50):
    ''' Evaluate the untrained model on the Lotka-Volterra dataset
    Parameters:
        - context_length: The length of the context segment
    Returns:
        - results: total number of flops 
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
    
    result = calculate_qwen_flops(seq_length=len(inputs["input_ids"][0]))

    return result


if __name__ == "__main__":
    context_length = 50 


    print(calculate_flops())