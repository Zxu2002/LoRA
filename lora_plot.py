from src.preprocessor import LLMTIMEPreprocessor, determine_scaling_factor
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.untrained_Qwen import load_data
from qwen import load_qwen
from lora_skeleton import LoRALinear

def plot_trajectory(model, trajectory_id, tokenizer, context_length=80, forecast_length = 20,save_path="graphs/lora_skele_trajectory.png"):
    model.eval()
    trajectories, time_points = load_data()
    scaling_factor = determine_scaling_factor(trajectories[:])
    preprocessor = LLMTIMEPreprocessor(scaling_factor=scaling_factor)
    trajectory = trajectories[trajectory_id]
    input_segment = trajectory[:context_length]
    target_segment = trajectory[context_length:context_length+forecast_length]
    encoded_input = preprocessor.encode_sequence(input_segment)
    inputs = tokenizer(encoded_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs,max_length= 1000)     
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated_sequence = generated_text[len(prompt_text):]
    forecast_parts = [part.strip() for part in generated_sequence.split(';') if part.strip()]
    clean_forecast = ';'.join(forecast_parts)
    predicted_segment = preprocessor.decode_sequence(clean_forecast)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(np.arange(context_length), input_segment[:, 0], label="Input")
    axes[0].plot(np.arange(context_length, context_length + forecast_length), target_segment[:, 0], label="Target")
    axes[0].plot(np.arange(context_length, context_length + forecast_length), predicted_segment[:, 0], label="Prediction")
    axes[0].legend()
    axes[0].set_title("Prey")

    # Example of adding a second subplot
    axes[1].plot(np.arange(context_length), input_segment[:, 1], label="Input")
    axes[1].plot(np.arange(context_length, context_length + forecast_length), target_segment[:, 1], label="Target")
    axes[1].plot(np.arange(context_length, context_length + forecast_length), predicted_segment[:, 1], label="Prediction")
    axes[1].legend()
    axes[1].set_title("Predator")

    plt.tight_layout()
    fig.suptitle("Trajectory {}".format(trajectory_id))
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    model, tokenizer = load_qwen()

    # Ensure LoRA is applied before loading the state dict
    lora_rank = 4
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    # Load the saved parameters
    # model.load_state_dict(torch.load("results/lora_model.pth"))
    model.load_state_dict(torch.load("results/lora_model.pth", map_location=torch.device('cpu')))

    print("Model parameters loaded successfully!")
    model, tokenizer = load_qwen()

    # Ensure LoRA is applied before loading the state dict
    lora_rank = 8
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    plot_trajectory(model, trajectory_id=0, tokenizer=tokenizer, context_length=80, forecast_length=20)

    model.load_state_dict(torch.load("graphs/lora_model_hyper.pth", map_location=torch.device('cpu')))
    print("Model parameters loaded successfully!")
    plot_trajectory(model, trajectory_id=0, tokenizer=tokenizer, context_length=80, forecast_length=20,save_path="results/lora_hyper.png")

    model.load_state_dict(torch.load("graphs/lora_model_final.pth", map_location=torch.device('cpu')))
    print("Model parameters loaded successfully!")
    plot_trajectory(model, trajectory_id=0, tokenizer=tokenizer, context_length=80, forecast_length=20,save_path="results/lora_final.png")

    model.load_state_dict(torch.load("graphs/lora_model_final_cxt.pth", map_location=torch.device('cpu')))
    print("Model parameters loaded successfully!")
    plot_trajectory(model, trajectory_id=0, tokenizer=tokenizer, context_length=80, forecast_length=20,save_path="results/lora_final_cxt.png")