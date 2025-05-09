import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
# Import the model 
from qwen import load_qwen
from transformers import AutoTokenizer
from .preprocessor import LLMTIMEPreprocessor, determine_scaling_factor


# qwen, tokenizer = load_qwen()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# qwen = qwen.to(device)
def load_data():
    with h5py.File("lotka_volterra_data.h5", "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
    return trajectories, time_points

# Split data into train/validation sets
def split_data(trajectories, train_ratio=0.8):
    n_systems = trajectories.shape[0]
    n_train = int(n_systems * train_ratio)
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:]
    
    return train_data, val_data

def evaluate_untrained_model(ratio=0.8,test_size=100,context_length = 80, forecast_length = 20):
    '''Evaluate the untrained model on the Lotka-Volterra dataset
    
    Parameters:
        - ratio: The ratio of the dataset to use for training
        - test_size: The number of systems to evaluate
        - context_length: The length of the context segment
        - forecast_length: The length of the forecast segment

    Returns:
        - results: A dictionary containing the evaluation results
    '''

    # Load the data
    trajectories, time_points = load_data()

    #Load the model
    qwen, tokenizer = load_qwen()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen = qwen.to(device)
    qwen.eval()

    # preprocessor = LLMTIMEPreprocessor()
    errors_mse = []
    errors_mae = []
    errors_r2 = []
    validation_size = int(len(trajectories) * (1-ratio))
    validation_data = trajectories[-validation_size:]

    scaling_factor = determine_scaling_factor(trajectories[:])
    preprocessor = LLMTIMEPreprocessor(scaling_factor=scaling_factor)
    with torch.no_grad():
        for system_idx in tqdm(range(min(len(validation_data),len(trajectories)))): 
            #Load preprocessor


            # Extract input and target segments
            input_segment = trajectories[system_idx, :context_length]
            target_segment = trajectories[system_idx, context_length:context_length+forecast_length]
            
            # Preprocess input segment
            encoded_input = preprocessor.encode_sequence(input_segment)
            
            # Tokenize for model input
            inputs = tokenizer(encoded_input, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # inputs = {k: v.to(qwen.device) for k, v in inputs}
            print(len(inputs["input_ids"]))
            # Generate continuation
            output_ids = qwen.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=forecast_length * 10,
                do_sample=False,      # Deterministic generation
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract the generated sequence 
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            generated_sequence = generated_text[len(prompt_text):]
    
            # Split the generated sequence and truncate to forecast length

            forecast_parts = [part.strip() for part in generated_sequence.split(';') if part.strip()]
            # print(forecast_parts)
            clean_forecast = ';'.join(forecast_parts)
            predicted_segment = preprocessor.decode_sequence(clean_forecast)
                
            pred_shaped = predicted_segment[:forecast_length, :] if len(predicted_segment) >= forecast_length else np.pad(
                    predicted_segment, ((0, forecast_length - len(predicted_segment)), (0, 0)))
                
            # Calculate metrics
            mse = mean_squared_error(target_segment.flatten(), pred_shaped.flatten())
            mae = mean_absolute_error(target_segment.flatten(), pred_shaped.flatten())
            r2 = r2_score(target_segment.flatten(), pred_shaped.flatten())
                
            errors_mse.append(mse)
            errors_mae.append(mae)
            errors_r2.append(r2)
                

    results = {
        'MSE': errors_mse,
        'MAE': errors_mae,
        'R²': errors_r2,
        'Num_Systems_Evaluated': len(errors_mse)
    }

    return results


def plot_trajectory(trajectory_id, context_length=80, forecast_length = 20):
    qwen, tokenizer = load_qwen()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen = qwen.to(device)
    qwen.eval()
    trajectories, time_points = load_data()
    scaling_factor = determine_scaling_factor(trajectories[:])
    preprocessor = LLMTIMEPreprocessor(scaling_factor=scaling_factor)
    trajectory = trajectories[trajectory_id]
    input_segment = trajectory[:context_length]
    target_segment = trajectory[context_length:context_length+forecast_length]
    encoded_input = preprocessor.encode_sequence(input_segment)
    inputs = tokenizer(encoded_input, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = qwen.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=forecast_length * 10,
        do_sample=False,      # Deterministic generation
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
    plt.savefig("graphs/trajectory_{}.png".format(trajectory_id))
    plt.show()

if __name__ == "__main__":
    result = evaluate_untrained_model()
    save_path = "results/untrained_qwen_results.json"
    with open(save_path, 'w') as f:
        json.dump(result, f)
    print(result)
    qwen, tokenizer = load_qwen()
    print(qwen)
    
    plot_trajectory(0)