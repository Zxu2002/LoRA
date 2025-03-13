import numpy as np
from transformers import AutoTokenizer
import json

class LLMTIMEPreprocessor:
    """
    Preprocessor for time series data based on the LLMTIME scheme.
    
    Implements preprocessing for Lotka-Volterra predator-prey time series data
    to be used with Qwen2.5-Instruct model for forecasting.
    """
    
    def __init__(self, scaling_factor=1.0, decimal_places=2,
                 var_separator=",", time_separator=";",
                 model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """Initialize the LLMTIME preprocessor.
        
        Parameters:
            - scaling_factor: Factor to scale numeric values
            - decimal_places: Number of decimal places to round to
            - var_separator: Separator between variables (default: ",")
            - time_separator: Separator between timesteps (default: ";")
            - model_name: Name of the model for tokenization
        """
        self.scaling_factor = scaling_factor
        self.decimal_places = decimal_places
        self.var_separator = var_separator
        self.time_separator = time_separator
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    
    
    def encode_timestep(self, values):
        """Encode values at a single timestep.
        
        Parameters:
            - values: Array of values for different variables at one timestep
            
        Returns:
            - final: String representation of the values using var_separator
        """
        # Scale 
        scaled = values / self.scaling_factor
        
        
        # Convert to strings 
        if isinstance(scaled, (int, float)):
            formatted = [f"{scaled:.{self.decimal_places}f}"]
        else:    
            formatted = [f"{val:.{self.decimal_places}f}" for val in scaled]

        #join with ,
        final = self.var_separator.join(formatted)
        
        return final
    
    def encode_sequence(self, sequence):
        """Encode a multivariate time series sequence.
        
        Parameters:
            - sequence: Array of shape (timesteps, variables) containing the time series
            
        Returns:
            - final: String representation of the entire sequence
        """
        # Encode each timestep
        encoded_timesteps = [self.encode_timestep(timestep) for timestep in sequence]

        #join with ;
        final = self.time_separator.join(encoded_timesteps)
        
        return final
    
    def decode_timestep(self, encoded_timestep):
        """Decode a single timestep from string to numeric values.
        
        Parameters:
            - encoded_timestep: String representation of values at one timestep
            
        Returns:
            - result: Array of numeric values for different variables
        """
        # Split by variable separator
        parts = encoded_timestep.split(self.var_separator)
        
        values = np.array([float(part) for part in parts if part != ""])
        result = values * self.scaling_factor
        
        return result
    
    def decode_sequence(self, encoded_sequence):
        """Decode an entire sequence from string to numeric array.
        
        Parameters:
            - encoded_sequence: String representation of the time series
            
        Returns:
            - decoded_timesteps: Time series shape(timesteps, variables) 
        """
        # Split by time separator
        encoded_timesteps = encoded_sequence.split(self.time_separator)
        
        # Decode each timestep
        decoded_timesteps = np.array([self.decode_timestep(step) for step in encoded_timesteps])
        
        
        return decoded_timesteps
    
    def tokenize(self, text):
        """Tokenize text using the model's tokenizer.
        
        Parameters:
            - text: Text to tokenize
            
        Returns:
            - tokens: List of token IDs
        """
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].tolist()[0]
        return tokens
    
    def prepare_training_data(self, data, context_length, forecast_length, stride=1):
        """Prepare time series data for training by creating input-output pairs.
        
        Parameters:
            - data: Array of shape (n_systems, timesteps, variables) or (timesteps, variables)
            - context_length: Number of timesteps to use as input context
            - forecast_length: Number of timesteps to predict
            - stride: Step size for sliding window (default: 1)
            
        Returns:
            - pairs: List of (input_sequence, target_sequence) pairs as strings
        """
        # Handle single system case
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        
        pairs = []
        
        for system in data:
            # For each possible starting point
            for i in range(0, len(system) - context_length - forecast_length + 1, stride):
                # Extract input and target segments
                input_segment = system[i:i+context_length]
                target_segment = system[i+context_length:i+context_length+forecast_length]
                
                # Encode both segments
                encoded_input = self.encode_sequence(input_segment)
                encoded_target = self.encode_sequence(target_segment)
                
                pairs.append((encoded_input, encoded_target))
        
        return pairs

    def get_example_sequences(self, data, seq_length=3):
        """Generate example sequences to demonstrate preprocessing and tokenization.
        
        Parameters:
            - data: Time series data array
            - num_examples: Number of examples to generate
            - seq_length: Length of each example sequence
            
        Returns:
            - examples: List of dictionaries containing original, preprocessed, and tokenized sequences
        """
        examples = []
        

        for _ in range(5):
       
            sequence = data[:seq_length]
                
        
            preprocessed = self.encode_sequence(sequence)
                
            # Tokenize the preprocessed sequence
            tokens = self.tokenize(preprocessed)
                
            examples.append({
                    'original': sequence,
                    'preprocessed': preprocessed,
                    'tokenized': tokens
                })
        
        return examples

def determine_scaling_factor(data):
    """Calculate scaling factor based on data.
        
    Parameters:
        - data: Array containing time series data
            
    Returns:
        - val: Saling factor for the data
    """
    # Calculate a scaling factor to get most values in 0-10 range 
    max_abs_val = np.max(np.abs(data))
    val = max_abs_val / 10.0
    return val


if __name__ == "__main__":
    data = [[1,2],[2,4],[3,6]]
    factor = determine_scaling_factor(data)


    data2 = [2,3,5]
    factor2 = determine_scaling_factor(data2)

    preprocessor = LLMTIMEPreprocessor(scaling_factor=factor)
    preprocessor2 = LLMTIMEPreprocessor(scaling_factor=factor2)
    result = {}
    result["example1"] = preprocessor.get_example_sequences(data)
    result["example2"] = preprocessor2.get_example_sequences(data2)
    save_path = "results/examples.json"
    with open(save_path, 'w') as f:
        json.dump(result, f)