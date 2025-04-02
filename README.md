# M2 Coursework: LoRA Fine-Tuning for Time Series Forecasting

This repository contains code and report for the M2 coursework. 


## Setting up the environment 

### Prerequisites

- Python 3.9.6
- `pip` (Python package installer)

### Setting Up a Virtual Environment

1. Open a terminal and navigate to root of this directory:


2. Create a virtual environment:
    try:
    ```sh
    python -m venv venv
    ```
    if above does not work: 
    ```sh
    python3 -m venv venv
    ```

3. Activate the virtual environment:

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

### Installing Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

## Usage
All scripts should be run directly from the root directory.

### Key Scripts in Root Directory

- **`lora_hyperparameter.py`** – Runs hyperparameter tuning for the model.
- **`lora_skeleton.py`** – Provides the basic structure of the LoRA model.
- **`lora_final.py`** – Trains the model using the optimal hyperparameters.

### Scripts in `src/` Directory

- **`calculate_Qwen_result.py`** – Performs numerical analysis of metrics for the untrained Qwen model.
- **`FLOPs.py`** – Calculates the FLOPs used by each model.
- **`preprocessor.py`** – Defines the data preprocessing pipeline.
- **`untrained_Qwen.py`** – Loads and evaluates the untrained Qwen model.


