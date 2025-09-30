# AI Digit Recogniser

A Python project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. Includes training, evaluation, and a simple application to test digit recognition.

## Prerequisites
- Python 3.10+
- PyTorch with CUDA (optional, for GPU acceleration)

## Setting Up the Environment

1. Create a virtual environment:
    ```shell
    python -m venv .venv
    ```

2. Activate the virtual environment:
   - On Windows:
     ```shell
     .\.venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```shell
     source .venv/bin/activate
     ```

3. Installing Required Packages
    ```shell
    pip install -r requirements.txt
    ```

## Training and evaluating the Model
```shell
python src/train.py
```

## Running the app
```shell
python src/main.py
```