# Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Stochastic Differential Equations

This repository contains the implementation of the **Multi-level PEnet** (MLPEnet), a model designed for parameter estimation in **stochastic differential equations (SDEs)**, particularly in cases driven by **non-Gaussian noises** such as Lévy processes. The work builds on recent advancements in machine learning, addressing challenges posed by traditional estimation techniques, especially when dealing with jump processes and heterogeneous parameters in SDEs.

## Key Features

1. **Three-Stage Architecture**: The model introduces a robust three-stage architecture for parameter estimation that improves generalization and performance over long sequences of observations.
2. **Handling Non-Gaussian Noise**: Specifically designed to manage complex noises like **α-stable Lévy** and **Student Lévy** processes, where traditional methods such as Maximum Likelihood Estimation (MLE) are inefficient.
3. **Integration of SAM**: The **Sharpness-Aware Minimization (SAM)** optimizer is integrated to enhance the generalization performance of the model, especially in boundary regions of the parameter space.

## Architecture Overview

1. **Multi-Level Condensation Phase**:
   - **1D Convolutional Neural Networks (CNNs)** and **split-concatenation operations** are used to handle heterogeneous parameters, significantly improving the processing of long sequences.
   
2. **Information Extraction Phase**:
   - The condensed information is processed using **LSTM** layers, capturing deep features from variable-length input sequences.
   
3. **Mapping to Parameter Space**:
   - A fully connected neural network maps the extracted features to the parameter space, allowing for accurate parameter estimation.

## Code Structure

- `main.py`: Main script to train and test the MLPEnet model.
- `model.py`: Defines the MLPEnet model architecture.
- `train.py`: Contains the training loop and the SAM optimizer integration.
- `test.py`: Script for testing the model's performance and generating results.
- `data_utils.py`: Handles data generation and pre-processing for SDEs driven by non-Gaussian noise.
- `utils.py`: Includes utility functions such as plotting, saving, and loading data.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- pickle

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the MLPEnet model on a dataset:

```bash
python main.py
```

The training configuration can be modified in the `config.py` file, where you can set parameters such as learning rate, batch size, and optimizer type.

### Testing the Model

To test the model on a specific dataset:

```bash
# Set test_model_file in config
python main.py --test
```

### Custom Dataset

To use your dataset, modify the data preprocessing pipeline in `data_utils.py` and ensure your dataset is formatted similarly to the provided examples.

## Citation

This project extends the work from:

- **PENN**: [https://github.com/xiaolong-snnu/PENN](https://github.com/xiaolong-snnu/PENN)
- **SAM**: [https://github.com/davda54/sam](https://github.com/davda54/sam)

Please also cite the associated paper if you use this repository in your research:

**Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Non-Gaussian Noise-Driven Stochastic Differential Equations**  
Shuaiyu Li, Hiroto Saigo, Yang Ruan, Yuzhong Cheng.  
*(To appear)*

## Acknowledgments

We extend our thanks to the developers of [PENN](https://github.com/xiaolong-snnu/PENN) and [SAM](https://github.com/davda54/sam) for their contributions to this field of research, which significantly inspired parts of this work.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
