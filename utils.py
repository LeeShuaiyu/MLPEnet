import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import torch

def set_title(ax, string, fs=17):
    """
    Set the title for a matplotlib axis.

    Args:
        ax: The axis to set the title on.
        string: The title string.
        fs: Font size for the title.
    """
    ax.set_title(string, fontsize=fs, pad=10)  # Adjust the padding between the title and the plot

def enable_running_stats(model):
    """
    Enable running statistics for BatchNorm layers in the model.

    Args:
        model: The model containing BatchNorm layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.train()  # Set the BatchNorm to training mode
            module.track_running_stats = True  # Enable running statistics

def disable_running_stats(model):
    """
    Disable running statistics for BatchNorm layers in the model.

    Args:
        model: The model containing BatchNorm layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()  # Set the BatchNorm to evaluation mode
            module.track_running_stats = False  # Disable running statistics

def load_data(file_name):
    """
    Load data from a pickle file.

    Args:
        file_name: Path to the pickle file.

    Returns:
        data: The data loaded from the file.
    """
    with open(file_name, "rb") as fp:
        data = pickle.load(fp)
    return data

def save_data(file_name, data):
    """
    Save data to a pickle file.

    Args:
        file_name: Path to save the pickle file.
        data: The data to be saved.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def draw_gt_vs_estimated(config, saved_name='./data/temp.pkl'):
    """
    Plot ground truth vs. estimated values.

    Args:
        config: The configuration object containing model and dataset parameters.
        saved_name: Path to the saved data file.
    """
    # ... (Fill this in with your code and necessary comments)
    pass  # Replace with actual implementation

def plot_residuals(y_true, y_pred, param_name):
    """
    Plot the residuals between true and predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        param_name: The name of the parameter being plotted.
    """
    # ... (Fill this in with your code and necessary comments)
    pass  # Replace with actual implementation

def plot_error_distribution(y_true, y_pred, param_name):
    """
    Plot the distribution of errors between true and predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        param_name: The name of the parameter for which the errors are plotted.
    """
    # ... (Fill this in with your code and necessary comments)
    pass  # Replace with actual implementation

def save_results_to_csv(file_path, results):
    """
    Save a dictionary of results to a CSV file.

    Args:
        file_path: The path to the CSV file.
        results: A dictionary containing the results to be saved.
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file does not exist
            writer.writerow(results.keys())
        # Write the data
        writer.writerow(results.values())
