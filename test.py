import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from model import MLPenet
from utils import load_data, save_data, draw_gt_vs_estimated, plot_residuals, plot_error_distribution, \
    save_results_to_csv
from dataloader import MyDataset  # Ensure the correct import
import os


def test_MLPenet_model(config, saved_name='./data/temp.pkl', device_mode='cuda'):
    """
    Test function for the MLPenet model.

    Args:
        config: Configuration object containing model and dataset parameters.
        saved_name: File path to save the test results.
        device_mode: Specify whether to use 'cpu' or 'cuda' for testing.

    Returns:
        ys: Ground truth labels from the test dataset.
        os: Model predictions for the test dataset.
    """
    model_file = config.param['test_model_file']
    print('Start loading model', model_file)

    # Set device to CPU if CUDA is not available
    if device_mode == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        batch_size = 2000
        model_CKPT = torch.load(model_file, map_location=torch.device('cpu'))
    else:
        device = torch.device('cuda')
        batch_size = 10000
        model_CKPT = torch.load(model_file, map_location=torch.device('cuda'))

    # Load the test dataset
    test_data = MyDataset(config.param['test_file'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize the model
    model = MLPenet(
        config.param['mlpenet_in_dim'],
        config.param['mlpenet_hidden_dim'],
        config.param['mlpenet_n_layer'],
        conv_out_channels=config.param['mlpenet_conv_out_channels'],
        conv_kernel_size=config.param['mlpenet_conv_kernel_size'],
        activation=config.param['mlpenet_activation'],
        operation='split'
    ).to(device)

    # Load the model state
    model.load_state_dict(model_CKPT['state_dict'], strict=False)
    model = model.double()
    ys = []
    os = []
    time_start = time.time()

    # Disable gradient calculation for testing
    with torch.no_grad():
        model.eval()
        for i, (x, y, l, h, _) in enumerate(test_loader):
            x = x.to(device)
            l = l.to(device)
            h = h.to(device)
            outputs = model(x, l, h)
            ys.append(y.cpu().numpy())
            os.append(outputs.cpu().numpy())

    # Stack the results and calculate time used
    ys = np.vstack(ys)
    os = np.vstack(os)
    time_used = time.time() - time_start
    print(f'Tested {len(test_data)} samples in {time_used:.2f} seconds')

    # Save the results
    save_data(saved_name, [ys, os])
    return ys, os


def test_MLPenet(config):
    """
    Main testing function for the MLPenet model. Generates results and plots.

    Args:
        config: Configuration object containing model and dataset parameters.
    """
    # Run the model and get predictions
    gt, pred = test_MLPenet_model(config)

    # Plot ground truth vs estimated values
    draw_gt_vs_estimated(config)

    # Get the parameter name for labeling plots
    param_name = config.param['param_name_latex'][0]

    # Plot residuals
    plot_residuals(gt, pred, param_name)

    # Plot error distribution
    plot_error_distribution(gt, pred, param_name)

    # Create a dictionary to store test results
    results = {}

    # Set dataset-related parameters
    results['Train Samples'] = config.param['train_num']
    results['Eval Samples'] = config.param['eval_num']
    results['Test Samples'] = config.param['test_num']
    results['Eta Range'] = config.param['eta']
    results['Epsilon Range'] = config.param['epsilon']
    results['Alpha Range'] = config.param['alpha']

    # Set network-related parameters
    results['LSTM Layers'] = config.param['mlpenet_n_layer']
    results['Hidden Units'] = config.param['mlpenet_hidden_dim']
    results['Activation Function'] = config.param['mlpenet_activation']
    results['Learning Rate'] = config.param['learning_rate']
    results['Batch Size'] = config.param['batch_size']
    results['Training Epochs'] = config.param['num_epochs']

    # Compute and log test statistics
    mae = np.mean(np.abs(gt[:, 0] - pred))  # Mean Absolute Error
    results[f'{param_name} Predicted Mean'] = np.mean(pred)
    results[f'{param_name} Predicted Std'] = np.std(pred)
    results[f'{param_name} Residual Mean'] = np.mean(gt[:, 0] - pred)
    results[f'{param_name} Residual Std'] = np.std(gt[:, 0] - pred)
    results[f'{param_name} Predicted Min'] = np.min(pred)
    results[f'{param_name} Predicted Max'] = np.max(pred)
    results[f'{param_name} Residual Min'] = np.min(gt[:, 0] - pred)
    results[f'{param_name} Residual Max'] = np.max(gt[:, 0] - pred)
    results['MAE'] = mae

    # Print the results
    for key, value in results.items():
        print(f'{key}: {value}')

    # Ensure the result directory exists
    if not os.path.exists('MLPenet_result'):
        os.makedirs('MLPenet_result')

    # Save results to a CSV file
    save_results_to_csv('MLPenet_result/test_results.csv', results)
