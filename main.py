from config import Config
from data_utils import sampling_func
from train import train_MLPenet
from data import generate_param, sample_data
from dataloader import MyDataset
from test import test_MLPenet

if __name__ == '__main__':
    # Step 1: Initialize the configuration
    config = Config()

    # Step 2 (optional): Modify the configuration parameters if needed
    # Example: config.param['train_num'] = 1500
    # This step allows you to customize settings for experiments

    # Step 3: Sample data (uncomment if data generation is needed)
    # This step generates the training data if not already available
    #sample_data(config, sampling_func, mode='train')

    # Step 4: Train the MLPenet model
    # This step trains the model based on the configuration settings
    train_MLPenet(config)

    # Step 5: Test the model (uncomment if testing is required)
    # To test, specify the path to the saved model file and ensure the model exists
    config.param['test_model_file'] = 'path_to_your_saved_model.ckpt'
    sample_data(config, sampling_func, mode='test')  # Generate testing data if needed
    test_MLPenet(config)  # Run model inference and evaluation
