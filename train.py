import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from sam import SAM
from utils import enable_running_stats, disable_running_stats
from dataloader import MyDataset  # Ensure correct import
from model import MLPenet

def train_MLPenet(config):
    """
    Training function for the MLPenet model.

    Args:
        config: Configuration object containing model and training parameters.
    """
    P = config.param
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train file:', P['train_file'])
    print('Eval file:', P['eval_file'])
    print('CUDA availability:', torch.cuda.is_available())

    train_data = MyDataset(P['train_file'])
    eval_data = MyDataset(P['eval_file'])

    num_epochs = P['num_epochs']
    init_weight_file = P['mlpenet_init_weight_file']
    param_name = P['param_name']
    optimizer_type = P.get('optimizer', 'adam')  # Get optimizer type

    # Initialize the model
    model = MLPenet(
        P['mlpenet_in_dim'],
        P['mlpenet_hidden_dim'],
        P['mlpenet_n_layer'],
        conv_out_channels=P['mlpenet_conv_out_channels'],
        conv_kernel_size=P['mlpenet_conv_kernel_size'],
        activation=P['mlpenet_activation'],
        operation='split'
    ).to(device)

    model = model.double()  # Use double precision
    if init_weight_file:
        print('=' * 50)
        print('Loading model', init_weight_file)
        print('=' * 50)
        model_CKPT = torch.load(init_weight_file)
        model.load_state_dict(model_CKPT['state_dict'])
        # Initialize the optimizer based on the specified type
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=P['learning_rate'])
        elif optimizer_type == 'sam':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=P['learning_rate'], momentum=0.9)
        else:
            raise ValueError("Invalid optimizer type")
        optimizer.load_state_dict(model_CKPT['optimizer'])
        init_epoch = model_CKPT['epoch']
    else:
        model.init_weights()  # Initialize model weights
        # Initialize the optimizer based on the specified type
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=P['learning_rate'])
        elif optimizer_type == 'sam':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=P['learning_rate'], momentum=0.9)
        else:
            raise ValueError("Invalid optimizer type")
        init_epoch = 0

    criterion = nn.MSELoss()  # Loss function
    train_loader = DataLoader(dataset=train_data, batch_size=P['batch_size'], shuffle=True, num_workers=1, drop_last=P['mlpenet_drop_last'])
    eval_loader = DataLoader(dataset=eval_data, batch_size=len(eval_data), shuffle=False, num_workers=1)

    save_path = os.path.join(config.save_path, P['mlpenet_architecture_name'])
    if not os.path.exists(os.path.join(save_path, 'runs')):
        os.makedirs(os.path.join(save_path, 'runs'))
    writer = SummaryWriter(os.path.join(save_path, 'runs'))  # For TensorBoard logging

    # Prepare evaluation data
    for i, (x_eval, y_eval, l_eval, h_eval, _) in enumerate(eval_loader):
        print(x_eval.shape)
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        l_eval = l_eval.to(device)
        h_eval = h_eval.to(device)
        y_eval = y_eval[:, 0].unsqueeze(1)  # Only use the first dimension of y_eval
        break  # Evaluation only needs to be set up once

    for epoch in range(init_epoch, num_epochs):
        start_time = time.time()
        for i, (x, y, l, h, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            h = h.to(device)
            y = y[:, 0].unsqueeze(1)  # Only use the first dimension of y

            if optimizer_type == 'adam':
                optimizer.zero_grad()
                outputs = model(x, l, h)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                training_loss = loss.item()
            elif optimizer_type == 'sam':
                # First forward-backward pass
                optimizer.zero_grad()
                enable_running_stats(model)
                outputs = model(x, l, h)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                training_loss = loss.item()

                # Second forward-backward pass
                disable_running_stats(model)
                outputs = model(x, l, h)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                raise ValueError("Invalid optimizer type")

        used_time = time.time() - start_time
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                outputs = model(x_eval, l_eval, h_eval)
                eval_loss = criterion(outputs, y_eval)
            writer.add_scalar('Loss/train', training_loss, epoch + 1)
            writer.add_scalar('Loss/eval', eval_loss.item(), epoch + 1)
            param_num = len(param_name)
            for rr in range(param_num):
                writer.add_scalar(f'Loss/train_{param_name[rr]}', training_loss, epoch + 1)
                writer.add_scalar(f'Loss/eval_{param_name[rr]}', eval_loss.item(), epoch + 1)
            writer.add_scalar('time/train', used_time, epoch + 1)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {training_loss:.4f}/{eval_loss.item():.4f}, Time: {used_time:.1f}s')
            for param_group in optimizer.param_groups:
                print('Learning rate:', param_group['lr'])
            if torch.isnan(eval_loss):
                print('Training failed: Encountered NaN values.')
                return

        # Save the model every epoch
        if (epoch + 1) % 1 == 0:
            save_dict = {
                'network': model.type,
                'init_param': model.init_param,
                'activation': model.activation,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'param_dic': P
            }
            if not os.path.exists(os.path.join(save_path, 'model')):
                os.makedirs(os.path.join(save_path, 'model'))
            torch.save(save_dict, os.path.join(save_path, f'model_{P["mlpenet_architecture_name"]}_epoch_{epoch + 1:05d}.ckpt'))
