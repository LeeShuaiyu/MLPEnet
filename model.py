import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLPenet(nn.Module):
    """
    Definition of the MLPenet model, which combines CNN, LSTM, and fully connected layers.
    """

    def __init__(self, in_dim, hidden_dim, n_layer, conv_out_channels=50, conv_kernel_size=3, activation='leakyrelu',
                 operation='cnn', split_ratio=0.5):
        super(MLPenet, self).__init__()
        self.type = 'MLPenet'
        self.operation = operation  # The operation type ('cnn', 'split', or 'truncate')
        self.split_ratio = split_ratio  # Ratio used for 'split' and 'truncate' operations

        self.activation = activation  # Activation function to use
        self.init_param = [in_dim, hidden_dim, n_layer, conv_out_channels,
                           conv_kernel_size]  # Store model parameters for initialization
        print('init_param', self.init_param)
        self.n_layer = n_layer  # Number of LSTM layers
        self.hidden_dim = hidden_dim  # Size of LSTM hidden layer

        # Define input processing based on the operation type
        if self.operation == 'cnn':
            # 1D CNN is used to process the input if the operation is 'cnn'
            self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=conv_out_channels, kernel_size=conv_kernel_size,
                                    stride=1, padding=conv_kernel_size // 2)
            lstm_input_dim = conv_out_channels  # Output dimension of CNN serves as LSTM input
        elif self.operation == 'split':
            # In 'split' mode, the input dimension is doubled after splitting
            lstm_input_dim = in_dim * 2
        elif self.operation == 'truncate':
            # In 'truncate' mode, the input dimension remains the same
            lstm_input_dim = in_dim
        else:
            raise ValueError("Invalid operation type. Choose from 'cnn', 'split', 'truncate'.")

        # LSTM layer definition
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, n_layer, batch_first=True)

        # Fully connected layers
        self.nn_size = 20
        self.linears = nn.ModuleList()  # List to store fully connected layers
        for k in range(3):
            if k == 0:
                # First layer connects LSTM output with an additional input (hidden_dim + 1)
                self.linears.append(nn.Linear(hidden_dim + 1, self.nn_size))
                self.linears.append(nn.BatchNorm1d(self.nn_size))  # Batch normalization
            else:
                self.linears.append(nn.Linear(self.nn_size, self.nn_size))  # Subsequent layers maintain the same size
            # Activation functions
            if self.activation == 'elu':
                self.linears.append(nn.ELU())
            elif self.activation == 'leakyrelu':
                self.linears.append(nn.LeakyReLU())
            elif self.activation == 'tanh':
                self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(self.nn_size, 1))  # Output layer, reduces to 1 dimension

        # Constant tensor used in forward calculation
        self.ones = torch.ones((1, self.hidden_dim)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.init_weights()  # Initialize weights

    def _init_lstm(self, weight):
        """
        Orthogonally initialize LSTM weights for stability.
        """
        for w in weight.chunk(4, 0):
            init.orthogonal_(w)

    def init_weights(self):
        """
        Initialize all model weights, including LSTM and fully connected layers.
        """
        self._init_lstm(self.lstm.weight_ih_l0)  # Initialize input-hidden weights for LSTM
        self._init_lstm(self.lstm.weight_hh_l0)  # Initialize hidden-hidden weights for LSTM
        self.lstm.bias_ih_l0.data.zero_()  # Zero the input-hidden bias
        self.lstm.bias_hh_l0.data.zero_()  # Zero the hidden-hidden bias

    def forward(self, x, l, h):
        """
        Forward pass for the model.

        Args:
            x (tensor): Input tensor, representing the trajectory data.
            l (tensor): Tensor representing the lengths of the trajectories.
            h (tensor): Tensor representing auxiliary input (such as time span).

        Returns:
            x (tensor): Final output after passing through CNN (optional), LSTM, and fully connected layers.
        """
        if self.operation == 'cnn':
            # Apply 1D CNN to the full input
            x_conv = x.transpose(1, 2)  # Transpose to (batch_size, in_dim, seq_len) for Conv1d
            x_conv = self.conv1d(x_conv)
            x_conv = x_conv.transpose(1, 2)  # Transpose back to (batch_size, seq_len, hidden_dim)
            x = x_conv
        elif self.operation == 'split':
            # Split the input and concatenate the segments
            split_size = int(x.size(1) * self.split_ratio)
            remainder_size = x.size(1) - split_size
            if remainder_size > 0:
                # If there's remainder, pad the second part
                x1, x2 = x[:, :split_size, :], x[:, split_size:, :]
                x2 = F.pad(x2, (0, 0, 0, split_size - remainder_size))
            else:
                x1, x2 = x[:, :split_size, :], torch.zeros_like(x[:, :split_size, :])
            x = torch.cat((x1, x2), dim=2)
        elif self.operation == 'truncate':
            # Truncate the input to a smaller size
            truncate_size = int(x.size(1) * self.split_ratio)
            if truncate_size > x.size(1):
                # If the truncate size is larger than the input, pad the input
                x = F.pad(x, (0, 0, 0, truncate_size - x.size(1)))
            else:
                # Otherwise, simply truncate the input
                x = x[:, :truncate_size, :]

        # LSTM processing
        x, (hn, cn) = self.lstm(x)

        # Fully connected network
        x = torch.sum(x, dim=1)  # Summing over the sequence length dimension
        x = torch.div(x, torch.mm(l, self.ones))  # Normalize by trajectory length
        x = torch.cat([x, h], dim=1)  # Concatenate with the auxiliary input
        for m in self.linears:
            x = m(x)  # Pass through the fully connected layers

        return x
