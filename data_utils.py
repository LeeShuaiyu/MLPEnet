import numpy as np
import pickle
import time
from data import generate_param  # Ensure correct import for parameter generation
import signalz  # Assuming signalz is a custom module for generating noise

def sampling_func(M=1000, eta=[0, 5], epsilon=[0, 0.05], alpha=[1.2, 2], N=[100, 400], T=[5, 15], save_name='ou_1d.pkl'):
    """
    Function to generate and save sample data for training/testing.

    Args:
        M (int): Number of samples to generate.
        eta (list): Range of eta parameter for the Ornstein-Uhlenbeck (OU) process.
        epsilon (list): Range of epsilon parameter for the OU process.
        alpha (list): Range of alpha parameter for the Levy noise.
        N (list): Range of the number of discretized steps.
        T (list): Range of time intervals for simulation.
        save_name (str): Name of the file to save the generated data.
    """
    discard_T = 10  # Discard the transient state over [0, discard_T]
    xs = []  # Store trajectories
    ys = []  # Store parameters (eta, epsilon, alpha)
    NUM_FOR_SHOW = 2000  # Interval for showing progress

    # Generate random parameters based on ranges provided
    N = generate_param(N, M)
    T = generate_param(T, M)
    dt = T / N  # Time step for discretization
    discard_T = generate_param(discard_T, M)  # Generate discard time
    discard_N = discard_T / dt  # Number of steps to discard
    eta = generate_param(eta, M)  # Generate eta values
    epsilon = generate_param(epsilon, M)  # Generate epsilon values
    alpha = generate_param(alpha, M)  # Generate alpha values

    time_start = time.time()  # Start timing for process tracking

    # Loop over M samples to generate data
    for idx in range(M):
        if (idx + 1) % NUM_FOR_SHOW == 0:  # Display progress every NUM_FOR_SHOW iterations
            time_end = time.time()
            time_used = time_end - time_start  # Elapsed time
            time_left = time_used / (idx + 1) * (M - idx)  # Estimate remaining time
            print(f"{idx + 1} of {M}, {time_used:.1f}s used {time_left:.1f}s left")

        _N = int(discard_N[idx])  # Number of steps to discard
        _N2 = int(_N + N[idx])  # Total number of steps including discarded ones

        # Generate a single trajectory using the Ornstein-Uhlenbeck process
        x = trajectory_1d_ou(_N2, dt[idx], 0.0, eta[idx], epsilon[idx], alpha[idx], x0=0)
        x = x[_N:]  # Discard the initial transient states
        xs.append(np.array([x]).T)  # Store the trajectory

        # Store the parameters used for this sample
        ys.append([eta[idx], epsilon[idx], alpha[idx]])

    # Save generated data (trajectories and corresponding parameters) to a file
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def trajectory_1d_ou(N, dt, mu, eta, epsilon, alpha, x0=0):
    """
    Generate a 1D Ornstein-Uhlenbeck trajectory with Levy noise.

    Args:
        N (int): Number of steps.
        dt (float): Time step size.
        mu (float): Mean value of the OU process.
        eta (float): Strength of the restoring force in the OU process.
        epsilon (float): Noise coefficient.
        alpha (float): Stability parameter of the Levy noise.
        x0 (float): Initial condition.

    Returns:
        data (ndarray): Generated OU trajectory of size (N,).
    """
    time_ratio = np.power(dt, 1.0 / alpha)  # Scale factor for the noise
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.0)  # Generate alpha-stable Levy noise
    data = np.zeros(N)  # Initialize trajectory data array
    data[0] = x0  # Set initial condition

    # Use Euler's method to integrate the OU process
    for i in range(N - 1):
        # OU process update equation: 
        # x_{i+1} = x_i - eta * (x_i - mu) * dt + epsilon * (dt^(1/alpha)) * Levy noise
        data[i + 1] = data[i] - eta * (data[i] - mu) * dt + epsilon * time_ratio * dL[i]

    return data

def load_data(file_name):
    """
    Load data from a pickle file.

    Args:
        file_name (str): Path to the file.

    Returns:
        data: Loaded data.
    """
    with open(file_name, "rb") as fp:
        data = pickle.load(fp)
    return data

def save_data(file_name, data):
    """
    Save data to a pickle file.

    Args:
        file_name (str): Path to save the data.
        data: Data to save.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
