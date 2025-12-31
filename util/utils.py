import numpy as np
import matplotlib.pyplot as plt


def plot_constellation(data, title='Constellation Diagram', point_size=1):
    """
    Draw a constellation diagram.
    Parameters:
    data (np.ndarray): Input signal data, with a shape of (2, N), where the first row is the I channel and the second row is the Q channel.
    title (str): The title of the graph.
    point_size (int): The size of the points to be drawn.
    x_limits (tuple): The display range of the x-axis.
    y_limits (tuple): The display range of the y-axis.
    """
    # I-channel and Q-channel data
    I = data[0, :]
    Q = data[1, :]
    # Draw a constellation map
    plt.figure(figsize=(8, 8))
    plt.scatter(I, Q, color='blue', s=point_size)
    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True)
    plt.axis('square')
    # Display graphics
    plt.show()


def plot_iq_combined(data, title='Combined IQ Curves', sample_rate=1):
    """
    Plot the curves of the I-channel and Q-channel signals on the same graph.
    Parameters:
    data (np.ndarray): Input signal data with a shape of (2, N), where the first row is the I-channel and the second row is the Q-channel.
    title (str): The title of the graph.
    sample_rate (int): The sampling rate, used for calculating the time scale of the x-axis.
    """
    # Check the data shape
    if data.shape[0] != 2:
        raise ValueError("The data shape must b (2, N)")
    # Extract I-channel and Q-channel data
    I = data[0, :]
    Q = data[1, :]
    time_axis = np.arange(I.size) / sample_rate
    # 创Create graphics
    plt.figure(figsize=(10, 6))
    plt.title(title)
    # Draw I-channel
    plt.plot(time_axis, I, label='In-phase (I)', color='blue')
    plt.plot(time_axis, Q, label='Quadrature (Q)', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()  # Add a legend
    plt.show()
