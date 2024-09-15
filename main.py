import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

def ewma_detection(alpha: float, y: float, ewma: float | None, rollingStd: deque, threshFac: int) -> tuple[float, bool]:
    """
    Perform Exponentially Weighted Moving Average (EWMA) detection on a given data point.
    
    Parameters:
    alpha (float): Smoothing factor for EWMA, determines the weight given to recent data points.
    y (float): Current data point to be evaluated.
    ewma (float | None): Previous EWMA value. If None, initialize EWMA with the current data point.
    rollingStd (deque): Rolling history of standard deviations from previous data points.
    threshFac (int): Multiplier for the standard deviation to set the anomaly detection threshold.
    
    Returns:
    tuple[float, bool]: Updated EWMA value and a boolean indicating if the current data point is an anomaly.
    """
    
    # Initialize EWMA if no previous value exists
    if ewma is None:
        ewma = y
    else:
        # Calculate the new EWMA value using the current data point
        ewma = alpha * y + (1 - alpha) * ewma
    
    # Add the deviation of the current data point from the EWMA to the rolling standard deviation history
    rollingStd.append(abs(y - ewma))
    
    # Compute the standard deviation from the rolling history
    std_dev = np.std(rollingStd) if len(rollingStd) > 1 else 0
    
    # Define the threshold for anomaly detection
    threshold = threshFac * std_dev
    
    # Determine if the current data point is an anomaly
    is_anomaly = abs(y - ewma) > threshold
    
    return ewma, is_anomaly

def simulateDataStream(t : int) -> float:
    """
    Simulate a data stream by generating a data point with a combination of regular patterns, 
    seasonal variations, and random noise.
    
    Parameters:
    t (int or float): The time index, which should be a numerical value.
    
    Returns:
    float: A data point combining regular patterns, seasonal variations, and noise.
           Returns NaN if an error occurs.
    Raises:
    ValueError: If the input 't' is not a numerical value.
    Exception: For any unexpected errors during the simulation.
    """
    
    try:
        if not isinstance(t, (int, float)):
            raise ValueError("Time 't' must be a numerical value.")
        
        # Generate a regular pattern based on a sine wave
        reg_pattern = np.sin(2 * np.pi * 0.05 * t)
        # Generate a slower, longer-term seasonal variation
        seasonal_var = np.sin(2 * np.pi * 0.01 * t)
        # Add random noise to the data
        noise = np.random.normal(0, 0.5)
        
        # Combine the regular pattern, seasonal variation, and noise
        RSN = reg_pattern + seasonal_var + noise
        
        return RSN

    except ValueError as e:
        print(f"ValueError: {e}")
        return np.nan
    except Exception as e:
        print(f"Unexpected error in simulateDataStream: {e}")
        return np.nan

class StreamPlot:
    """
    Class for plotting the real-time data stream, including detected anomalies.
    """
    
    def __init__(self):
        self.time = []
        self.data = []
        self.x_anomaly = []
        self.y_anomaly = []
        self.fig, self.ax = plt.subplots()

        # Set plot labels and title
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Real-time Data Stream with Patterns, Seasonal Elements, and Noise')
        self.ax.set_ylim(-3, 3)
    
    def plot(self, t, d, a=None):
        """
        Update the plot with a new data point and highlight anomalies if detected.
        
        Parameters:
        t (int): Time index for the data point.
        d (float): Data point value.
        a (float or None): Data point value if an anomaly is detected. None if no anomaly.
        """
        # Append data for plotting
        self.time.append(t)
        self.data.append(d)

        if a is not None:
            self.x_anomaly.append(t)
            self.y_anomaly.append(a)
            # Plot anomalies
            self.ax.plot(self.x_anomaly, self.y_anomaly, 'bo', label='Anomaly Detected')
        
        # Plot data stream
        self.ax.plot(self.time, self.data, 'r-', label='Data Stream')
        self.ax.set_xlim(0, t+1)
        
        plt.ion()  # Enable interactive mode for real-time updates
        plt.pause(0.01)
        plt.show()

class DataStream:
    """
    Class for managing the data stream, applying EWMA detection, and plotting results.
    """
    
    def __init__(self, alpha: float, threshFac: int):
        self.alpha = alpha
        self.threshFac = threshFac
        self.ewma = None
        self.rollingStd = deque(maxlen=10)
        self.t = 0
        self.plot = StreamPlot()
        self.y_pred = []
        self.y_true = []
    
    def update(self, data: float) -> None:
        """
        Update the data stream with a new data point and detect anomalies.
        
        Parameters:
        data (float): The new data point to be evaluated.
        """
        self.ewma, is_anomaly = ewma_detection(self.alpha, data, self.ewma, self.rollingStd, self.threshFac)
        self.t += 1
        
        # Record the anomaly detection results
        self.y_pred.append(1 if is_anomaly else 0)
        self.y_true.append(1 if is_anomaly else 0)

        if is_anomaly:
            print(f"Anomaly detected at time {self.t} with data point {data}")
            self.plot.plot(self.t, data, data)
        else:
            self.plot.plot(self.t, data, None)
    
    def simulateDataStream(self) -> None:
        """
        Continuously simulate and process data points from the data stream.
        """
        while True:
            data = simulateDataStream(self.t)
            # Simulate anomalies every 100 time steps
            self.update(data)
            time.sleep(0.1)

if __name__ == '__main__':
    # Initialize and run the data stream simulation
    stream = DataStream(alpha=0.2, threshFac=3)
    stream.simulateDataStream()
    
    plt.ioff()  # Disable interactive mode to finalize the plot
    plt.show()
