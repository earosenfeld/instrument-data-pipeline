import numpy as np
import random
import time
from datetime import datetime
import logging
from typing import Dict, List, Union, Tuple
import socket
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DAQSimulator:
    """Simulates different data acquisition sources for testing."""
    
    def __init__(self):
        self.sampling_rate = 1000  # Hz
        self.buffer_size = 1024
        self.connected = False
        
    def simulate_daq_data(self, 
                         source_type: str, 
                         duration: float = 1.0,
                         num_channels: int = 1) -> Dict[str, Union[List[float], List[datetime]]]:
        """
        Simulate data acquisition from different sources.
        
        Args:
            source_type (str): Type of data source ('analog', 'digital', 'ethernet')
            duration (float): Duration of data collection in seconds
            num_channels (int): Number of channels to simulate
            
        Returns:
            Dict containing timestamps and channel data
        """
        if source_type not in ['analog', 'digital', 'ethernet']:
            raise ValueError(f"Unsupported source type: {source_type}")
            
        num_samples = int(duration * self.sampling_rate)
        timestamps = [datetime.now() for _ in range(num_samples)]
        
        if source_type == 'analog':
            # Simulate analog DAQ with noise and drift
            data = self._simulate_analog_data(num_samples, num_channels)
        elif source_type == 'digital':
            # Simulate digital I/O with state changes
            data = self._simulate_digital_data(num_samples, num_channels)
        else:  # ethernet
            # Simulate network data with packet loss and latency
            data = self._simulate_ethernet_data(num_samples, num_channels)
            
        return {
            'timestamps': timestamps,
            'data': data
        }
    
    def _simulate_analog_data(self, 
                            num_samples: int, 
                            num_channels: int) -> List[List[float]]:
        """Simulate analog DAQ data with realistic characteristics."""
        data = []
        for _ in range(num_channels):
            # Generate base signal
            t = np.linspace(0, 2*np.pi, num_samples)
            base_signal = np.sin(t) * 5.0  # 5V amplitude
            
            # Add noise
            noise = np.random.normal(0, 0.1, num_samples)
            
            # Add drift
            drift = np.linspace(0, 0.5, num_samples)
            
            # Combine signals
            channel_data = base_signal + noise + drift
            data.append(channel_data.tolist())
            
        return data
    
    def _simulate_digital_data(self, 
                             num_samples: int, 
                             num_channels: int) -> List[List[int]]:
        """Simulate digital I/O data with state changes."""
        data = []
        for _ in range(num_channels):
            # Generate random state changes
            states = [random.randint(0, 1) for _ in range(num_samples)]
            
            # Add some debouncing
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    # Simulate contact bounce
                    for j in range(1, 5):
                        if i+j < len(states):
                            states[i+j] = states[i]
            
            data.append(states)
            
        return data
    
    def _simulate_ethernet_data(self, 
                              num_samples: int, 
                              num_channels: int) -> List[List[float]]:
        """Simulate network data with realistic characteristics."""
        data = []
        for _ in range(num_channels):
            # Generate base data
            base_data = np.random.normal(0, 1, num_samples)
            
            # Simulate packet loss (5% loss rate)
            packet_loss = np.random.random(num_samples) < 0.05
            base_data[packet_loss] = np.nan
            
            # Add network latency (random delays)
            latency = np.random.normal(0.02, 0.005, num_samples)  # 20ms ± 5ms
            base_data = np.roll(base_data, int(latency[0] * self.sampling_rate))
            
            data.append(base_data.tolist())
            
        return data
    
    def connect(self, source_type: str) -> bool:
        """Simulate connecting to a data source."""
        try:
            # Simulate connection delay
            time.sleep(0.5)
            self.connected = True
            logger.info(f"Connected to {source_type} source")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {source_type} source: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Simulate disconnecting from a data source."""
        if self.connected:
            time.sleep(0.2)  # Simulate disconnection delay
            self.connected = False
            logger.info("Disconnected from data source")
    
    def read_data(self, 
                 source_type: str, 
                 duration: float = 1.0,
                 num_channels: int = 1) -> Dict[str, Union[List[float], List[datetime]]]:
        """Read data from the simulated source."""
        if not self.connected:
            raise ConnectionError("Not connected to data source")
            
        return self.simulate_daq_data(source_type, duration, num_channels) 