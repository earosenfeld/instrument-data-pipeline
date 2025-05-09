import numpy as np
import socket
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthernetSimulator:
    """Simulates network-connected test instruments."""
    
    def __init__(self, host: str = 'localhost', port: int = 5025):
        """
        Initialize the ethernet simulator.
        
        Args:
            host (str): Host address for the instrument
            port (int): Port number for the instrument
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.packet_loss_rate = 0.01  # 1% packet loss
        self.latency = 0.05  # 50ms latency
        
    def connect(self) -> bool:
        """
        Simulate connecting to a network instrument.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Simulate connection delay
            time.sleep(self.latency)
            
            # Simulate random connection failures
            if random.random() < 0.05:  # 5% chance of connection failure
                logger.error("Failed to connect to instrument")
                return False
                
            self.connected = True
            logger.info(f"Connected to instrument at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to instrument: {str(e)}")
            return False
            
    def disconnect(self) -> None:
        """Simulate disconnecting from the instrument."""
        if self.connected:
            self.connected = False
            logger.info("Disconnected from instrument")
            
    def read_data(self, duration: float, num_channels: int = 1) -> Dict:
        """
        Simulate reading data from network instruments.
        
        Args:
            duration (float): Duration of data acquisition in seconds
            num_channels (int): Number of channels to read
            
        Returns:
            Dict: Dictionary containing timestamps and data arrays
        """
        if not self.connected:
            raise ConnectionError("Not connected to instrument")
            
        try:
            # Calculate number of samples based on typical network instrument sampling rate
            sampling_rate = 10  # 10 Hz typical for network instruments
            num_samples = int(duration * sampling_rate)
            
            # Generate timestamps
            timestamps = [datetime.now() for _ in range(num_samples)]
            
            # Simulate data with realistic characteristics
            data = []
            for _ in range(num_channels):
                # Generate base signal
                base_signal = np.random.normal(loc=0.5, scale=0.1, size=num_samples)
                
                # Add some periodic variation
                t = np.linspace(0, duration, num_samples)
                periodic = 0.05 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz variation
                
                # Add random noise
                noise = np.random.normal(0, 0.01, num_samples)
                
                # Combine signals
                channel_data = base_signal + periodic + noise
                
                # Simulate packet loss
                mask = np.random.random(num_samples) > self.packet_loss_rate
                channel_data[~mask] = np.nan
                
                data.append(channel_data)
                
            # Simulate network latency
            time.sleep(self.latency)
            
            return {
                'timestamps': timestamps,
                'data': data
            }
            
        except Exception as e:
            logger.error(f"Error reading data from instrument: {str(e)}")
            raise
            
    def send_command(self, command: str) -> str:
        """
        Simulate sending a command to the instrument.
        
        Args:
            command (str): Command to send
            
        Returns:
            str: Instrument response
        """
        if not self.connected:
            raise ConnectionError("Not connected to instrument")
            
        try:
            # Simulate command processing delay
            time.sleep(self.latency)
            
            # Simulate random command failures
            if random.random() < self.packet_loss_rate:
                raise ConnectionError("Command failed due to network error")
                
            # Simulate instrument response
            return f"OK: {command}"
            
        except Exception as e:
            logger.error(f"Error sending command to instrument: {str(e)}")
            raise 