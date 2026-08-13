"""
Abstract base class for FLock federated learning models.

This module defines the interface that all FLock models must implement
to participate in federated learning.
"""

from abc import ABC, abstractmethod
from typing import List


class FlockModel(ABC):
    """
    Abstract base class for federated learning models in FLock.
    
    All models participating in FLock federated learning must inherit from this class
    and implement the required abstract methods for training, evaluation, and aggregation.
    """
    
    @abstractmethod
    def init_dataset(self, dataset_path: str) -> None:
        """
        Initialize the model with a dataset.
        
        Args:
            dataset_path: Path to the dataset file.
        """
        pass

    @abstractmethod
    def train(self, parameters: bytes) -> bytes:
        """
        Train the model using the provided parameters.
        
        This method should:
        1. Load the provided model parameters into your model according to your ML framework
        2. Train the model on the local dataset
        3. Return the updated model parameters as bytes
        
        Args:
            parameters: Model parameters as bytes to initialize training.
            
        Returns:
            bytes: Updated model parameters after training.
        """
        pass

    @abstractmethod
    def evaluate(self, parameters: bytes) -> float:
        """
        Evaluate the model using the provided parameters.
        
        This method should:
        1. Load the provided model parameters into your model according to your ML framework
        2. Evaluate the model on the local dataset
        3. Return the evaluation metric (typically loss) as a float
        
        Args:
            parameters: Model parameters as bytes to use for evaluation.
            
        Returns:
            float: Evaluation metric (loss) of the model on the dataset.
        """
        pass

    @abstractmethod
    def aggregate(self, parameters_list: List[bytes]) -> bytes:
        """
        Aggregate multiple sets of model parameters.
        
        This method should:
        1. Take a list of model parameter sets (from different clients)
        2. Aggregate them using an appropriate strategy (typically averaging)
        3. Return the aggregated parameters as bytes
        
        Args:
            parameters_list: List of model parameter sets as bytes from different clients.
            
        Returns:
            bytes: Aggregated model parameters.
        """
        pass
