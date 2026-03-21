from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseIndicator(ABC):
    """
    Base class for all technical indicators.
    All indicators must inherit from this class and implement the calculate method.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the indicator with optional parameters.
        
        Args:
            **kwargs: Indicator-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values from input data.
        
        Args:
            data: Input DataFrame containing OHLCV data (Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with calculated indicator values
        """
        pass

    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate the input data has required columns.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        return True

    @property
    def name(self) -> str:
        """Return the indicator name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return string representation of the indicator."""
        return f"{self.name}({self.params})"
