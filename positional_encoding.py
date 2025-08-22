import logging
import numpy as np
import torch
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding:
    """
    Positional encoding implementation based on the Transformer paper.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.
        device (torch.device): The device to use for computations.

    Attributes:
        pe (torch.Tensor): The positional encoding tensor.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        # Create the positional encoding tensor
        self.pe = self._create_positional_encoding()

    def _create_positional_encoding(self) -> torch.Tensor:
        """
        Create the positional encoding tensor.

        Returns:
            torch.Tensor: The positional encoding tensor.
        """
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with positional encoding applied.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class LearnedPositionalEncoding:
    """
    Learned positional encoding implementation.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.
        device (torch.device): The device to use for computations.

    Attributes:
        pe (torch.nn.Module): The learned positional encoding module.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        # Create the learned positional encoding module
        self.pe = torch.nn.Embedding(max_len, d_model)
        self.pe.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / d_model))
        self.pe.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with learned positional encoding applied.
        """
        pe = self.pe(torch.arange(x.size(1), device=self.device))
        pe = pe.unsqueeze(0)
        x = x + pe
        return x


class SinePositionalEncoding:
    """
    Sine positional encoding implementation.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.
        device (torch.device): The device to use for computations.

    Attributes:
        pe (torch.Tensor): The sine positional encoding tensor.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        # Create the sine positional encoding tensor
        self.pe = self._create_sine_positional_encoding()

    def _create_sine_positional_encoding(self) -> torch.Tensor:
        """
        Create the sine positional encoding tensor.

        Returns:
            torch.Tensor: The sine positional encoding tensor.
        """
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the sine positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with sine positional encoding applied.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PositionalEncodingFactory:
    """
    Factory class for creating positional encoding instances.

    Attributes:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum length of the input sequence.
        dropout (float): The dropout rate.
        device (torch.device): The device to use for computations.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

    def create_positional_encoding(self, encoding_type: str = "positional") -> PositionalEncoding:
        """
        Create a positional encoding instance.

        Args:
            encoding_type (str): The type of positional encoding to create. Defaults to "positional".

        Returns:
            PositionalEncoding: The created positional encoding instance.
        """
        if encoding_type == "positional":
            return PositionalEncoding(self.d_model, self.max_len, self.dropout, self.device)
        elif encoding_type == "learned":
            return LearnedPositionalEncoding(self.d_model, self.max_len, self.dropout, self.device)
        elif encoding_type == "sine":
            return SinePositionalEncoding(self.d_model, self.max_len, self.dropout, self.device)
        else:
            raise ValueError("Invalid encoding type")


if __name__ == "__main__":
    # Example usage
    d_model = 512
    max_len = 1024
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factory = PositionalEncodingFactory(d_model, max_len, dropout, device)
    pe = factory.create_positional_encoding("positional")
    x = torch.randn(1, 10, d_model)
    encoded_x = pe.forward(x)
    print(encoded_x.shape)