import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_frft.layer import DFrFTLayer, FrFTLayer
from torch_frft.dfrft_module import dfrft

class FeatureExtractor_STFRFT(nn.Module):
    """
    Feature extractor that performs feature extraction using windowing and FrFT.
    This class takes in time-series data and applies windowing, followed by FrFT.
    """

    def __init__(self, window_size, frft_order, overlap=128, order_is_trainable=None, only_use_real=None):
        super(FeatureExtractor_STFRFT, self).__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.only_use_real = only_use_real
        if order_is_trainable:
            self.order1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.order1 =frft_order

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, seq_length)
        """
        batch_size, channels, seq_length = x.size()
        # if isinstance(self.order1, nn.Parameter):  # Check if 'a' is learnable
        #     print(f"Learnable FrFT order: {self.order1.item()}")

        # Step 1: Apply windowing with overlap
        XW = self._apply_window(x)

        # Step 2: Apply FrFT to each windowed segment (row vector)
        XFrFT = dfrft(XW, self.order1, dim=-1)  # Shape: (batch_size, channels, num_windows, window_size)
        # XFrFT_Mix =torch.abs(XFrFT)
        XFrFT_real = XFrFT.real
        XFrFT_imag = XFrFT.imag
        if self.only_use_real:
            XFrFT_Mix = XFrFT_real
        else:
            XFrFT_Mix = torch.concat([XFrFT_real, XFrFT_imag], dim=1)
        return XFrFT_Mix

    def _apply_window(self, x):
        """
        Apply sliding window with overlap to the input tensor (batch_size, channels, seq_length).
        Returns a tensor of shape (batch_size, channels, num_windows, window_size).
        """
        # Calculate the step size (hop length) based on the overlap
        step = self.window_size - self.overlap
        batch_size, channels, seq_length = x.size()

        # Create a window function (e.g., Hann window)
        window = torch.hann_window(self.window_size).to(x.device)

        # Use unfold to create overlapping windows
        XW = x.unfold(dimension=2, size=self.window_size, step=step)  # Shape: (batch_size, channels, num_windows, window_size)

        # Apply the window function to each segment
        XW = XW * window  # Broadcasting the window across all segments

        return XW  # Shape: (batch_size, channels, num_windows, window_size)

class FeatureExtractor_STFT(nn.Module):
    """
    Feature extractor that performs feature extraction using windowing and FFT.
    This class takes in time-series data and applies windowing, followed by FFT.
    """

    def __init__(self, window_size, overlap=128):
        super(FeatureExtractor_STFT, self).__init__()
        self.window_size = window_size
        self.overlap = overlap

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, seq_length)
        """
        batch_size, channels, seq_length = x.size()

        # Step 1: Apply windowing with overlap
        XW = self._apply_window(x)

        # Step 2: Apply FFT to each windowed segment
        XFFT = torch.fft.fft(XW, dim=-1)  # Shape: (batch_size, channels, num_windows, window_size)
        XFFT_real = XFFT.real
        XFFT_imag = XFFT.imag
        XFFT_Mix = torch.concat([XFFT_real, XFFT_imag], dim=1)

        return XFFT_Mix

    def _apply_window(self, x):
        """
        Apply sliding window with overlap to the input tensor (batch_size, channels, seq_length).
        Returns a tensor of shape (batch_size, channels, num_windows, window_size).
        """
        # Calculate the step size (hop length) based on the overlap
        step = self.window_size - self.overlap
        batch_size, channels, seq_length = x.size()

        # Create a window function (e.g., Hann window)
        window = torch.hann_window(self.window_size).to(x.device)

        # Use unfold to create overlapping windows
        XW = x.unfold(dimension=2, size=self.window_size,
                      step=step)  # Shape: (batch_size, channels, num_windows, window_size)

        # Apply the window function to each segment
        XW = XW * window  # Broadcasting the window across all segments

        return XW  # Shape: (batch_size, channels, num_windows, window_size)

class FeatureExtractor_spec(nn.Module):
    """
    Feature extractor that performs feature extraction using windowing and FFT.
    This class takes in time-series data and applies windowing, followed by FFT.
    """

    def __init__(self, window_size, overlap=128):
        super(FeatureExtractor_spec, self).__init__()
        self.window_size = window_size
        self.overlap = overlap

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, seq_length)
        """
        batch_size, channels, seq_length = x.size()

        # Step 1: Apply windowing with overlap
        XW = self._apply_window(x)

        # Step 2: Apply FFT to each windowed segment
        XFFT = torch.fft.fft(XW, dim=-1)  # Shape: (batch_size, channels, num_windows, window_size)
        XFFT_Mix = torch.abs(XFFT)


        return XFFT_Mix

    def _apply_window(self, x):
        """
        Apply sliding window with overlap to the input tensor (batch_size, channels, seq_length).
        Returns a tensor of shape (batch_size, channels, num_windows, window_size).
        """
        # Calculate the step size (hop length) based on the overlap
        step = self.window_size - self.overlap
        batch_size, channels, seq_length = x.size()

        # Create a window function (e.g., Hann window)
        window = torch.hann_window(self.window_size).to(x.device)

        # Use unfold to create overlapping windows
        XW = x.unfold(dimension=2, size=self.window_size,
                      step=step)  # Shape: (batch_size, channels, num_windows, window_size)

        # Apply the window function to each segment
        XW = XW * window  # Broadcasting the window across all segments

        return XW  # Shape: (batch_size, channels, num_windows, window_size)