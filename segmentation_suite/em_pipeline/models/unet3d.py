"""
3D U-Net architecture for volumetric segmentation.

This implementation is based on the architecture used in:
- Funke et al., "Large Scale Image Segmentation with Structured Loss Based on Segmentation Aware Affinities"
- Sheridan et al., "Local Shape Descriptors for Neuron Segmentation"

Features:
- Configurable depth and feature channels
- Group normalization (works better than batch norm for small batches)
- Optional residual connections
- Support for anisotropic data (different downsampling in z)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv3D -> GroupNorm -> ReLU (x2).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Convolution kernel size.
    padding : int or tuple, optional
        Padding size. Defaults to kernel_size // 2 for same padding.
    num_groups : int
        Number of groups for GroupNorm. Default: 8.
    use_residual : bool
        Whether to add residual connection. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        padding: Optional[Union[int, Tuple[int, int, int]]] = None,
        num_groups: int = 8,
        use_residual: bool = False,
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        # Ensure num_groups divides out_channels
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        else:
            self.residual_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = out + residual

        out = self.relu2(out)
        return out


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool3D -> ConvBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pool_size : tuple
        Pooling kernel size. Default: (1, 2, 2) for anisotropic data.
    kernel_size : int or tuple
        Convolution kernel size.
    num_groups : int
        Number of groups for GroupNorm.
    use_residual : bool
        Whether to use residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int, int] = (1, 2, 2),
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        num_groups: int = 8,
        use_residual: bool = False,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_size)
        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, num_groups=num_groups, use_residual=use_residual
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Concat skip -> ConvBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels (from previous layer).
    skip_channels : int
        Number of channels from skip connection.
    out_channels : int
        Number of output channels.
    scale_factor : tuple
        Upsampling scale factor. Default: (1, 2, 2) for anisotropic data.
    kernel_size : int or tuple
        Convolution kernel size.
    num_groups : int
        Number of groups for GroupNorm.
    use_residual : bool
        Whether to use residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: Tuple[int, int, int] = (1, 2, 2),
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        num_groups: int = 8,
        use_residual: bool = False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_block = ConvBlock(
            in_channels + skip_channels, out_channels, kernel_size, num_groups=num_groups, use_residual=use_residual
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)

        # Handle size mismatch from rounding
        if x.shape != skip.shape:
            diff_z = skip.shape[2] - x.shape[2]
            diff_y = skip.shape[3] - x.shape[3]
            diff_x = skip.shape[4] - x.shape[4]
            x = F.pad(x, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
                diff_z // 2, diff_z - diff_z // 2,
            ])

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Architecture:
    - Encoder: series of DownBlocks with increasing channels
    - Bottleneck: ConvBlock at lowest resolution
    - Decoder: series of UpBlocks with skip connections

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale EM).
    base_features : int
        Number of features in first layer. Doubles at each level.
    depth : int
        Number of encoder/decoder levels. Default: 4.
    out_channels : int, optional
        Number of output channels. If None, returns features before final conv.
    pool_sizes : list of tuple, optional
        Pooling sizes for each level. Default: anisotropic (1,2,2).
    kernel_size : int or tuple
        Convolution kernel size. Default: 3.
    num_groups : int
        Groups for GroupNorm. Default: 8.
    use_residual : bool
        Use residual connections. Default: True.

    Example
    -------
    >>> model = UNet3D(in_channels=1, base_features=32, depth=4, out_channels=3)
    >>> x = torch.randn(1, 1, 64, 256, 256)
    >>> out = model(x)
    >>> print(out.shape)  # (1, 3, 64, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        out_channels: Optional[int] = None,
        pool_sizes: Optional[List[Tuple[int, int, int]]] = None,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        num_groups: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_features = base_features
        self.depth = depth
        self.out_channels = out_channels

        # Default anisotropic pooling (common for EM data: 4nm xy, 40nm z)
        if pool_sizes is None:
            # First levels: pool only xy; later levels: pool all
            pool_sizes = []
            for i in range(depth):
                if i < depth - 2:
                    pool_sizes.append((1, 2, 2))  # Anisotropic
                else:
                    pool_sizes.append((2, 2, 2))  # Isotropic at coarse levels

        self.pool_sizes = pool_sizes

        # Initial convolution
        self.initial_conv = ConvBlock(
            in_channels, base_features, kernel_size, num_groups=num_groups, use_residual=use_residual
        )

        # Encoder
        self.encoders = nn.ModuleList()
        encoder_channels = [base_features]

        for i in range(depth):
            in_ch = encoder_channels[-1]
            out_ch = in_ch * 2
            self.encoders.append(
                DownBlock(in_ch, out_ch, pool_sizes[i], kernel_size, num_groups, use_residual)
            )
            encoder_channels.append(out_ch)

        # Bottleneck
        bottleneck_ch = encoder_channels[-1]
        self.bottleneck = ConvBlock(
            bottleneck_ch, bottleneck_ch, kernel_size, num_groups=num_groups, use_residual=use_residual
        )

        # Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = [bottleneck_ch]

        for i in range(depth - 1, -1, -1):
            in_ch = decoder_channels[-1]
            skip_ch = encoder_channels[i]  # Skip from encoder
            out_ch = skip_ch
            scale = pool_sizes[i]

            self.decoders.append(
                UpBlock(in_ch, skip_ch, out_ch, scale, kernel_size, num_groups, use_residual)
            )
            decoder_channels.append(out_ch)

        # Final output
        self.final_features = decoder_channels[-1]
        if out_channels is not None:
            self.final_conv = nn.Conv3d(self.final_features, out_channels, 1)
        else:
            self.final_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, D, H, W).
        """
        # Initial conv
        x = self.initial_conv(x)

        # Encoder with skip connections
        skips = [x]
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i  # Skip from corresponding encoder level
            x = decoder(x, skips[skip_idx])

        # Final convolution
        if self.final_conv is not None:
            x = self.final_conv(x)

        return x

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape for a given input shape.

        Parameters
        ----------
        input_shape : tuple
            Input shape (D, H, W) without batch and channel dims.

        Returns
        -------
        tuple
            Output shape (D, H, W).
        """
        # For a fully-convolutional network with same padding,
        # output spatial dimensions equal input spatial dimensions
        return input_shape

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNet25D(nn.Module):
    """
    2.5D U-Net for faster inference on anisotropic data.

    Processes each z-slice with a 2D U-Net and aggregates features
    across a small z-window. Much faster than full 3D but captures
    some z-context.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    base_features : int
        Number of features in first layer.
    depth : int
        Number of encoder/decoder levels.
    out_channels : int
        Number of output channels.
    z_context : int
        Number of z-slices for context. Default: 3.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        out_channels: int = 3,
        z_context: int = 3,
    ):
        super().__init__()
        self.z_context = z_context

        # 2D encoder/decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Modified to accept z_context * in_channels
        encoder_ch = [z_context * in_channels]
        for i in range(depth):
            in_ch = encoder_ch[-1]
            out_ch = base_features * (2 ** i)
            self.encoder.append(self._make_2d_block(in_ch, out_ch))
            encoder_ch.append(out_ch)

        # Decoder with skip connections
        for i in range(depth - 1, -1, -1):
            in_ch = encoder_ch[i + 1]
            skip_ch = encoder_ch[i]
            out_ch = encoder_ch[i]
            self.decoder.append(self._make_2d_up_block(in_ch, skip_ch, out_ch))

        self.final_conv = nn.Conv2d(encoder_ch[0], out_channels, 1)

    def _make_2d_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create 2D conv block with pooling."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _make_2d_up_block(self, in_ch: int, skip_ch: int, out_ch: int) -> nn.Module:
        """Create 2D upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output (N, out_channels, D, H, W).
        """
        N, C, D, H, W = x.shape
        outputs = []

        # Process each z-slice with context
        pad = self.z_context // 2
        x_padded = F.pad(x, (0, 0, 0, 0, pad, pad), mode='replicate')

        for z in range(D):
            # Extract z-context window and reshape to 2D
            z_window = x_padded[:, :, z:z + self.z_context]  # (N, C, z_ctx, H, W)
            z_window = z_window.reshape(N, -1, H, W)  # (N, C*z_ctx, H, W)

            # 2D forward pass
            skips = []
            for encoder in self.encoder:
                skips.append(z_window)
                z_window = encoder(z_window)

            for i, decoder in enumerate(self.decoder):
                skip = skips[-(i + 1)]
                # Resize if needed
                if z_window.shape[2:] != skip.shape[2:]:
                    z_window = F.interpolate(z_window, size=skip.shape[2:], mode='bilinear', align_corners=False)
                z_window = torch.cat([z_window, skip], dim=1)
                z_window = decoder(z_window)

            out_slice = self.final_conv(z_window)
            outputs.append(out_slice)

        # Stack along z
        return torch.stack(outputs, dim=2)


__all__ = ['UNet3D', 'UNet25D', 'ConvBlock', 'DownBlock', 'UpBlock']
