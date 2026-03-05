"""
Joint LSD-FFN Model.

Single network with shared encoder and multiple prediction heads:
- LSD head: Local Shape Descriptors (affinities + shape features)
- FFN head: Boundary/continuation prediction (FFN-style)

The shared encoder learns features useful for both tasks,
potentially achieving better segmentation than either alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import UNet3D, ConvBlock


@dataclass
class JointModelConfig:
    """
    Configuration for joint LSD-FFN model.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for grayscale EM).
    base_features : int
        Base feature channels.
    depth : int
        Encoder depth.
    num_lsd_affinities : int
        Number of LSD affinity channels.
    num_lsd_descriptors : int
        Number of LSD descriptor channels.
    num_ffn_outputs : int
        Number of FFN boundary channels.
    use_residual : bool
        Use residual connections.
    """

    in_channels: int = 1
    base_features: int = 32
    depth: int = 4
    num_lsd_affinities: int = 12  # 3 short + 9 long
    num_lsd_descriptors: int = 10
    num_ffn_outputs: int = 3  # x, y, z boundaries
    use_residual: bool = True


@dataclass
class JointModelOutput:
    """
    Output container for joint model.

    Attributes
    ----------
    lsd_affinities : torch.Tensor
        LSD affinity predictions (N, 12, D, H, W).
    lsds : torch.Tensor
        Local shape descriptors (N, 10, D, H, W).
    ffn_boundaries : torch.Tensor
        FFN-style boundary predictions (N, 3, D, H, W).
    shared_features : torch.Tensor, optional
        Shared encoder features for analysis.
    """

    lsd_affinities: torch.Tensor
    lsds: torch.Tensor
    ffn_boundaries: torch.Tensor
    shared_features: Optional[torch.Tensor] = None


class LSDHead(nn.Module):
    """LSD prediction head with affinities and descriptors."""

    def __init__(
        self,
        in_features: int,
        num_affinities: int = 12,
        num_descriptors: int = 10,
    ):
        super().__init__()

        hidden = in_features // 2

        # Affinity prediction
        self.affinity_conv = nn.Sequential(
            ConvBlock(in_features, hidden, use_residual=True),
            nn.Conv3d(hidden, num_affinities, 1),
        )

        # Descriptor prediction
        self.descriptor_conv = nn.Sequential(
            ConvBlock(in_features, hidden, use_residual=True),
            nn.Conv3d(hidden, num_descriptors, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict LSD outputs.

        Returns
        -------
        tuple
            (affinities, descriptors) tensors.
        """
        affinities = torch.sigmoid(self.affinity_conv(features))
        descriptors = torch.tanh(self.descriptor_conv(features))
        return affinities, descriptors


class FFNHead(nn.Module):
    """
    FFN-style boundary prediction head.

    Predicts boundary probability in each direction,
    similar to FFN's continuation prediction but simpler.
    """

    def __init__(
        self,
        in_features: int,
        num_outputs: int = 3,
    ):
        super().__init__()

        hidden = in_features // 2

        # Multi-scale boundary detection
        self.conv1 = ConvBlock(in_features, hidden, kernel_size=3, use_residual=True)
        self.conv2 = ConvBlock(hidden, hidden, kernel_size=5, use_residual=True)

        # Combine scales
        self.combine = nn.Sequential(
            nn.Conv3d(hidden * 2, hidden, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
        )

        # Output
        self.output = nn.Conv3d(hidden, num_outputs, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict boundary probabilities.

        Returns
        -------
        torch.Tensor
            Boundary predictions.
        """
        out1 = self.conv1(features)
        out2 = self.conv2(out1)  # Sequential: conv2 takes output of conv1

        combined = torch.cat([out1, out2], dim=1)
        combined = self.combine(combined)

        boundaries = torch.sigmoid(self.output(combined))
        return boundaries


class JointLSDFFN(nn.Module):
    """
    Joint LSD-FFN model with shared encoder.

    Architecture:
    - Shared U-Net encoder for feature extraction
    - LSD head: affinities + shape descriptors
    - FFN head: boundary predictions

    Example
    -------
    >>> config = JointModelConfig(base_features=32, depth=4)
    >>> model = JointLSDFFN(config)
    >>> x = torch.randn(1, 1, 64, 256, 256)
    >>> output = model(x)
    >>> print(output.lsd_affinities.shape)  # (1, 12, 64, 256, 256)
    >>> print(output.ffn_boundaries.shape)  # (1, 3, 64, 256, 256)
    """

    def __init__(self, config: Optional[JointModelConfig] = None):
        """
        Initialize joint model.

        Parameters
        ----------
        config : JointModelConfig, optional
            Model configuration.
        """
        super().__init__()

        if config is None:
            config = JointModelConfig()
        self.config = config

        # Shared encoder (U-Net backbone)
        self.encoder = UNet3D(
            in_channels=config.in_channels,
            base_features=config.base_features,
            depth=config.depth,
            out_channels=None,  # Return features
            use_residual=config.use_residual,
        )

        # Task-specific heads
        encoder_features = self.encoder.final_features

        self.lsd_head = LSDHead(
            encoder_features,
            num_affinities=config.num_lsd_affinities,
            num_descriptors=config.num_lsd_descriptors,
        )

        self.ffn_head = FFNHead(
            encoder_features,
            num_outputs=config.num_ffn_outputs,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> JointModelOutput:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (N, C, D, H, W).
        return_features : bool
            Include shared features in output.

        Returns
        -------
        JointModelOutput
            Predictions from all heads.
        """
        # Shared encoding
        features = self.encoder(x)

        # LSD predictions
        lsd_affinities, lsds = self.lsd_head(features)

        # FFN predictions
        ffn_boundaries = self.ffn_head(features)

        return JointModelOutput(
            lsd_affinities=lsd_affinities,
            lsds=lsds,
            ffn_boundaries=ffn_boundaries,
            shared_features=features if return_features else None,
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'JointLSDFFN':
        """
        Load pretrained model.

        Parameters
        ----------
        path : str
            Checkpoint path.
        device : str
            Device to load on.

        Returns
        -------
        JointLSDFFN
            Loaded model.
        """
        checkpoint = torch.load(path, map_location=device)

        if 'config' in checkpoint:
            config = JointModelConfig(**checkpoint['config'])
        else:
            config = JointModelConfig()

        model = cls(config)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model.to(device)

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'in_channels': self.config.in_channels,
                'base_features': self.config.base_features,
                'depth': self.config.depth,
                'num_lsd_affinities': self.config.num_lsd_affinities,
                'num_lsd_descriptors': self.config.num_lsd_descriptors,
                'num_ffn_outputs': self.config.num_ffn_outputs,
                'use_residual': self.config.use_residual,
            },
        }
        torch.save(checkpoint, path)


class JointLoss(nn.Module):
    """
    Combined loss for joint model training.

    Weights:
    - LSD affinity loss (BCE)
    - LSD descriptor loss (MSE)
    - FFN boundary loss (BCE)
    """

    def __init__(
        self,
        lsd_affinity_weight: float = 1.0,
        lsd_descriptor_weight: float = 0.5,
        ffn_weight: float = 1.0,
    ):
        super().__init__()
        self.lsd_affinity_weight = lsd_affinity_weight
        self.lsd_descriptor_weight = lsd_descriptor_weight
        self.ffn_weight = ffn_weight

    def forward(
        self,
        output: JointModelOutput,
        target_affinities: torch.Tensor,
        target_lsds: torch.Tensor,
        target_boundaries: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Returns
        -------
        dict
            Loss components and total.
        """
        # LSD affinity loss
        if mask is not None:
            aff_loss = F.binary_cross_entropy(
                output.lsd_affinities * mask,
                target_affinities * mask,
            )
        else:
            aff_loss = F.binary_cross_entropy(
                output.lsd_affinities,
                target_affinities,
            )

        # LSD descriptor loss
        if mask is not None:
            lsd_loss = F.mse_loss(
                output.lsds * mask,
                target_lsds * mask,
            )
        else:
            lsd_loss = F.mse_loss(output.lsds, target_lsds)

        # FFN boundary loss
        if mask is not None:
            ffn_loss = F.binary_cross_entropy(
                output.ffn_boundaries * mask,
                target_boundaries * mask,
            )
        else:
            ffn_loss = F.binary_cross_entropy(
                output.ffn_boundaries,
                target_boundaries,
            )

        # Total
        total = (
            self.lsd_affinity_weight * aff_loss +
            self.lsd_descriptor_weight * lsd_loss +
            self.ffn_weight * ffn_loss
        )

        return {
            'total': total,
            'lsd_affinity': aff_loss,
            'lsd_descriptor': lsd_loss,
            'ffn_boundary': ffn_loss,
        }


__all__ = [
    'JointModelConfig',
    'JointModelOutput',
    'JointLSDFFN',
    'JointLoss',
    'LSDHead',
    'FFNHead',
]
