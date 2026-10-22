"""Experimental Realism v4 Simulator & Interactive Studio for Minirhizotron Root Segmentation.

Designed for environment-specific synthetic dataset generation:
researchers select root-free soil images from their target site/camera, tune root
architecture, epidermal texture, and soil-root physical interactions until the
synthetic roots match their real roots, and generate a tailored training set for U-Net.
"""

from .realism_params import RealismConfig, PRESET_ENVIRONMENTS
from .engine import RealismSimulator

__all__ = ["RealismConfig", "PRESET_ENVIRONMENTS", "RealismSimulator"]
