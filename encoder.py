"""
encoder.py
----------
Abstract interface between the MODEL — provided by the partner — and the SEARCH SYSTEM.

The partner must provide a class that inherits from Encoder and implements embed_batch. The search system does not know, and must not know, what is inside: it could be ImageNet ResNet50, fine-tuned ResNet50, an ensemble, etc.

This makes it possible to develop in parallel without getting in each other’s way.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
from PIL import Image


class Encoder(ABC):
    """Interface for every model."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the product embedding (eg. 2048 for ResNet50)."""
        ...

    @abstractmethod
    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extracts embeddings from a list of PIL images.

        Args:
        images: list of N PIL images

        Returns:
        Shape tensor (N, D) NOT normalized.
        The search system does the normalization, so we can be sure
        it is consistent throughout the pipeline.
        """
        ...


class ResNet50ImageNetEncoder(Encoder):
    """ResNet50 pretrained on Facenet, fc removed -> output 2048-d."""

    def __init__(self, device: torch.device):
        from torchvision.models import resnet50, ResNet50_Weights
        self.device = device
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.preprocess = weights.transforms()

    @property
    def embedding_dim(self) -> int:
        return 2048

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([
            self.preprocess(img.convert("RGB")) for img in images
        ]).to(self.device)
        with torch.no_grad():
            features = self.model(tensors)
        return features.cpu()

