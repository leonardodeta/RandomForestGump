"""
encoder.py
----------
Interfaccia astratta tra il MODELLO (che fa il compagno) e il SEARCH SYSTEM.

Il compagno deve fornire una classe che eredita da `Encoder` e implementa
`embed_batch`. Il search system non sa (e non deve sapere) cosa c'è dentro:
puo' essere ResNet50 ImageNet, ResNet50 fine-tunato, ensemble, etc.

Questo permette di sviluppare in parallelo senza pestarsi i piedi.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
from PIL import Image


class Encoder(ABC):
    """Interfaccia che ogni modello deve rispettare."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionalita' dell'embedding prodotto (es. 2048 per ResNet50)."""
        ...

    @abstractmethod
    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Estrae embedding da una lista di immagini PIL.

        Args:
            images: lista di N immagini PIL

        Returns:
            Tensor di shape (N, D) NON normalizzato.
            La normalizzazione la fa il search system, cosi' siamo sicuri
            che sia consistente in tutta la pipeline.
        """
        ...


# ============================================================================
# Implementazione di default: ResNet50 ImageNet (lo script di partenza).
# Serve come BASELINE e come stand-in finche' il compagno non ha pronto
# il modello fine-tunato.
# ============================================================================

class ResNet50ImageNetEncoder(Encoder):
    """ResNet50 pretrainata su ImageNet, fc rimossa -> output 2048-d."""

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


# ============================================================================
# Quando il compagno ha pronto il modello fine-tunato, creera' qualcosa tipo:
#
# class FinetunedResNet50Encoder(Encoder):
#     def __init__(self, checkpoint_path, device):
#         ...carica i pesi fine-tunati...
#
#     def embed_batch(self, images):
#         ...stesso input/output, modello diverso dentro...
#
# Il search system non cambia di una riga.
# ============================================================================
