"""
facenet_encoder.py
------------------
Implementazione di Encoder basata su FaceRetrievalModel (face_retrieval_model.py).

Due modalita' di utilizzo:

1. Pretrained puro (baseline forte, nessun training necessario):
       encoder = FaceNetEncoder(device)

2. Fine-tunato con SimCLR (dopo aver eseguito simclr_train.py):
       encoder = FaceNetEncoder(device, checkpoint_path="simclr_checkpoint.pt")

Il search system (search_system.py) non cambia in nessuno dei due casi.
"""

from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from face_retrieval_model import FaceRetrievalModel, load_checkpoint, get_face_transform


class FaceNetEncoder(Encoder):
    """
    Feature extractor basato su FaceRetrievalModel.

    arch='inception_resnet_v1' (default): InceptionResNetV1 pretrained VGGFace2, 512-d, image 160px.
    arch='inception_resnet_v2':           InceptionResNetV2 via timm, ImageNet, 1536-d, image 299px.

    Args:
        device:          torch.device su cui girare il modello.
        checkpoint_path: path al checkpoint salvato da simclr_train.py.
                         Se None, usa i pesi pretrained.
        image_size:      lato dell'immagine in input. Se None, usa il default dell'arch.
        arch:            architettura da usare ('inception_resnet_v1' o 'inception_resnet_v2').
    """

    _DEFAULT_IMAGE_SIZE = {
        "inception_resnet_v1": 160,
        "inception_resnet_v2": 299,
    }

    _EMBEDDING_DIM = {
        "inception_resnet_v1": 512,
        "inception_resnet_v2": 1536,
    }

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        image_size: Optional[int] = None,
        arch: str = "inception_resnet_v1",
    ):
        self.device = device
        self.arch = arch
        self._emb_dim = self._EMBEDDING_DIM[arch]

        img_size = image_size or self._DEFAULT_IMAGE_SIZE[arch]
        self.transform = get_face_transform(img_size)

        if checkpoint_path is not None:
            model, config, _ = load_checkpoint(checkpoint_path, device)
            print(f"[FaceNetEncoder] Caricato checkpoint da: {checkpoint_path}")
        else:
            model = FaceRetrievalModel(num_classes=None, pretrained="vggface2", arch=arch)
            model = model.to(device)
            print(f"[FaceNetEncoder] Usando pesi pretrained ({arch}).")

        self.model = model.eval()

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Riceve una lista di immagini PIL, restituisce Tensor (N, embedding_dim).
        NON normalizzato: ci pensa RetrievalSystem.
        """
        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode(tensors, normalize=False)

        return features.cpu()
