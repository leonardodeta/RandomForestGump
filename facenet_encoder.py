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
    InceptionResnetV1 pretrained su VGGFace2 come feature extractor.

    Produce embedding a 512 dimensioni.
    La normalizzazione L2 viene fatta da RetrievalSystem, NON qui dentro
    (come richiesto dall'interfaccia Encoder).

    Args:
        device:          torch.device su cui girare il modello.
        checkpoint_path: path al checkpoint salvato da simclr_train.py.
                         Se None, usa i pesi pretrained VGGFace2.
        image_size:      lato dell'immagine in input (default 160).
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        image_size: int = 160,
    ):
        self.device = device
        self.transform = get_face_transform(image_size)

        if checkpoint_path is not None:
            model, config, _ = load_checkpoint(checkpoint_path, device)
            print(f"[FaceNetEncoder] Caricato checkpoint da: {checkpoint_path}")
        else:
            model = FaceRetrievalModel(num_classes=None, pretrained="vggface2")
            model = model.to(device)
            print("[FaceNetEncoder] Usando pesi pretrained VGGFace2.")

        self.model = model.eval()

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Riceve una lista di immagini PIL, restituisce Tensor (N, 512).
        NON normalizzato: ci pensa RetrievalSystem.
        """
        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode(tensors, normalize=False)  # (N, 512)

        return features.cpu()
