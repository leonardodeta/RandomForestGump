"""
facenet_encoder.py
------------------
Encoder implementation based on FaceRetrievalModel.

Usage modes:

1. Strong pretrained baseline:
       encoder = FaceNetEncoder(device)

2. Fine-tuned checkpoint:
       encoder = FaceNetEncoder(device, checkpoint_path="selfsup_checkpoint.pt")

When a checkpoint is provided, the architecture stored inside the checkpoint is
used automatically. This prevents loading an Inception-ResNet-V2 checkpoint into
an Inception-ResNet-V1 model by mistake.
"""

from typing import List, Optional

import torch
from PIL import Image

from encoder import Encoder
from face_retrieval_model import FaceRetrievalModel, load_checkpoint, get_face_transform


class FaceNetEncoder(Encoder):
    """
    Feature extractor based on FaceRetrievalModel.

    arch='inception_resnet_v1': FaceNet InceptionResNetV1 pretrained on VGGFace2,
                                512-d embeddings, 160px input.
    arch='inception_resnet_v2': timm InceptionResNetV2 pretrained on ImageNet,
                                1536-d embeddings, 299px input.

    Args:
        device:          torch.device used for inference.
        checkpoint_path: optional checkpoint saved by the training scripts.
        image_size:      input side length. If None, uses the architecture default.
        arch:            architecture for pretrained mode. If checkpoint_path is
                         provided, the checkpoint metadata takes precedence.
    """

    _DEFAULT_IMAGE_SIZE = {
        "inception_resnet_v1": 160,
        "inception_resnet_v2": 299,
    }

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        image_size: Optional[int] = None,
        arch: str = "inception_resnet_v1",
    ):
        self.device = device

        if checkpoint_path is not None:
            model, _, _ = load_checkpoint(checkpoint_path, device)
            resolved_arch = model.arch
            if arch != resolved_arch:
                print(
                    f"[FaceNetEncoder] Checkpoint architecture is {resolved_arch}; "
                    f"ignoring requested arch={arch}."
                )
            print(f"[FaceNetEncoder] Loaded checkpoint from: {checkpoint_path}")
        else:
            resolved_arch = arch
            model = FaceRetrievalModel(
                num_classes=None,
                pretrained="vggface2" if resolved_arch == "inception_resnet_v1" else "imagenet",
                arch=resolved_arch,
            ).to(device)
            print(f"[FaceNetEncoder] Using pretrained weights ({resolved_arch}).")

        self.arch = resolved_arch
        self._emb_dim = model.embedding_dim
        self.image_size = image_size or self._DEFAULT_IMAGE_SIZE[resolved_arch]
        self.transform = get_face_transform(self.image_size)
        self.model = model.eval()

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Receives a list of PIL images and returns a CPU tensor of shape
        (N, embedding_dim). The RetrievalSystem performs the final normalization.
        """
        if not images:
            return torch.empty((0, self.embedding_dim))

        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode(tensors, normalize=False)

        return features.cpu()
