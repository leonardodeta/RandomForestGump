"""
facenet_encoder.py
------------------
Encoder implementation based on FaceRetrievalModel.

If a checkpoint is provided, the checkpoint metadata is used to restore both the
architecture and the input image size. This avoids silently resizing inference
images differently from the training run that produced the checkpoint.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image

from encoder import Encoder
from face_retrieval_model import FaceRetrievalModel, get_face_transform, load_checkpoint


class FaceNetEncoder(Encoder):
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

        checkpoint_image_size = None
        if checkpoint_path is not None:
            # Read lightweight metadata before constructing the model. We still
            # call load_checkpoint for the actual model restoration.
            metadata = torch.load(checkpoint_path, map_location="cpu")
            config = metadata.get("config", None)
            checkpoint_image_size = metadata.get("image_size", None) or getattr(config, "image_size", None)

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

        if resolved_arch not in self._DEFAULT_IMAGE_SIZE:
            raise ValueError(f"Unsupported architecture: {resolved_arch}")

        if image_size is not None:
            resolved_image_size = int(image_size)
        elif checkpoint_image_size is not None:
            resolved_image_size = int(checkpoint_image_size)
        else:
            resolved_image_size = self._DEFAULT_IMAGE_SIZE[resolved_arch]

        if resolved_image_size <= 0:
            raise ValueError("image_size must be positive")

        self.arch = resolved_arch
        self._emb_dim = model.embedding_dim
        self.image_size = resolved_image_size
        self.transform = get_face_transform(self.image_size)
        self.model = model.eval()

        print(f"[FaceNetEncoder] image_size = {self.image_size}")

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode(tensors, normalize=False)

        return features.cpu()
