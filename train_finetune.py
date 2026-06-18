"""
train_finetune.py
-----------------
Optional supervised fine-tuning utilities.

This file is not required for the main competition baseline. It is useful only
when identity labels are available. The final retrieval pipeline can still run
with the pretrained FaceNet encoder without using this module.
"""


import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_functions import ArcFaceLoss, CosFaceLoss, NormalizedSoftmaxLoss

# Configuration

@dataclass
class FineTuneConfig:
    num_classes: int

    # Training stages
    frozen_epochs: int = 5
    finetune_epochs: int = 8

    # Learning rates
    classifier_lr: float = 1e-3
    backbone_lr: float = 1e-5

    # Regularization
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05

    # Supervised objective. "cross_entropy" uses the model classifier head.
    # The angular-margin alternatives are better aligned with cosine retrieval,
    # but should still be validated before use.
    supervised_loss: str = "cross_entropy"
    margin: float = 0.35
    scale: float = 30.0

    # Reproducibility and checkpoint metadata
    seed: int = 42
    image_size: int = 160

    # Retrieval validation
    top_k: int = 10
    use_flip_tta: bool = True

    # Logging / early stopping
    log_every: int = 25
    patience: int = 4

    # Checkpoint
    checkpoint_path: str = "best_face_retrieval_model.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set common random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Batch utilities

def unpack_batch(batch, device: torch.device):
    """
    Supports both:
        batch = (images, labels)
        batch = {"image": images, "label": labels}
    """

    if isinstance(batch, dict):
        images = None
        labels = None

        for key in ["image", "images", "x"]:
            if key in batch:
                images = batch[key]
                break

        for key in ["label", "labels", "y", "identity", "identities"]:
            if key in batch:
                labels = batch[key]
                break

        if images is None or labels is None:
            raise ValueError(
                "Dictionary batch must contain image/images/x and label/labels/y."
            )

    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must contain at least images and labels.")

        images, labels = batch[0], batch[1]

    else:
        raise ValueError("Unsupported batch format.")

    images = images.to(device, non_blocking=True)
    labels = torch.as_tensor(labels).long().to(device, non_blocking=True)

    return images, labels


# Freezing strategy

def freeze_backbone(model: nn.Module) -> None:
    """
    Stage 1:
    Keep the pretrained face feature extractor fixed.
    Train only the classifier head.
    """

    for param in model.backbone.parameters():
        param.requires_grad = False

    if getattr(model, "classifier", None) is None:
        raise ValueError("Supervised fine-tuning requires model.classifier to be defined.")

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_layers(model: nn.Module) -> None:
    """
    Stage 2:
    Fine-tune only the last layers of the face backbone.

    This is safer than full fine-tuning, because the dataset is not huge.
    """

    for param in model.backbone.parameters():
        param.requires_grad = False

    if getattr(model, "arch", "inception_resnet_v1") == "inception_resnet_v1":
        trainable_keywords = [
            "repeat_3",
            "block8",
            "last_linear",
            "last_bn",
        ]
        for name, param in model.backbone.named_parameters():
            if any(keyword in name for keyword in trainable_keywords):
                param.requires_grad = True
    else:
        # timm Inception-ResNet-V2 names can vary by version; unfreeze the
        # final parameter tensors instead of relying on exact block names.
        named_params = list(model.backbone.named_parameters())
        for _, param in named_params[-30:]:
            param.requires_grad = True

    if getattr(model, "classifier", None) is not None:
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_full_backbone(model: nn.Module) -> None:
    """
    Optional aggressive fine-tuning.
    Use only if validation shows improvement.
    """

    for param in model.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Optimizer

def build_optimizer(
    model: nn.Module,
    classifier_lr: float,
    backbone_lr: float,
    weight_decay: float,
    extra_modules: Optional[Iterable[nn.Module]] = None,
) -> torch.optim.Optimizer:

    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    if getattr(model, "classifier", None) is None:
        classifier_params = []
    else:
        classifier_params = [
            p for p in model.classifier.parameters()
            if p.requires_grad
        ]

    param_groups = []

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
        })

    if classifier_params:
        param_groups.append({
            "params": classifier_params,
            "lr": classifier_lr,
        })

    if extra_modules is not None:
        extra_params = [
            p
            for module in extra_modules
            for p in module.parameters()
            if p.requires_grad
        ]
        if extra_params:
            param_groups.append({
                "params": extra_params,
                "lr": classifier_lr,
            })

    if not param_groups:
        raise ValueError(
            "No trainable parameters found. Check the freezing strategy and classifier head."
        )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

# Supervised objective

def build_supervised_criterion(config: FineTuneConfig, embedding_dim: int) -> Optional[nn.Module]:
    """Create an optional retrieval-oriented supervised loss.

    ``None`` means ordinary cross entropy over the model classifier head.
    The other options operate directly on embeddings and better match cosine
    retrieval, but they should still be chosen only after validation.
    """
    name = config.supervised_loss.lower().replace("-", "_")
    if name in {"cross_entropy", "ce"}:
        return None
    if name == "arcface":
        return ArcFaceLoss(
            num_classes=config.num_classes,
            embedding_dim=embedding_dim,
            scale=config.scale,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
    if name == "cosface":
        return CosFaceLoss(
            num_classes=config.num_classes,
            embedding_dim=embedding_dim,
            scale=config.scale,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
    if name in {"normalized_softmax", "norm_softmax"}:
        return NormalizedSoftmaxLoss(
            num_classes=config.num_classes,
            embedding_dim=embedding_dim,
            scale=config.scale,
            label_smoothing=config.label_smoothing,
        )
    raise ValueError(
        "Unsupported supervised_loss. Use cross_entropy, arcface, cosface, or normalized_softmax."
    )

# One training epoch

def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
    log_every: int = 25,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, batch in enumerate(loader, start=1):
        images, labels = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        if criterion is None:
            if "logits" not in outputs:
                raise ValueError("cross-entropy training requires a model with a classifier head.")
            logits = outputs["logits"]
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=label_smoothing,
            )
        else:
            raw_embeddings = outputs.get("raw_embeddings", outputs.get("embeddings"))
            if raw_embeddings is None:
                raise ValueError("metric-loss training requires model outputs to include embeddings")
            loss_output = criterion(raw_embeddings, labels)
            loss = loss_output["loss"]
            logits = loss_output.get("logits", loss_output.get("cosine_logits"))
            if logits is None:
                raise ValueError("criterion output must include logits or cosine_logits for accuracy logging")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

        if log_every > 0 and step % log_every == 0:
            avg_loss = total_loss / total_examples
            avg_acc = total_correct / total_examples

            print(
                f"step {step:04d} | "
                f"loss {avg_loss:.4f} | "
                f"classification acc {avg_acc:.4f}"
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "classification_accuracy": total_correct / max(total_examples, 1),
    }

# Classification validation

@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images, labels = unpack_batch(batch, device)

        outputs = model(images)
        if "logits" not in outputs:
            raise ValueError("evaluate_classifier requires a model with a classifier head.")
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "classification_accuracy": total_correct / max(total_examples, 1),
    }

# Embedding extraction for retrieval validation

@torch.no_grad()
def extract_embeddings_with_labels(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    use_flip_tta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()

    all_embeddings = []
    all_labels = []

    for batch in loader:
        images, labels = unpack_batch(batch, device)

        embeddings = model.encode(images, normalize=True)

        if use_flip_tta:
            flipped_images = torch.flip(images, dims=[3])
            flipped_embeddings = model.encode(flipped_images, normalize=True)

            embeddings = F.normalize(
                (embeddings + flipped_embeddings) / 2.0,
                p=2,
                dim=1,
            )

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels


# Retrieval validation

def compute_topk_retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    top_k: int = 10,
) -> Dict[str, float]:

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    similarity_matrix = query_embeddings @ gallery_embeddings.T

    if gallery_embeddings.size(0) == 0:
        raise ValueError("gallery_embeddings is empty")
    safe_top_k = min(int(top_k), gallery_embeddings.size(0))

    _, topk_indices = torch.topk(
        similarity_matrix,
        k=safe_top_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    retrieved_labels = gallery_labels[topk_indices]

    top1_correct = (
        retrieved_labels[:, :1] == query_labels.unsqueeze(1)
    ).any(dim=1)

    top5_correct = (
        retrieved_labels[:, :5] == query_labels.unsqueeze(1)
    ).any(dim=1)

    top10_correct = (
        retrieved_labels[:, :10] == query_labels.unsqueeze(1)
    ).any(dim=1)

    top1 = top1_correct.float().mean().item()
    top5 = top5_correct.float().mean().item()
    top10 = top10_correct.float().mean().item()

    challenge_score = 600 * top1 + 300 * top5 + 100 * top10

    return {
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "challenge_score": challenge_score,
    }


@torch.no_grad()
def evaluate_retrieval(
    model: nn.Module,
    val_query_loader: Iterable,
    val_gallery_loader: Iterable,
    device: torch.device,
    use_flip_tta: bool = True,
    top_k: int = 10,
) -> Dict[str, float]:

    query_embeddings, query_labels = extract_embeddings_with_labels(
        model=model,
        loader=val_query_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    gallery_embeddings, gallery_labels = extract_embeddings_with_labels(
        model=model,
        loader=val_gallery_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    return compute_topk_retrieval_metrics(
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        top_k=top_k,
    )

# Full fine-tuning procedure

def fine_tune_face_model(
    model: nn.Module,
    train_loader: Iterable,
    val_query_loader: Optional[Iterable],
    val_gallery_loader: Optional[Iterable],
    config: FineTuneConfig,
    device: Optional[torch.device] = None,
) -> nn.Module:

    if device is None:
        device = get_device()

    print(f"Using device: {device}")
    set_seed(config.seed)

    model = model.to(device)
    criterion = build_supervised_criterion(config, getattr(model, "embedding_dim"))
    if criterion is not None:
        criterion = criterion.to(device)
        print(f"Using supervised retrieval loss: {config.supervised_loss}")
    else:
        print("Using supervised loss: cross_entropy")

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    epochs_without_improvement = 0

    # Helper: validate using retrieval, not only classification

    def run_retrieval_validation(stage_name: str, epoch: int) -> float:
        nonlocal best_state
        nonlocal best_score
        nonlocal epochs_without_improvement

        if val_query_loader is None or val_gallery_loader is None:
            return best_score

        metrics = evaluate_retrieval(
            model=model,
            val_query_loader=val_query_loader,
            val_gallery_loader=val_gallery_loader,
            device=device,
            use_flip_tta=config.use_flip_tta,
            top_k=config.top_k,
        )

        score = metrics["challenge_score"]

        print(
            f"{stage_name} epoch {epoch} retrieval | "
            f"top1 {metrics['top1']:.4f} | "
            f"top5 {metrics['top5']:.4f} | "
            f"top10 {metrics['top10']:.4f} | "
            f"score {score:.2f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "checkpoint_version": 2,
                    "model_state_dict": best_state,
                    "config": config,
                    "arch": getattr(model, "arch", "inception_resnet_v1"),
                    "num_classes": config.num_classes,
                    "image_size": config.image_size,
                    "best_score": best_score,
                    "training_mode": "supervised_finetune",
                    "supervised_loss": config.supervised_loss,
                },
                config.checkpoint_path,
            )

            print(f"New best model saved to {config.checkpoint_path}")

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best score: {best_score:.2f}"
            )

        return score

    # Stage 1: frozen backbone

    print("\n==============================")
    print("Stage 1: frozen backbone")
    print("==============================")

    freeze_backbone(model)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = build_optimizer(
        model=model,
        classifier_lr=config.classifier_lr,
        backbone_lr=config.backbone_lr,
        weight_decay=config.weight_decay,
        extra_modules=[criterion] if criterion is not None else None,
    )

    for epoch in range(1, config.frozen_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_smoothing=config.label_smoothing,
            log_every=config.log_every,
            criterion=criterion,
        )

        print(
            f"Stage 1 epoch {epoch}/{config.frozen_epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"classification acc {train_stats['classification_accuracy']:.4f}"
        )

        run_retrieval_validation("Stage 1", epoch)

    # Stage 2: partial fine-tuning

    print("\n==============================")
    print("Stage 2: fine-tune last layers")
    print("==============================")

    unfreeze_last_layers(model)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = build_optimizer(
        model=model,
        classifier_lr=config.classifier_lr * 0.2,
        backbone_lr=config.backbone_lr,
        weight_decay=config.weight_decay,
        extra_modules=[criterion] if criterion is not None else None,
    )

    epochs_without_improvement = 0

    for epoch in range(1, config.finetune_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_smoothing=config.label_smoothing,
            log_every=config.log_every,
            criterion=criterion,
        )

        print(
            f"Stage 2 epoch {epoch}/{config.finetune_epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"classification acc {train_stats['classification_accuracy']:.4f}"
        )

        run_retrieval_validation("Stage 2", epoch)

        if epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    # Load best model

    if val_query_loader is not None and val_gallery_loader is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with retrieval score: {best_score:.2f}")

    return model


# SimCLR: dataset a coppie e training epoch

import os
from typing import Tuple
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def _simclr_augmentation(image_size: int = 160) -> transforms.Compose:
    """
    Augmentazioni aggressive per SimCLR su volti.
    Forzano il modello a imparare feature invarianti all'aspetto visivo
    mantenendo l'identita'.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.6, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(
            kernel_size=int(0.1 * image_size) | 1,
            sigma=(0.1, 2.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


class PairDataset(Dataset):
    """
    Loads images from a folder (flat or ImageFolder).
    Returns two different augmentations for each image: (view_a, view_b).
    Labels are not used.

    Compatible with CelebA (flat folder) and any ImageFolder structure.
    """

    VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, image_size: int = 160):
        self.augment = _simclr_augmentation(image_size)
        self.paths = self._collect_paths(root)
        if len(self.paths) == 0:
            raise ValueError(f"Nessuna immagine trovata in: {root}")
        print(f"[PairDataset] {len(self.paths)} immagini caricate da: {root}")

    def _collect_paths(self, root: str):
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(self.VALID_EXT):
                    paths.append(os.path.join(dirpath, fn))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.augment(img), self.augment(img)


def train_simclr_epoch(
    backbone: nn.Module,
    projection_head: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int = 50,
) -> Dict[str, float]:
    """
    One epoch of self-supervised two-view training.

    The criterion can be NTXentLoss or TripletLoss. When the backbone is fully
    frozen we keep it in eval mode, otherwise BatchNorm running statistics would
    still change even though all parameters have requires_grad=False.
    """
    backbone_has_trainable_params = any(
        p.requires_grad for p in backbone.parameters()
    )
    if backbone_has_trainable_params:
        backbone.train()
    else:
        backbone.eval()

    projection_head.train()

    total_loss = 0.0
    total_steps = 0

    for step, (view_a, view_b) in enumerate(loader, start=1):
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        h_a = backbone.encode(view_a, normalize=False)
        h_b = backbone.encode(view_b, normalize=False)

        z_a = projection_head(h_a)
        z_b = projection_head(h_b)

        loss_output = criterion(z_a, z_b)
        loss = loss_output["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        if log_every > 0 and step % log_every == 0:
            avg = total_loss / total_steps
            print(f"  step {step:04d} | self-supervised loss {avg:.4f}")

    return {"loss": total_loss / max(total_steps, 1)}
