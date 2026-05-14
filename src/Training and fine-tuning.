# train_finetune.py

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Batch utilities
# ============================================================

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


# ============================================================
# Freezing strategy
# ============================================================

def freeze_backbone(model: nn.Module) -> None:
    """
    Stage 1:
    Keep the pretrained face feature extractor fixed.
    Train only the classifier head.
    """

    for param in model.backbone.parameters():
        param.requires_grad = False

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

    trainable_keywords = [
        "repeat_3",
        "block8",
        "last_linear",
        "last_bn",
    ]

    for name, param in model.backbone.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True

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


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model: nn.Module,
    classifier_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:

    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

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

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
    log_every: int = 25,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, batch in enumerate(loader, start=1):
        images, labels = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=label_smoothing,
        )

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


# ============================================================
# Classification validation
# ============================================================

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


# ============================================================
# Embedding extraction for retrieval validation
# ============================================================

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


# ============================================================
# Retrieval validation
# ============================================================

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

    _, topk_indices = torch.topk(
        similarity_matrix,
        k=top_k,
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


# ============================================================
# Full fine-tuning procedure
# ============================================================

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

    model = model.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Helper: validate using retrieval, not only classification
    # --------------------------------------------------------

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
                    "model_state_dict": best_state,
                    "config": config,
                    "best_score": best_score,
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

    # ========================================================
    # Stage 1: frozen backbone
    # ========================================================

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
    )

    for epoch in range(1, config.frozen_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_smoothing=config.label_smoothing,
            log_every=config.log_every,
        )

        print(
            f"Stage 1 epoch {epoch}/{config.frozen_epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"classification acc {train_stats['classification_accuracy']:.4f}"
        )

        run_retrieval_validation("Stage 1", epoch)

    # ========================================================
    # Stage 2: partial fine-tuning
    # ========================================================

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

    # ========================================================
    # Load best model
    # ========================================================

    if val_query_loader is not None and val_gallery_loader is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with retrieval score: {best_score:.2f}")

    return model
