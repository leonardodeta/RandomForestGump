# face_retrieval_model.py

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as exc:
    raise ImportError(
        "Missing dependency: facenet-pytorch. Install it with:\n"
        "pip install facenet-pytorch"
    ) from exc


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainingConfig:
    num_classes: int
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    log_every: int = 25
    use_flip_tta: bool = True
    top_k: int = 10


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Transform expected by FaceNet-style model
# ============================================================

def get_face_transform(image_size: int = 160):
    """
    Use this in your Dataset if your data pipeline does not already transform images.

    The model expects:
        - RGB image
        - size 160x160
        - tensor values normalized roughly to [-1, 1]
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


# ============================================================
# Batch handling
# ============================================================

def unpack_batch(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Supports common DataLoader outputs:

    1. Tuple:
        images, labels
        images, labels, filenames

    2. Dict:
        {
            "image" or "images" or "x": image_tensor,
            "label" or "labels" or "y" or "identity": label_tensor,
            "filename" or "filenames" or "path" or "paths": filenames
        }

    Returns:
        images: Tensor [B, 3, 160, 160]
        labels: LongTensor [B] or None
        filenames: list[str] or None
    """
    images = None
    labels = None
    filenames = None

    if isinstance(batch, dict):
        for key in ["image", "images", "x"]:
            if key in batch:
                images = batch[key]
                break

        for key in ["label", "labels", "y", "identity", "identities"]:
            if key in batch:
                labels = batch[key]
                break

        for key in ["filename", "filenames", "path", "paths", "image_id", "image_ids"]:
            if key in batch:
                filenames = batch[key]
                break

    elif isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            images, labels, filenames = batch
        else:
            raise ValueError(
                "Unsupported batch tuple/list format. Expected length 2 or 3."
            )
    else:
        raise ValueError("Unsupported batch format.")

    if images is None:
        raise ValueError("Could not find images in batch.")

    images = images.to(device, non_blocking=True)

    if labels is not None:
        labels = torch.as_tensor(labels).long().to(device, non_blocking=True)

    if filenames is not None:
        if isinstance(filenames, torch.Tensor):
            filenames = [str(x.item()) for x in filenames]
        else:
            filenames = [str(x) for x in filenames]

    return images, labels, filenames


# ============================================================
# Model: pretrained face-recognition backbone + classifier head
# ============================================================

# ============================================================
# Projection head (usato solo durante SimCLR training)
# ============================================================

class ProjectionHead(nn.Module):
    """
    MLP 512 -> 512 -> 128 per SimCLR.
    Viene usato SOLO durante il training, non a inference time.
    A inference si usano gli embedding a 512-d del backbone direttamente.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceRetrievalModel(nn.Module):
    """
    Pretrained face-recognition feature extractor.

    Modalita' SimCLR (num_classes=None, default):
        image -> backbone -> normalized embedding
        Il ProjectionHead viene creato esternamente in train_finetune.py.

    Modalita' classificazione (num_classes=N, opzionale):
        image -> backbone -> embedding -> classifier -> identity logits
        Utile solo se si hanno label di identita'.

    The embedding (512-d) is what we use for cosine similarity retrieval.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pretrained: str = "vggface2",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
        )

        self.embedding_dim = 512

        # Il classifier e' opzionale: serve solo se si hanno label di identita'.
        if num_classes is not None:
            self.classifier: Optional[nn.Module] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.embedding_dim, num_classes),
            )
        else:
            self.classifier = None

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_embeddings = self.backbone(images)
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        result: Dict[str, torch.Tensor] = {
            "embeddings": normalized_embeddings,
            "raw_embeddings": raw_embeddings,
        }

        if self.classifier is not None:
            result["logits"] = self.classifier(raw_embeddings)

        return result

    def encode(
        self,
        images: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        embeddings = self.backbone(images)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# ============================================================
# Freezing / unfreezing strategy
# ============================================================

def freeze_backbone(model: FaceRetrievalModel) -> None:
    """
    Stage 1:
    Freeze the pretrained face model and train only the classifier head.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_backbone_layers(model: FaceRetrievalModel) -> None:
    """
    Stage 2:
    Fine-tune only the last part of the backbone.

    This is safer than full fine-tuning when the dataset is small.
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


def unfreeze_full_backbone(model: FaceRetrievalModel) -> None:
    """
    More aggressive option.
    Use only if validation improves and overfitting is controlled.
    """
    for param in model.parameters():
        param.requires_grad = True


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(
    model: FaceRetrievalModel,
    head_lr: float,
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
            "lr": head_lr,
        })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )


# ============================================================
# Training and classification validation
# ============================================================

def train_one_epoch(
    model: FaceRetrievalModel,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
    log_every: int = 25,
) -> Dict[str, float]:
    model.train()

    backbone_has_trainable_params = any(
        p.requires_grad for p in model.backbone.parameters()
    )

    if not backbone_has_trainable_params:
        model.backbone.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, batch in enumerate(train_loader, start=1):
        images, labels, _ = unpack_batch(batch, device)

        if labels is None:
            raise ValueError("Training batches must contain labels.")

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
            current_loss = total_loss / max(total_examples, 1)
            current_acc = total_correct / max(total_examples, 1)
            print(
                f"step {step:04d} | "
                f"loss {current_loss:.4f} | "
                f"classification acc {current_acc:.4f}"
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "classification_accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate_classifier(
    model: FaceRetrievalModel,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images, labels, _ = unpack_batch(batch, device)

        if labels is None:
            raise ValueError("Validation batches must contain labels.")

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
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model: FaceRetrievalModel,
    loader: Iterable,
    device: torch.device,
    use_flip_tta: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Extract normalized embeddings.

    If use_flip_tta=True:
        embedding = average(model(image), model(horizontal_flip(image)))
        then normalize again.

    Returns:
        embeddings: Tensor [N, 512]
        labels: Tensor [N] or None
        filenames: list[str] or None
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    all_filenames = []

    has_labels = False
    has_filenames = False

    for batch in loader:
        images, labels, filenames = unpack_batch(batch, device)

        emb = model.encode(images, normalize=True)

        if use_flip_tta:
            flipped_images = torch.flip(images, dims=[3])
            flipped_emb = model.encode(flipped_images, normalize=True)
            emb = F.normalize((emb + flipped_emb) / 2.0, p=2, dim=1)

        all_embeddings.append(emb.cpu())

        if labels is not None:
            has_labels = True
            all_labels.append(labels.cpu())

        if filenames is not None:
            has_filenames = True
            all_filenames.extend(filenames)

    embeddings = torch.cat(all_embeddings, dim=0)

    labels_out = torch.cat(all_labels, dim=0) if has_labels else None
    filenames_out = all_filenames if has_filenames else None

    return embeddings, labels_out, filenames_out


# ============================================================
# Similarity search / retrieval
# ============================================================

def rank_gallery_for_queries(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int = 10,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Computes cosine similarity through matrix multiplication.

    Assumes both query_embeddings and gallery_embeddings are already L2-normalized.

    Returns:
        topk_indices: LongTensor [num_queries, top_k]
    """
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    all_topk = []

    for start in range(0, query_embeddings.size(0), chunk_size):
        end = start + chunk_size

        query_chunk = query_embeddings[start:end]
        similarity = query_chunk @ gallery_embeddings.T

        _, topk_indices = torch.topk(
            similarity,
            k=top_k,
            dim=1,
            largest=True,
            sorted=True,
        )

        all_topk.append(topk_indices.cpu())

    return torch.cat(all_topk, dim=0)


def build_retrieval_dictionary(
    query_filenames: Sequence[str],
    gallery_filenames: Sequence[str],
    topk_indices: torch.Tensor,
) -> Dict[str, List[str]]:
    """
    Builds the required submission dictionary:

        {
            "query_1.jpg": ["gallery_7.jpg", ..., "gallery_10.jpg"],
            ...
        }
    """
    if len(query_filenames) != topk_indices.size(0):
        raise ValueError("Number of query filenames does not match top-k matrix.")

    results = {}

    for query_idx, query_name in enumerate(query_filenames):
        ranked_gallery_names = [
            gallery_filenames[gallery_idx]
            for gallery_idx in topk_indices[query_idx].tolist()
        ]

        results[query_name] = ranked_gallery_names

    return results


# ============================================================
# Retrieval validation
# ============================================================

def retrieval_topk_accuracy(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Computes Top-1, Top-5, Top-10 retrieval accuracy.

    A query is correct at Top-k if at least one of the first k gallery
    images has the same identity label.
    """
    max_k = max(ks)

    topk_indices = rank_gallery_for_queries(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        top_k=max_k,
    )

    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()

    metrics = {}

    for k in ks:
        retrieved_labels = gallery_labels[topk_indices[:, :k]]
        correct = (retrieved_labels == query_labels.unsqueeze(1)).any(dim=1)
        metrics[f"top{k}"] = correct.float().mean().item()

    return metrics


def challenge_score_from_metrics(metrics: Dict[str, float]) -> float:
    """
    Challenge scoring:
        Top-1: 600 points
        Top-5: 300 points
        Top-10: 100 points
    """
    return (
        600.0 * metrics.get("top1", 0.0)
        + 300.0 * metrics.get("top5", 0.0)
        + 100.0 * metrics.get("top10", 0.0)
    )


@torch.no_grad()
def evaluate_retrieval(
    model: FaceRetrievalModel,
    query_loader: Iterable,
    gallery_loader: Iterable,
    device: torch.device,
    use_flip_tta: bool = True,
) -> Dict[str, float]:
    """
    Validation should imitate the real test setup:

        query = natural images
        gallery = synthetic images
    """
    query_embeddings, query_labels, _ = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    gallery_embeddings, gallery_labels, _ = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    if query_labels is None or gallery_labels is None:
        raise ValueError(
            "Retrieval validation requires labels for both query and gallery."
        )

    metrics = retrieval_topk_accuracy(
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        ks=(1, 5, 10),
    )

    metrics["challenge_score"] = challenge_score_from_metrics(metrics)

    return metrics


# ============================================================
# Full two-stage training
# ============================================================

def fit_two_stage_model(
    model: FaceRetrievalModel,
    train_loader: Iterable,
    config: TrainingConfig,
    device: torch.device,
    val_query_loader: Optional[Iterable] = None,
    val_gallery_loader: Optional[Iterable] = None,
) -> FaceRetrievalModel:
    """
    Stage 1:
        Freeze pretrained backbone.
        Train classifier head.

    Stage 2:
        Unfreeze last backbone layers.
        Fine-tune with smaller backbone learning rate.

    If validation query/gallery loaders are provided, the best model is selected
    using retrieval challenge score.
    """
    model = model.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")

    def maybe_validate(epoch_name: str):
        nonlocal best_state, best_score

        if val_query_loader is None or val_gallery_loader is None:
            return

        metrics = evaluate_retrieval(
            model=model,
            query_loader=val_query_loader,
            gallery_loader=val_gallery_loader,
            device=device,
            use_flip_tta=config.use_flip_tta,
        )

        score = metrics["challenge_score"]

        print(
            f"{epoch_name} retrieval | "
            f"top1 {metrics['top1']:.4f} | "
            f"top5 {metrics['top5']:.4f} | "
            f"top10 {metrics['top10']:.4f} | "
            f"score {score:.2f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            print(f"New best model. Score: {best_score:.2f}")

    # -------------------------
    # Stage 1
    # -------------------------
    if config.stage1_epochs > 0:
        print("\nStage 1: training classifier head with frozen backbone")
        freeze_backbone(model)

        optimizer = create_optimizer(
            model=model,
            head_lr=config.head_lr,
            backbone_lr=config.backbone_lr,
            weight_decay=config.weight_decay,
        )

        for epoch in range(1, config.stage1_epochs + 1):
            stats = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                label_smoothing=config.label_smoothing,
                log_every=config.log_every,
            )

            print(
                f"stage 1 epoch {epoch}/{config.stage1_epochs} | "
                f"loss {stats['loss']:.4f} | "
                f"classification acc {stats['classification_accuracy']:.4f}"
            )

            maybe_validate(f"stage 1 epoch {epoch}")

    # -------------------------
    # Stage 2
    # -------------------------
    if config.stage2_epochs > 0:
        print("\nStage 2: fine-tuning last backbone layers")
        unfreeze_last_backbone_layers(model)

        optimizer = create_optimizer(
            model=model,
            head_lr=config.head_lr * 0.2,
            backbone_lr=config.backbone_lr,
            weight_decay=config.weight_decay,
        )

        for epoch in range(1, config.stage2_epochs + 1):
            stats = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                label_smoothing=config.label_smoothing,
                log_every=config.log_every,
            )

            print(
                f"stage 2 epoch {epoch}/{config.stage2_epochs} | "
                f"loss {stats['loss']:.4f} | "
                f"classification acc {stats['classification_accuracy']:.4f}"
            )

            maybe_validate(f"stage 2 epoch {epoch}")

    if val_query_loader is not None and val_gallery_loader is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best validation model. Best score: {best_score:.2f}")

    return model


# ============================================================
# Final test retrieval
# ============================================================

@torch.no_grad()
def make_test_submission_dictionary(
    model: FaceRetrievalModel,
    query_loader: Iterable,
    gallery_loader: Iterable,
    device: torch.device,
    top_k: int = 10,
    use_flip_tta: bool = True,
) -> Dict[str, List[str]]:
    """
    Use this for the hidden test set.

    query_loader and gallery_loader must provide filenames.
    Labels are not required here.
    """
    query_embeddings, _, query_filenames = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    gallery_embeddings, _, gallery_filenames = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        use_flip_tta=use_flip_tta,
    )

    if query_filenames is None:
        raise ValueError("query_loader must provide query filenames.")

    if gallery_filenames is None:
        raise ValueError("gallery_loader must provide gallery filenames.")

    topk_indices = rank_gallery_for_queries(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        top_k=top_k,
    )

    results = build_retrieval_dictionary(
        query_filenames=query_filenames,
        gallery_filenames=gallery_filenames,
        topk_indices=topk_indices,
    )

    return results


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    path: str,
    model: FaceRetrievalModel,
    config: TrainingConfig,
    label_to_identity: Optional[Dict[int, str]] = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "label_to_identity": label_to_identity,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to: {path}")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> Tuple[FaceRetrievalModel, TrainingConfig, Optional[Dict[int, str]]]:
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", None)

    # Supporta checkpoint SimCLR (num_classes assente) e checkpoint supervisionati
    num_classes = getattr(config, "num_classes", None) if config is not None else None

    model = FaceRetrievalModel(
        num_classes=num_classes,
        pretrained=None,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    label_to_identity = checkpoint.get("label_to_identity", None)

    return model, config, label_to_identity


# ============================================================
# Example usage with your existing data pipeline
# ============================================================

def run_training_and_retrieval(
    train_loader: Iterable,
    test_query_loader: Iterable,
    test_gallery_loader: Iterable,
    num_classes: int,
    val_query_loader: Optional[Iterable] = None,
    val_gallery_loader: Optional[Iterable] = None,
) -> Dict[str, List[str]]:
    """
    This is the high-level function your group can call once the data pipeline exists.

    Required loaders:

        train_loader:
            returns images and identity labels

        test_query_loader:
            returns query images and filenames

        test_gallery_loader:
            returns gallery images and filenames

    Optional validation loaders:

        val_query_loader:
            natural validation images with labels

        val_gallery_loader:
            synthetic validation images with labels
    """
    device = get_device()
    print(f"Using device: {device}")

    config = TrainingConfig(
        num_classes=num_classes,
        stage1_epochs=5,
        stage2_epochs=5,
        head_lr=1e-3,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        label_smoothing=0.05,
        log_every=25,
        use_flip_tta=True,
        top_k=10,
    )

    model = FaceRetrievalModel(
        num_classes=config.num_classes,
        pretrained="vggface2",
        dropout=0.2,
    )

    model = fit_two_stage_model(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        val_query_loader=val_query_loader,
        val_gallery_loader=val_gallery_loader,
    )

    results = make_test_submission_dictionary(
        model=model,
        query_loader=test_query_loader,
        gallery_loader=test_gallery_loader,
        device=device,
        top_k=config.top_k,
        use_flip_tta=config.use_flip_tta,
    )

    return results
