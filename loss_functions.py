# loss_functions.py

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class LossConfig:
    num_classes: int
    embedding_dim: int = 512

    # Recommended main loss
    loss_name: str = "arcface"  # "arcface", "cosface", "normalized_softmax", "supcon"

    # ArcFace / CosFace parameters
    scale: float = 30.0
    margin: float = 0.35

    # Cross entropy regularization
    label_smoothing: float = 0.05

    # Supervised contrastive loss parameter
    temperature: float = 0.07


# ============================================================
# ArcFace loss
# ============================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace-style angular-margin classification loss.

    Input:
        embeddings: normalized or unnormalized face embeddings [B, D]
        labels: identity labels [B]

    Output:
        dictionary containing loss and logits

    Why useful here:
        - The final retrieval uses cosine similarity.
        - ArcFace directly improves angular separation between identities.
        - Same-identity images should become closer in embedding space.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.35,
        label_smoothing: float = 0.05,
        easy_margin: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        return {
            "loss": loss,
            "logits": logits,
            "cosine_logits": cosine,
        }


# ============================================================
# CosFace loss
# ============================================================

class CosFaceLoss(nn.Module):
    """
    CosFace is similar to ArcFace, but subtracts a cosine margin directly.

    Usually:
        ArcFace is the first one I would try.
        CosFace is a good backup if ArcFace is unstable.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.35,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = cosine - one_hot * self.margin
        logits = logits * self.scale

        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        return {
            "loss": loss,
            "logits": logits,
            "cosine_logits": cosine,
        }


# ============================================================
# Normalized softmax baseline
# ============================================================

class NormalizedSoftmaxLoss(nn.Module):
    """
    Simpler classification loss.

    This is basically cross entropy over cosine-normalized class weights.
    It is weaker than ArcFace but more stable.

    Use this if ArcFace overfits or training becomes unstable.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 30.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.label_smoothing = label_smoothing

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        logits = F.linear(embeddings, weights)
        logits = logits * self.scale

        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        return {
            "loss": loss,
            "logits": logits,
            "cosine_logits": logits / self.scale,
        }


# ============================================================
# Optional supervised contrastive loss
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Optional retrieval-style loss.

    This pulls together all images with the same identity inside a batch
    and pushes apart images from different identities.

    For this to work well, the batch should contain multiple images
    per identity, ideally both natural and synthetic examples.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        device = embeddings.device

        embeddings = F.normalize(embeddings, p=2, dim=1)
        labels = labels.view(-1, 1)

        batch_size = embeddings.size(0)

        similarity = torch.matmul(embeddings, embeddings.T)
        similarity = similarity / self.temperature

        logits_mask = torch.ones_like(similarity, device=device)
        logits_mask.fill_diagonal_(0)

        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask * logits_mask

        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - logits_max.detach()

        exp_logits = torch.exp(similarity) * logits_mask

        log_prob = similarity - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-12
        )

        positives_per_sample = positive_mask.sum(dim=1)

        valid_samples = positives_per_sample > 0

        mean_log_prob_pos = (
            positive_mask * log_prob
        ).sum(dim=1) / torch.clamp(positives_per_sample, min=1.0)

        loss = -mean_log_prob_pos[valid_samples].mean()

        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            "loss": loss,
            "logits": None,
            "cosine_logits": similarity,
        }


# ============================================================
# Loss factory
# ============================================================

def build_loss_function(config: LossConfig) -> nn.Module:
    loss_name = config.loss_name.lower()

    if loss_name == "arcface":
        return ArcFaceLoss(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            scale=config.scale,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )

    if loss_name == "cosface":
        return CosFaceLoss(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            scale=config.scale,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )

    if loss_name == "normalized_softmax":
        return NormalizedSoftmaxLoss(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            scale=config.scale,
            label_smoothing=config.label_smoothing,
        )

    if loss_name == "supcon":
        return SupervisedContrastiveLoss(
            temperature=config.temperature,
        )

    raise ValueError(f"Unknown loss name: {config.loss_name}")


# ============================================================
# Optimizer helper
# ============================================================

def build_optimizer_with_loss(
    model: nn.Module,
    loss_function: nn.Module,
    backbone_lr: float = 1e-5,
    loss_head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Important:
        ArcFace/CosFace/NormalizedSoftmax have their own trainable class weights.

    Therefore the optimizer must receive:
        - trainable model parameters
        - trainable loss-function parameters
    """

    model_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    loss_params = [
        p for p in loss_function.parameters()
        if p.requires_grad
    ]

    param_groups = []

    if model_params:
        param_groups.append({
            "params": model_params,
            "lr": backbone_lr,
        })

    if loss_params:
        param_groups.append({
            "params": loss_params,
            "lr": loss_head_lr,
        })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )


# ============================================================
# Batch utility
# ============================================================

def unpack_training_batch(batch, device: torch.device):
    """
    Supports:
        images, labels
        images, labels, filenames
        {"image": images, "label": labels}
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
            raise ValueError("Batch dictionary must contain images and labels.")

    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple must contain at least images and labels.")

        images = batch[0]
        labels = batch[1]

    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    images = images.to(device, non_blocking=True)
    labels = torch.as_tensor(labels).long().to(device, non_blocking=True)

    return images, labels


# ============================================================
# Training epoch using the chosen loss
# ============================================================

def train_one_epoch_with_identity_loss(
    model: nn.Module,
    loss_function: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 25,
) -> Dict[str, float]:

    model.train()
    loss_function.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    has_logits = False

    for step, batch in enumerate(loader, start=1):
        images, labels = unpack_training_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        if "embeddings" not in outputs:
            raise ValueError(
                "Model must return an 'embeddings' tensor. "
                "The FaceRetrievalModel from block 1 already does this."
            )

        embeddings = outputs["embeddings"]

        loss_output = loss_function(
            embeddings=embeddings,
            labels=labels,
        )

        loss = loss_output["loss"]
        logits = loss_output.get("logits", None)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if logits is not None:
            has_logits = True
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

        if log_every > 0 and step % log_every == 0:
            avg_loss = total_loss / max(total_examples, 1)

            if has_logits:
                avg_acc = total_correct / max(total_examples, 1)
                print(
                    f"step {step:04d} | "
                    f"loss {avg_loss:.4f} | "
                    f"identity acc {avg_acc:.4f}"
                )
            else:
                print(
                    f"step {step:04d} | "
                    f"loss {avg_loss:.4f}"
                )

    result = {
        "loss": total_loss / max(total_examples, 1),
    }

    if has_logits:
        result["identity_accuracy"] = total_correct / max(total_examples, 1)

    return result


# ============================================================
# Validation using the chosen loss
# ============================================================


@torch.no_grad()
def evaluate_identity_loss(
    model: nn.Module,
    loss_function: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()
    loss_function.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    has_logits = False

    for batch in loader:
        images, labels = unpack_training_batch(batch, device)

        outputs = model(images)
        embeddings = outputs["embeddings"]

        loss_output = loss_function(
            embeddings=embeddings,
            labels=labels,
        )

        loss = loss_output["loss"]
        logits = loss_output.get("logits", None)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if logits is not None:
            has_logits = True
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

    result = {
        "loss": total_loss / max(total_examples, 1),
    }

    if has_logits:
        result["identity_accuracy"] = total_correct / max(total_examples, 1)

    return result


# ============================================================
# NT-Xent loss (SimCLR) — non richiede label di identita'
# ============================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss per SimCLR.

    Non richiede label: impara embedding continui da coppie di viste
    augmentate della stessa immagine.

    Per un batch di N immagini si hanno 2N viste (view_a, view_b).
    Per ogni vista v_i, la positiva e' la gemella v_j (stessa immagine),
    le negative sono tutte le altre 2(N-1) viste nel batch.

    Batch piu' grandi = piu' negativi = training piu' efficace.
    Minimo consigliato: 128 immagini per batch.

    Args:
        temperature: tau. Tipico 0.07 (stringente) o 0.1-0.2 (con batch piccoli).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,  # (N, D) proiezioni vista A
        z_b: torch.Tensor,  # (N, D) proiezioni vista B
    ) -> Dict[str, torch.Tensor]:

        N = z_a.size(0)
        device = z_a.device

        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)

        # Concatena: [z_a_0 ... z_a_{N-1}, z_b_0 ... z_b_{N-1}]
        z = torch.cat([z_a, z_b], dim=0)           # (2N, D)
        sim = (z @ z.T) / self.temperature           # (2N, 2N)

        # Escludi la diagonale (un campione con se stesso)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, float("-inf"))

        # La positiva di i in [0,N) e' i+N, e viceversa
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        loss = F.cross_entropy(sim, labels)
        return {"loss": loss, "logits": None}


# ============================================================
# Triplet Loss con hard negative mining online
# ============================================================

class TripletLoss(nn.Module):
    """
    Triplet Loss con hard negative mining online.

    Per ogni immagine nel batch:
        - Anchor:   embedding vista A
        - Positivo: embedding vista B (stessa immagine, augmentazione diversa)
        - Negativo: l'embedding PIU' SIMILE all'anchor tra tutti gli altri
                    del batch (hard negative)

    Loss = mean( max(0, d(anchor, pos) - d(anchor, neg) + margin) )

    Usa distanza coseno (1 - cosine_similarity) coerentemente
    con il retrieval in search_system.py.

    Args:
        margin: separazione minima tra positivo e negativo (tipico 0.2-0.5)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z_a: torch.Tensor,  # (N, D) embedding vista A
        z_b: torch.Tensor,  # (N, D) embedding vista B
    ) -> Dict[str, torch.Tensor]:

        N = z_a.size(0)
        device = z_a.device

        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)

        # Distanza coseno: 1 - similarita'
        pos_dist = 1.0 - (z_a * z_b).sum(dim=1)  # (N,)

        # Similarita' tra ogni anchor (z_a) e tutti i z_b nel batch
        sim_matrix = z_a @ z_b.T  # (N, N)

        # Escludi il positivo (diagonale)
        mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Hard negative: z_b piu' simile all'anchor (escluso il suo positivo)
        hardest_neg_sim, _ = sim_matrix.max(dim=1)   # (N,)
        neg_dist = 1.0 - hardest_neg_sim              # (N,)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return {"loss": loss, "logits": None}
