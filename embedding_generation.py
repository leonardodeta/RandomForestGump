# embedding_generation.py

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class EmbeddingConfig:
    top_k: int = 10
    use_flip_tta: bool = True
    similarity_chunk_size: int = 512
    save_embeddings: bool = True
    query_embeddings_path: str = "query_embeddings.pt"
    gallery_embeddings_path: str = "gallery_embeddings.pt"
    results_path: str = "retrieval_results.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Batch unpacking
# ============================================================

def unpack_inference_batch(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[List[str]], Optional[torch.Tensor]]:
    """
    Supports common DataLoader outputs.

    Accepted tuple/list formats:
        images
        images, filenames
        images, labels
        images, labels, filenames

    Accepted dict formats:
        {
            "image" / "images" / "x": image tensor,
            "filename" / "filenames" / "path" / "paths": filenames,
            "label" / "labels" / "y" / "identity": labels
        }

    Returns:
        images: Tensor [B, 3, H, W]
        filenames: list[str] or None
        labels: Tensor [B] or None
    """

    images = None
    filenames = None
    labels = None

    if isinstance(batch, dict):
        for key in ["image", "images", "x"]:
            if key in batch:
                images = batch[key]
                break

        for key in ["filename", "filenames", "path", "paths", "image_id", "image_ids"]:
            if key in batch:
                filenames = batch[key]
                break

        for key in ["label", "labels", "y", "identity", "identities"]:
            if key in batch:
                labels = batch[key]
                break

    elif isinstance(batch, torch.Tensor):
        images = batch

    elif isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            images = batch[0]

        elif len(batch) == 2:
            images = batch[0]

            second = batch[1]

            if isinstance(second, torch.Tensor):
                labels = second
            else:
                filenames = second

        elif len(batch) >= 3:
            images = batch[0]
            labels = batch[1]
            filenames = batch[2]

        else:
            raise ValueError("Empty batch received.")

    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    if images is None:
        raise ValueError("Could not find image tensor in batch.")

    images = images.to(device, non_blocking=True)

    if labels is not None:
        labels = torch.as_tensor(labels).long()

    if filenames is not None:
        filenames = [str(name) for name in filenames]

    return images, filenames, labels


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def encode_batch(
    model: nn.Module,
    images: torch.Tensor,
    use_flip_tta: bool = True,
) -> torch.Tensor:
    """
    Produces normalized embeddings for one batch.

    Requires the model to have:

        model.encode(images, normalize=True)

    This matches the FaceRetrievalModel from block 1.
    """

    embeddings = model.encode(images, normalize=True)

    if use_flip_tta:
        flipped_images = torch.flip(images, dims=[3])
        flipped_embeddings = model.encode(flipped_images, normalize=True)

        embeddings = (embeddings + flipped_embeddings) / 2.0
        embeddings = F.normalize(embeddings, p=2, dim=1)

    else:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    use_flip_tta: bool = True,
) -> Tuple[torch.Tensor, Optional[List[str]], Optional[torch.Tensor]]:
    """
    Extracts embeddings for an entire loader.

    Returns:
        embeddings: Tensor [N, D]
        filenames: list[str] or None
        labels: Tensor [N] or None
    """

    model.eval()
    model.to(device)

    all_embeddings = []
    all_filenames = []
    all_labels = []

    has_filenames = False
    has_labels = False

    for batch in loader:
        images, filenames, labels = unpack_inference_batch(batch, device)

        embeddings = encode_batch(
            model=model,
            images=images,
            use_flip_tta=use_flip_tta,
        )

        all_embeddings.append(embeddings.cpu())

        if filenames is not None:
            has_filenames = True
            all_filenames.extend(filenames)

        if labels is not None:
            has_labels = True
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)

    filenames_out = all_filenames if has_filenames else None
    labels_out = torch.cat(all_labels, dim=0) if has_labels else None

    return embeddings, filenames_out, labels_out


# ============================================================
# Saving / loading embeddings
# ============================================================

def save_embedding_file(
    path: str,
    embeddings: torch.Tensor,
    filenames: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
) -> None:
    payload = {
        "embeddings": embeddings,
        "filenames": filenames,
        "labels": labels,
    }

    torch.save(payload, path)
    print(f"Saved embeddings to: {path}")


def load_embedding_file(
    path: str,
) -> Tuple[torch.Tensor, Optional[List[str]], Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")

    return (
        payload["embeddings"],
        payload.get("filenames", None),
        payload.get("labels", None),
    )


# ============================================================
# Similarity search
# ============================================================

def compute_topk_indices(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int = 10,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes top-k gallery indices for every query.

    Uses cosine similarity:

        similarity = normalized_query @ normalized_gallery.T

    Chunking avoids creating a huge matrix all at once.
    """

    if device is None:
        device = get_device()

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    gallery_embeddings = gallery_embeddings.to(device)

    all_topk_indices = []

    for start in range(0, query_embeddings.size(0), chunk_size):
        end = start + chunk_size

        query_chunk = query_embeddings[start:end].to(device)

        similarity = query_chunk @ gallery_embeddings.T

        _, topk_indices = torch.topk(
            similarity,
            k=top_k,
            dim=1,
            largest=True,
            sorted=True,
        )

        all_topk_indices.append(topk_indices.cpu())

    return torch.cat(all_topk_indices, dim=0)


def build_results_dictionary(
    query_filenames: Sequence[str],
    gallery_filenames: Sequence[str],
    topk_indices: torch.Tensor,
) -> Dict[str, List[str]]:
    """
    Converts top-k indices into the challenge submission format:

        {
            "query_001.jpg": [
                "gallery_017.jpg",
                "gallery_203.jpg",
                ...
            ]
        }
    """

    if len(query_filenames) != topk_indices.size(0):
        raise ValueError(
            "Number of query filenames does not match number of query embeddings."
        )

    results = {}

    for query_idx, query_name in enumerate(query_filenames):
        gallery_indices = topk_indices[query_idx].tolist()

        ranked_gallery_names = [
            gallery_filenames[gallery_idx]
            for gallery_idx in gallery_indices
        ]

        if len(ranked_gallery_names) != topk_indices.size(1):
            raise ValueError("Each query must have exactly top_k gallery results.")

        results[query_name] = ranked_gallery_names

    return results


def save_results_json(
    results: Dict[str, List[str]],
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved retrieval results to: {path}")


# ============================================================
# Full block 3 pipeline
# ============================================================

@torch.no_grad()
def generate_query_gallery_embeddings(
    model: nn.Module,
    query_loader: Iterable,
    gallery_loader: Iterable,
    config: EmbeddingConfig,
    device: Optional[torch.device] = None,
) -> Tuple[
    torch.Tensor,
    List[str],
    torch.Tensor,
    List[str],
]:
    """
    Extracts query and gallery embeddings.

    The query_loader and gallery_loader must provide filenames.
    Labels are not required for the hidden test set.
    """

    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    print("Extracting query embeddings...")
    query_embeddings, query_filenames, _ = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        use_flip_tta=config.use_flip_tta,
    )

    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_filenames, _ = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        use_flip_tta=config.use_flip_tta,
    )

    if query_filenames is None:
        raise ValueError("query_loader must return filenames.")

    if gallery_filenames is None:
        raise ValueError("gallery_loader must return filenames.")

    if config.save_embeddings:
        save_embedding_file(
            path=config.query_embeddings_path,
            embeddings=query_embeddings,
            filenames=query_filenames,
        )

        save_embedding_file(
            path=config.gallery_embeddings_path,
            embeddings=gallery_embeddings,
            filenames=gallery_filenames,
        )

    return (
        query_embeddings,
        query_filenames,
        gallery_embeddings,
        gallery_filenames,
    )


def make_retrieval_results(
    model: nn.Module,
    query_loader: Iterable,
    gallery_loader: Iterable,
    config: Optional[EmbeddingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[str]]:
    """
    Complete block 3 function.

    Input:
        trained model
        query_loader
        gallery_loader

    Output:
        results dictionary ready for submission.
    """

    if config is None:
        config = EmbeddingConfig()

    if device is None:
        device = get_device()

    (
        query_embeddings,
        query_filenames,
        gallery_embeddings,
        gallery_filenames,
    ) = generate_query_gallery_embeddings(
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        config=config,
        device=device,
    )

    print("Computing top-k similarity search...")

    topk_indices = compute_topk_indices(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        top_k=config.top_k,
        chunk_size=config.similarity_chunk_size,
        device=device,
    )

    results = build_results_dictionary(
        query_filenames=query_filenames,
        gallery_filenames=gallery_filenames,
        topk_indices=topk_indices,
    )

    save_results_json(results, config.results_path)

    return results


# ============================================================
# Optional: validation metrics using embeddings
# ============================================================

def compute_retrieval_metrics_from_embeddings(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    top_k: int = 10,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Use this on your validation split.

    A query is correct at top-k if at least one of the first k gallery
    images has the same identity label.
    """

    topk_indices = compute_topk_indices(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        top_k=top_k,
        chunk_size=chunk_size,
        device=device,
    )

    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()

    retrieved_labels = gallery_labels[topk_indices]

    top1 = (
        retrieved_labels[:, :1] == query_labels.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    top5 = (
        retrieved_labels[:, :5] == query_labels.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    top10 = (
        retrieved_labels[:, :10] == query_labels.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    challenge_score = 600 * top1 + 300 * top5 + 100 * top10

    return {
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "challenge_score": challenge_score,
    }


@torch.no_grad()
def evaluate_retrieval_from_loaders(
    model: nn.Module,
    val_query_loader: Iterable,
    val_gallery_loader: Iterable,
    config: Optional[EmbeddingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Validation version of block 3.

    This requires validation query/gallery loaders to return labels.
    """

    if config is None:
        config = EmbeddingConfig()

    if device is None:
        device = get_device()

    query_embeddings, _, query_labels = extract_embeddings(
        model=model,
        loader=val_query_loader,
        device=device,
        use_flip_tta=config.use_flip_tta,
    )

    gallery_embeddings, _, gallery_labels = extract_embeddings(
        model=model,
        loader=val_gallery_loader,
        device=device,
        use_flip_tta=config.use_flip_tta,
    )

    if query_labels is None:
        raise ValueError("Validation query loader must return labels.")

    if gallery_labels is None:
        raise ValueError("Validation gallery loader must return labels.")

    metrics = compute_retrieval_metrics_from_embeddings(
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        top_k=config.top_k,
        chunk_size=config.similarity_chunk_size,
        device=device,
    )

    return metrics
