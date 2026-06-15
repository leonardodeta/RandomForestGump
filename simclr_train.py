"""
simclr_train.py
---------------
Self-supervised two-view fine-tuning for the face-retrieval encoder.

The filename is kept for compatibility with the original project, but the script
now makes the loss explicit:

    --loss triplet   online hard-negative triplet loss on two augmented views
    --loss ntxent    true SimCLR / NT-Xent loss

Important implementation detail:
Stage 1 trains only the projection head while the backbone is frozen. Since the
projection head is not used at inference time, Stage 1 is treated as warm-up and
is not saved as the best backbone checkpoint.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

from encoder import Encoder
from face_retrieval_model import (
    FaceRetrievalModel,
    ProjectionHead,
    TrainingConfig,
    create_optimizer,
    freeze_backbone,
    get_device,
    get_face_transform,
    load_checkpoint,
    save_checkpoint,
    unfreeze_last_backbone_layers,
)
from loss_functions import NTXentLoss, TripletLoss
from run_ablation import compute_metrics
from search_system import RetrievalSystem
from train_finetune import PairDataset, train_simclr_epoch


DEFAULT_IMAGE_SIZE = {
    "inception_resnet_v1": 160,
    "inception_resnet_v2": 299,
}

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def load_folder(folder: str | Path):
    """Load all images from a flat folder while preserving filenames.

    A local copy is used here instead of importing run_competition.py so that
    validation inside this training script does not pull in submission/cropping
    dependencies unnecessarily.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    images, filenames = [], []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name.lower().endswith(VALID_EXTENSIONS):
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
            filenames.append(path.name)
    return images, filenames


class InMemoryModelEncoder(Encoder):
    """Small adapter so RetrievalSystem can evaluate an in-memory model."""

    def __init__(self, model: FaceRetrievalModel, device: torch.device, image_size: int):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.transform = get_face_transform(image_size)

    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.embedding_dim))
        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            feats = self.model.encode(tensors, normalize=False)
        return feats.cpu()


def _peek_checkpoint_arch(path: Optional[str], fallback: str) -> str:
    if not path:
        return fallback
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt.get("config", None)
    return ckpt.get("arch") or getattr(config, "arch", None) or fallback


def _make_criterion(args) -> torch.nn.Module:
    if args.loss == "ntxent":
        return NTXentLoss(temperature=args.temperature)
    return TripletLoss(margin=args.margin)


def _evaluate_retrieval_score(
    model: FaceRetrievalModel,
    val_folder: str,
    device: torch.device,
    image_size: int,
    use_tta: bool = True,
) -> Dict[str, float]:
    val_folder = Path(val_folder)
    with open(val_folder / "ground_truth.json", encoding="utf-8") as f:
        import json
        ground_truth = json.load(f)

    query_images, query_filenames = load_folder(val_folder / "query")
    gallery_images, gallery_filenames = load_folder(val_folder / "gallery")

    encoder = InMemoryModelEncoder(model=model, device=device, image_size=image_size)
    system = RetrievalSystem(
        encoder=encoder,
        use_tta=use_tta,
        use_kreciprocal=False,
        use_qe=False,
        use_mmr=False,
        top_k_output=10,
    )
    preds = system.run(
        query_images,
        query_filenames,
        gallery_images,
        gallery_filenames,
        verbose=False,
    )
    return compute_metrics(preds, ground_truth)


def _save_selfsup_checkpoint(
    path: str,
    model: FaceRetrievalModel,
    args,
    *,
    epoch: int,
    stage: str,
    train_loss: float,
    validation_metrics: Optional[Dict[str, float]],
) -> None:
    config = TrainingConfig(num_classes=None, arch=model.arch)
    extra = {
        "training_mode": "self_supervised_two_view",
        "loss_name": args.loss,
        "epoch": epoch,
        "stage": stage,
        "train_loss": train_loss,
        "image_size": DEFAULT_IMAGE_SIZE[model.arch] if model.arch == "inception_resnet_v2" else args.image_size,
        "validation_metrics": validation_metrics,
        "known_limitation": (
            "Two-view self-supervised training can create false negatives when "
            "different images of the same identity appear in the same batch. "
            "Use validation ablations before preferring this checkpoint over the pretrained baseline."
        ),
    }
    save_checkpoint(path, model, config=config, extra=extra)


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised two-view fine-tuning for face retrieval."
    )
    parser.add_argument("--data-folder", required=True,
                        help="Training image folder, flat or ImageFolder-style")
    parser.add_argument("--val-folder", default=None,
                        help="Optional validation folder with query/, gallery/, ground_truth.json")
    parser.add_argument("--output", default="selfsup_checkpoint.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["triplet", "ntxent"], default="triplet",
                        help="Self-supervised loss. Use ntxent for true SimCLR.")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet margin, used only with --loss triplet")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature, used only with --loss ntxent")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--freeze-stage-epochs", type=int, default=3,
                        help="Epochs with frozen backbone. 0 skips Stage 1.")
    parser.add_argument("--resume", default=None,
                        help="Optional checkpoint to resume backbone weights from")
    parser.add_argument("--arch", default="inception_resnet_v1",
                        choices=["inception_resnet_v1", "inception_resnet_v2"],
                        help="Backbone architecture for a fresh run")
    args = parser.parse_args()

    device = get_device()
    resolved_arch = _peek_checkpoint_arch(args.resume, args.arch)
    image_size = DEFAULT_IMAGE_SIZE[resolved_arch] if resolved_arch == "inception_resnet_v2" else args.image_size
    args.image_size = image_size

    print(f"[selfsup_train] device       = {device}")
    print(f"[selfsup_train] data         = {args.data_folder}")
    print(f"[selfsup_train] val          = {args.val_folder}")
    print(f"[selfsup_train] output       = {args.output}")
    print(f"[selfsup_train] arch         = {resolved_arch}")
    print(f"[selfsup_train] image_size   = {image_size}")
    print(f"[selfsup_train] loss         = {args.loss}")
    print(
        "[selfsup_train] warning      = self-supervised positives/negatives are built without identity labels; "
        "false negatives are possible if the same person appears twice in a batch."
    )
    if args.val_folder is None:
        print(
            "[selfsup_train] warning      = no validation folder was provided; checkpoint selection will use training loss only."
        )

    dataset = PairDataset(root=args.data_folder, image_size=image_size)
    if len(dataset) < args.batch_size:
        print(
            f"[selfsup_train] warning      = dataset has {len(dataset)} images, smaller than batch size {args.batch_size}; "
            "drop_last is disabled for this run."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) >= args.batch_size),
    )

    if args.resume:
        model, _, _ = load_checkpoint(args.resume, device)
        if model.classifier is not None:
            model.classifier = None
        print(f"[selfsup_train] Loaded backbone from: {args.resume}")
    else:
        model = FaceRetrievalModel(
            num_classes=None,
            pretrained="vggface2" if resolved_arch == "inception_resnet_v1" else "imagenet",
            arch=resolved_arch,
        ).to(device)

    head = ProjectionHead(
        input_dim=model.embedding_dim,
        hidden_dim=model.embedding_dim,
        output_dim=128,
    ).to(device)
    criterion = _make_criterion(args)

    best_value = float("-inf") if args.val_folder else float("inf")
    saved_any_checkpoint = False

    # --------------------------------------------------------
    # Stage 1: train projection head only; do not save backbone.
    # --------------------------------------------------------
    if args.freeze_stage_epochs > 0:
        print(f"\n[Stage 1] Frozen backbone for {args.freeze_stage_epochs} epoch(s)")
        freeze_backbone(model)

        optimizer = torch.optim.AdamW(
            [{"params": head.parameters(), "lr": args.head_lr}],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.freeze_stage_epochs, eta_min=1e-6
        )

        for epoch in range(1, args.freeze_stage_epochs + 1):
            print(f"\n[Epoch {epoch}/{args.freeze_stage_epochs}] stage 1 — projection-head warm-up")
            stats = train_simclr_epoch(
                backbone=model,
                projection_head=head,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                log_every=args.log_every,
            )
            scheduler.step()
            print(
                f"  => loss: {stats['loss']:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                "no backbone checkpoint saved in Stage 1"
            )

    # --------------------------------------------------------
    # Stage 2: partially fine-tune backbone. Save checkpoints here.
    # --------------------------------------------------------
    finetune_epochs = max(0, args.epochs - args.freeze_stage_epochs)
    if finetune_epochs > 0:
        print(f"\n[Stage 2] Fine-tuning final backbone layers for {finetune_epochs} epoch(s)")
        unfreeze_last_backbone_layers(model)

        optimizer = create_optimizer(
            model=model,
            head_lr=args.head_lr * 0.2,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )
        optimizer.add_param_group({"params": head.parameters(), "lr": args.head_lr})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=1e-6
        )

        for epoch in range(1, finetune_epochs + 1):
            global_epoch = args.freeze_stage_epochs + epoch
            print(f"\n[Epoch {epoch}/{finetune_epochs}] stage 2 — partial fine-tune")
            stats = train_simclr_epoch(
                backbone=model,
                projection_head=head,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                log_every=args.log_every,
            )
            scheduler.step()
            avg_loss = stats["loss"]

            validation_metrics = None
            if args.val_folder:
                validation_metrics = _evaluate_retrieval_score(
                    model=model,
                    val_folder=args.val_folder,
                    device=device,
                    image_size=image_size,
                    use_tta=True,
                )
                current_value = validation_metrics["score"]
                is_better = current_value > best_value
                print(
                    f"  => loss: {avg_loss:.4f} | val score: {current_value:.1f} | "
                    f"top1: {validation_metrics['top1']:.3f} | lr: {scheduler.get_last_lr()[0]:.2e}"
                )
            else:
                current_value = avg_loss
                is_better = current_value < best_value
                print(
                    f"  => loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}"
                )

            if is_better:
                best_value = current_value
                saved_any_checkpoint = True
                _save_selfsup_checkpoint(
                    args.output,
                    model,
                    args,
                    epoch=global_epoch,
                    stage="stage2",
                    train_loss=avg_loss,
                    validation_metrics=validation_metrics,
                )
                metric_name = "validation score" if args.val_folder else "training loss"
                print(f"  => New best checkpoint saved by {metric_name}: {best_value:.4f}")

    if not saved_any_checkpoint:
        print(
            "\n[WARNING] No Stage 2 checkpoint was saved. This usually means epochs <= "
            "freeze_stage_epochs, so only the projection head was trained. Saving the "
            "current backbone for reproducibility, but it may be unchanged."
        )
        _save_selfsup_checkpoint(
            args.output,
            model,
            args,
            epoch=args.epochs,
            stage="stage1_only_or_no_update",
            train_loss=float("nan"),
            validation_metrics=None,
        )

    print(f"\n[selfsup_train] Done. Checkpoint: {args.output}")
    print("Use it with:")
    print(f"    python run_competition.py --data-folder DATA --group-name random_forest_gump --checkpoint {args.output}")


if __name__ == "__main__":
    main()
