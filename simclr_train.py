"""
simclr_train.py
---------------
Fine-tuning self-supervised di InceptionResnetV1 con Triplet Loss (hard negative mining).

Non richiede label di identita': impara embedding continui da sole immagini.

Come funziona:
    Per ogni immagine nel dataset, si generano DUE augmentazioni diverse
    (stessa faccia, aspetto leggermente diverso). La rete impara a produrre
    embedding simili per le due viste della stessa immagine, e diversi per
    immagini diverse presenti nello stesso batch.

    Al termine, viene salvato SOLO il backbone (senza projection head):
    e' quello che usa FaceNetEncoder a inference time.

Struttura del dataset attesa:
    Cartella piatta (es. img_align_celeba/):
        000001.jpg
        000002.jpg
        ...

    Oppure struttura ImageFolder:
        dataset/classe_a/img1.jpg
        dataset/classe_b/img2.jpg

    Le label NON vengono usate in nessun caso.

Uso tipico:
    python simclr_train.py --data-folder /path/to/img_align_celeba \
                           --epochs 10 \
                           --batch-size 256 \
                           --output simclr_checkpoint.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from face_retrieval_model import (
    FaceRetrievalModel,
    ProjectionHead,
    freeze_backbone,
    unfreeze_last_backbone_layers,
    create_optimizer,
    save_checkpoint,
    get_device,
    TrainingConfig,
)
from loss_functions import TripletLoss
from train_finetune import PairDataset, train_simclr_epoch


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR fine-tuning di InceptionResnetV1 su dataset senza label"
    )
    parser.add_argument(
        "--data-folder", required=True,
        help="Cartella con le immagini di training (piatta o ImageFolder)"
    )
    parser.add_argument(
        "--output", default="simclr_checkpoint.pt",
        help="Path dove salvare il checkpoint (default: simclr_checkpoint.pt)"
    )
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=256,
                        help="Batch piu' grande = NT-Xent piu' efficace. Min consigliato: 128.")
    parser.add_argument("--head-lr",        type=float, default=1e-3,
                        help="Learning rate del projection head")
    parser.add_argument("--backbone-lr",    type=float, default=1e-5,
                        help="Learning rate del backbone (fine-tuning)")
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--margin",          type=float, default=0.3,
                        help="Margine Triplet Loss (default 0.3). Aumentare se gli embedding collassano.")
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--image-size",     type=int,   default=160)
    parser.add_argument("--log-every",      type=int,   default=50)
    parser.add_argument(
        "--freeze-stage-epochs", type=int, default=3,
        help="Quante epoch tenere il backbone congelato (stage 1). "
             "Poi si sblocca parzialmente (stage 2). 0 = salta stage 1."
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path a un checkpoint .pt da cui riprendere i pesi del backbone."
    )
    args = parser.parse_args()

    device = get_device()
    print(f"[simclr_train] device = {device}")
    print(f"[simclr_train] data   = {args.data_folder}")
    print(f"[simclr_train] output = {args.output}")

    # --------------------------------------------------------
    # Dataset e DataLoader
    # --------------------------------------------------------
    dataset = PairDataset(root=args.data_folder, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # NT-Xent funziona meglio con batch uniformi
    )

    # --------------------------------------------------------
    # Modello: backbone pretrained + projection head
    # --------------------------------------------------------
    model = FaceRetrievalModel(num_classes=None, pretrained="vggface2").to(device)
    head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128).to(device)
    criterion = TripletLoss(margin=args.margin)

    # --------------------------------------------------------
    # Resume da checkpoint esistente
    # --------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[simclr_train] Pesi caricati da: {args.resume}")

    # --------------------------------------------------------
    # Stage 1: backbone congelato, allena solo il projection head
    # --------------------------------------------------------
    best_loss = float("inf")

    if args.freeze_stage_epochs > 0:
        print(f"\n[Stage 1] Backbone congelato per {args.freeze_stage_epochs} epoch(s)")
        freeze_backbone(model)

        optimizer = torch.optim.AdamW(
            [{"params": head.parameters(), "lr": args.head_lr}],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.freeze_stage_epochs, eta_min=1e-6
        )

        for epoch in range(1, args.freeze_stage_epochs + 1):
            print(f"\n[Epoch {epoch}/{args.freeze_stage_epochs}] (stage 1 — backbone frozen)")
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
            print(f"  => loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {"model_state_dict": model.state_dict(),
                     "simclr_score": avg_loss, "epoch": epoch},
                    args.output,
                )
                print(f"  => Checkpoint salvato in: {args.output}")

    # --------------------------------------------------------
    # Stage 2: sblocca gli ultimi layer del backbone
    # --------------------------------------------------------
    finetune_epochs = args.epochs - args.freeze_stage_epochs
    if finetune_epochs > 0:
        print(f"\n[Stage 2] Fine-tuning ultimi layer per {finetune_epochs} epoch(s)")
        unfreeze_last_backbone_layers(model)

        optimizer = create_optimizer(
            model=model,
            head_lr=args.head_lr * 0.2,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )
        # Aggiungi anche i parametri del projection head all'ottimizzatore
        optimizer.add_param_group({"params": head.parameters(), "lr": args.head_lr})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=1e-6
        )

        for epoch in range(1, finetune_epochs + 1):
            print(f"\n[Epoch {epoch}/{finetune_epochs}] (stage 2 — partial fine-tune)")
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
            print(f"  => loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {"model_state_dict": model.state_dict(),
                     "simclr_score": avg_loss, "epoch": epoch},
                    args.output,
                )
                print(f"  => Nuovo best checkpoint salvato in: {args.output}")

    print(f"\n[simclr_train] Fine training. Best loss: {best_loss:.4f}")
    print(f"[simclr_train] Checkpoint: {args.output}")
    print(f"\nPer usarlo nella competition:")
    print(f"    from facenet_encoder import FaceNetEncoder")
    print(f"    encoder = FaceNetEncoder(device, checkpoint_path='{args.output}')")


if __name__ == "__main__":
    main()
