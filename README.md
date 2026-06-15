# Random Forest Gump

Face retrieval project for the *Introduction to Machine Learning* exam.
Given a set of query face images and a gallery of candidate images, the system returns the 10 most similar gallery images for each query.

The pipeline is:

1. optional face detection and cropping with MTCNN;
2. face embedding extraction with a pretrained/fine-tuned FaceNet-style encoder;
3. cosine similarity ranking;
4. optional test-time augmentation, k-reciprocal re-ranking, query expansion, and MMR diversification;
5. JSON submission to the competition server.

> Important: MMR is disabled by default because it can hurt a face-retrieval metric that rewards returning multiple images of the same identity. Enable it only if validation ablations show an improvement.

---

## Installation

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

If your machine already has a CUDA-specific PyTorch installation, check it first with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Expected data layout

For competition or validation, the folder should contain:

```text
data_folder/
  query/
    query_001.jpg
    query_002.jpg
  gallery/
    gallery_001.jpg
    gallery_002.jpg
```

For ablation, also add:

```text
data_folder/
  ground_truth.json
```

where `ground_truth.json` has the format:

```json
{
  "query_001.jpg": ["correct_gallery_01.jpg", "correct_gallery_02.jpg"]
}
```

---

## Optional face cropping

If the original images are not already centered face crops, crop them first:

```bash
python crop_faces.py \
  --input /path/to/original/query \
  --output /path/to/cropped/query \
  --image-size 160

python crop_faces.py \
  --input /path/to/original/gallery \
  --output /path/to/cropped/gallery \
  --image-size 160
```

For `inception_resnet_v2`, use `--image-size 299` instead.

The competition script can also crop automatically with `--auto-crop`.

---

## Run the competition pipeline

### Pretrained FaceNet baseline

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --submit-url http://videosim.disi.unitn.it:3001/retrieval/
```

### Dry run only, saving `submission.json`

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --dry-run
```

### With automatic cropping

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --auto-crop \
  --dry-run
```

This creates a cropped copy of the input under `.cropped_competition/` by default and runs retrieval on that copy.

### With a fine-tuned checkpoint

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --checkpoint selfsup_checkpoint.pt \
  --dry-run
```

The checkpoint stores its own backbone architecture, so the loader will automatically restore the correct architecture when possible.

### Optional components

Useful flags:

```bash
--no-tta              # disable horizontal flip test-time augmentation
--no-kreciprocal      # disable k-reciprocal re-ranking
--qe                  # enable alpha query expansion
--mmr                 # enable MMR diversification; use only if validation improves
--arch inception_resnet_v1
--arch inception_resnet_v2
```

---

## Run ablations

```bash
python run_ablation.py \
  --val-folder /path/to/validation_folder \
  --checkpoint selfsup_checkpoint.pt
```

This prints Top-1, Top-5, Top-10 and the weighted competition score for each configuration.
The metric denominator only includes queries that have a valid ground-truth entry.

---

## Self-supervised fine-tuning

The script `simclr_train.py` is kept for compatibility with the original project name, but it now supports two explicit losses:

- `--loss triplet`: two augmented views of the same image are positives; online hard negatives are other samples in the batch.
- `--loss ntxent`: true SimCLR / NT-Xent loss.

Example:

```bash
python -u simclr_train.py \
  --data-folder /path/to/training_images \
  --output selfsup_checkpoint.pt \
  --epochs 10 \
  --batch-size 256 \
  --freeze-stage-epochs 3 \
  --loss triplet \
  --margin 0.3 \
  --workers 8 \
  --log-every 10
```

Stage 1 trains only the projection head while the backbone is frozen. Because the projection head is not used at inference time, Stage 1 checkpoints are **not** saved as best backbone checkpoints. The saved checkpoint is selected during Stage 2. If `--val-folder` is provided, checkpoint selection uses retrieval score; otherwise it uses the self-supervised training loss as a fallback.

Example with validation-based checkpoint selection:

```bash
python -u simclr_train.py \
  --data-folder /path/to/training_images \
  --val-folder /path/to/validation_folder \
  --output selfsup_checkpoint.pt \
  --epochs 10 \
  --freeze-stage-epochs 3 \
  --loss triplet
```

---

## Notes for the report / oral exam

The strongest, most defensible baseline is the pretrained FaceNet encoder with cosine similarity and retrieval-specific post-processing. Self-supervised fine-tuning is experimental: because identity labels are not used, false negatives may occur when two images of the same person appear in the same batch.
