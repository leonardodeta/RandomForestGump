# Random Forest Gump

Face retrieval project for the *Introduction to Machine Learning* exam.
Given a set of query face images and a gallery of candidate images, the system returns the 10 most similar gallery images for each query.

## Requirements

Use **Python 3.10 or newer**. The code uses modern type-hint syntax such as `str | Path`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you need CUDA support, install the PyTorch build matching your CUDA version first, then install the remaining requirements.

---

## Expected data layout

For competition or validation:

```text
data_folder/
  query/
    query_001.jpg
    query_002.jpg
  gallery/
    gallery_001.jpg
    gallery_002.jpg
```

For ablation, also include:

```text
data_folder/
  ground_truth.json
```

`ground_truth.json` should look like this:

```json
{
  "query_001.jpg": ["correct_gallery_01.jpg", "correct_gallery_02.jpg"]
}
```

The validation metric counts every query with a non-empty ground-truth list. Missing predictions count as wrong instead of being silently ignored.

---

## Run the competition pipeline

### Safe default: TTA + cosine similarity

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --submit-url http://videosim.disi.unitn.it:3001/retrieval/
```

### Dry run only

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --dry-run
```

This writes:

```text
submission.json
submission_config.json
```

The config file records the exact architecture, image size, checkpoint, and retrieval parameters used for the run.

### Disable TTA for the pure cosine baseline

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --no-tta \
  --dry-run
```

### Enable optional post-processing

Only enable these if validation ablations support them:

```bash
python run_competition.py \
  --data-folder /path/to/data_folder \
  --group-name "random_forest_gump" \
  --kreciprocal \
  --k1 20 \
  --k2 6 \
  --kr-lambda 0.3 \
  --qe \
  --qe-top-k 5 \
  --qe-alpha 3.0 \
  --mmr \
  --mmr-lambda 0.8 \
  --mmr-pool 50 \
  --dry-run
```

Important flags:

```bash
--no-tta                              # disable horizontal flip TTA
--kreciprocal                         # enable k-reciprocal re-ranking
--k1 20 --k2 6 --kr-lambda 0.3        # k-reciprocal parameters
--max-kreciprocal-matrix-elements N   # memory guard for dense re-ranking matrix
--qe --qe-top-k 5 --qe-alpha 3.0      # alpha query expansion
--mmr --mmr-lambda 0.8 --mmr-pool 50  # MMR diversification
--auto-crop                           # crop query/gallery before retrieval
--checkpoint selfsup_checkpoint.pt    # use a validated checkpoint
```

`--no-kreciprocal` is still accepted as a compatibility flag, but k-reciprocal is already disabled unless `--kreciprocal` is passed.

---

## Optional face cropping

If the original images are not centered face crops, crop them first:

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

For `inception_resnet_v2`, use `--image-size 299`.

The competition and ablation scripts can also crop automatically with `--auto-crop`.

---

## Run ablations and parameter analysis

Compact ablation table:

```bash
python run_ablation.py \
  --val-folder /path/to/validation_folder
```

With a checkpoint:

```bash
python run_ablation.py \
  --val-folder /path/to/validation_folder \
  --checkpoint selfsup_checkpoint.pt
```

Full parameter sweep:

```bash
python run_ablation.py \
  --val-folder /path/to/validation_folder \
  --full-grid \
  --k1-values 10,20,30 \
  --k2-values 3,6,10 \
  --kr-lambda-values 0.1,0.3,0.5 \
  --qe-top-k-values 3,5,10 \
  --qe-alpha-values 1.0,2.0,3.0 \
  --mmr-lambda-values 0.5,0.7,0.9 \
  --mmr-pool-values 20,50,100
```

The script saves:

```text
ablation_results.csv
ablation_results.json
```

Those files contain Top-1, Top-5, Top-10, weighted score, and the exact parameters used by each configuration.

## Tests and sanity checks

Run the lightweight tests:

```bash
python test_pipeline.py
python test_retrieval_utils.py
```

Or with pytest:

```bash
pytest test_retrieval_utils.py
```

Optional real FaceNet integration check:

```bash
python test_pipeline.py --with-facenet
```

The default tests do not require downloaded FaceNet weights.

---

## Self-supervised fine-tuning

`simclr_train.py` supports two explicit losses:

- `--loss triplet`: two augmented views of the same image are positives; other batch samples are negatives.
- `--loss ntxent`: true SimCLR / NT-Xent loss.

Example:

```bash
python -u simclr_train.py \
  --data-folder /path/to/training_images \
  --val-folder /path/to/validation_folder \
  --output selfsup_checkpoint.pt \
  --epochs 10 \
  --batch-size 256 \
  --freeze-stage-epochs 3 \
  --loss triplet \
  --margin 0.3 \
  --workers 8 \
  --seed 42
```

Stage 1 trains only the projection head while the backbone is frozen. Since the projection head is not used at inference time, Stage 1 checkpoints are not treated as best retrieval checkpoints. Stage 2 partially fine-tunes the backbone and saves the best checkpoint by validation score when `--val-folder` is provided.

---

## Supervised fine-tuning

`train_finetune.py` is optional and requires identity labels. It now supports both ordinary cross entropy and retrieval-oriented angular-margin losses:

```text
cross_entropy
arcface
cosface
normalized_softmax
```

ArcFace/CosFace are better aligned with cosine retrieval than a plain classifier head, but they should still be selected only after validation.

---

## Known limitations

- k-reciprocal re-ranking builds a dense `(num_query + num_gallery)^2` matrix. The code now has a memory guard, but very large galleries still require cosine ranking or an approximate/chunked method.
- MMR can hurt identity retrieval by diversifying away correct same-person images.
- Face cropping can help with background clutter but can hurt if MTCNN fails or the dataset is already cleanly cropped.
- Self-supervised fine-tuning can create false negatives because identity labels are not known during two-view training.
- The pretrained FaceNet baseline is strong; use validation ablations to justify every optional component.
