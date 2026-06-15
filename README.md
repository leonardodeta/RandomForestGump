# Random Forest Gump

Face retrieval project for the *Introduction to Machine Learning* exam.
Given a set of query face images and a gallery of candidate images, the system returns the 10 most similar gallery images for each query.

The final, most defensible pipeline is:

1. optional face detection and cropping with MTCNN;
2. face embedding extraction with a pretrained FaceNet-style encoder;
3. cosine similarity ranking;
4. optional validation-controlled post-processing: horizontal-flip TTA, k-reciprocal re-ranking, alpha query expansion, and MMR diversification;
5. JSON submission to the competition server.

> Important: MMR is disabled by default because it can hurt a face-retrieval metric that rewards returning multiple images of the same identity. Enable it only if validation ablations show an improvement.

---

## Final recommendation

For the exam/report, treat the following as the stable final method:

```text
pretrained FaceNet / InceptionResnetV1 + L2-normalized embeddings + cosine similarity
```

Then add optional components only if the validation ablation confirms that they improve the score:

```text
+ horizontal flip TTA
+ k-reciprocal re-ranking
+ alpha query expansion
+ MMR diversification only if it helps
```

The self-supervised fine-tuning script is included as an experimental extension, not as the safest baseline. Since identity labels are not used during two-view self-supervised training, false negatives can occur when two different images of the same person appear in the same batch.

---

## Requirements

Use **Python 3.10 or newer**. The code uses modern type-hint syntax such as `str | Path`, which is not valid in Python 3.9 or older.

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

If your machine already has a CUDA-specific PyTorch installation, check it first with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If needed, install the PyTorch version matching your CUDA setup from the official PyTorch instructions, then install the remaining requirements.

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

where `ground_truth.json` has this format:

```json
{
  "query_001.jpg": ["correct_gallery_01.jpg", "correct_gallery_02.jpg"]
}
```

The validation metric counts every query with a non-empty ground-truth list. If a valid query is missing from the predictions, it is counted as wrong rather than silently ignored.

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

The competition and ablation scripts can also crop automatically with `--auto-crop`.

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

Without a checkpoint:

```bash
python run_ablation.py \
  --val-folder /path/to/validation_folder
```

This prints Top-1, Top-5, Top-10 and the weighted competition score for each configuration.

Recommended table for the report:

| Method | Top-1 | Top-5 | Top-10 | Score |
|---|---:|---:|---:|---:|
| FaceNet cosine | fill from `run_ablation.py` | fill | fill | fill |
| + TTA | fill | fill | fill | fill |
| + k-reciprocal | fill | fill | fill | fill |
| + alpha-QE | fill | fill | fill | fill |
| + MMR | fill | fill | fill | fill |

Use this table to justify which optional components are enabled in the final submission.

---

## Tests and sanity checks

Run the lightweight pipeline test:

```bash
python test_pipeline.py
```

This default test uses a deterministic dummy encoder and does not require pretrained FaceNet weights. It checks the retrieval pipeline, ranking output shape, re-ranking stability, query expansion, and MMR duplicate prevention.

Run utility tests:

```bash
python test_retrieval_utils.py
```

Or, if `pytest` is installed:

```bash
pytest test_retrieval_utils.py
```

Optional real FaceNet integration test:

```bash
python test_pipeline.py --with-facenet
```

This second mode requires `facenet-pytorch` and may need pretrained weights to be available.

---

## Self-supervised fine-tuning

The script `simclr_train.py` supports two explicit losses:

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

Stage 1 trains only the projection head while the backbone is frozen. Because the projection head is not used at inference time, Stage 1 checkpoints are **not** saved as best backbone checkpoints. The saved checkpoint is selected during Stage 2. If `--val-folder` is provided, checkpoint selection uses retrieval score; otherwise it uses self-supervised training loss as a fallback.

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

### Fine-tuning caveat

Self-supervised fine-tuning is not guaranteed to improve a strong pretrained face model. Because the training does not know identity labels, it may incorrectly push apart two images of the same person if they appear in the same batch. Always compare the checkpoint against the pretrained baseline with `run_ablation.py` before using it for the final submission.

---

## Known limitations

- k-reciprocal re-ranking builds a full `(num_query + num_gallery)^2` distance matrix, so it may become memory-heavy on very large galleries.
- MMR is not generally ideal for identity retrieval; it is useful only if validation shows that diversification improves the challenge score.
- Face cropping can help when images contain background, but can hurt if the detector fails or if the dataset is already cleanly cropped.
- The pretrained FaceNet baseline is strong, but the project should still report validation ablations to justify all final choices.

---

## Notes for the report / oral exam

A safe way to present the project is:

> We built a modular face-retrieval pipeline. The core model is a pretrained FaceNet-style encoder that maps each image to a normalized embedding. Retrieval is done with cosine similarity. We then evaluate optional retrieval post-processing methods, such as test-time augmentation, k-reciprocal re-ranking, alpha query expansion, and MMR, using a validation ablation table. The self-supervised training code is an experimental extension and is used only if validation proves it improves over the pretrained baseline.

This framing is honest and technically defensible.
