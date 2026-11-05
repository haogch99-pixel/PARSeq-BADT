# PARSeq-BADT

A compact, reproducible release of the Boundary-Aware Degradation Training (BADT) variant of PARSeq with a ViT-8 encoder.

This repository intentionally publishes only the BADT ViT8 implementation, its training configuration, one best checkpoint, and the corresponding benchmark summaries. It does not include ViT12 checkpoints, ablation implementations or weights, baseline weights, or datasets.

## Repository layout

```text
PARSeq-BADT/
├── train.py
├── test.py
├── strhub/
│   ├── data/
│   └── models/
│       ├── parseq/          # Public PARSeq dependency
│       └── parseq_badt/     # BADT ViT8 implementation
├── configs/
│   ├── main.yaml
│   ├── charset/94_full.yaml
│   ├── dataset/
│   ├── model/parseq_badt_vit8.yaml
│   └── experiment/parseq-badt-vit8.yaml
├── checkpoints/badt_vit8_best.ckpt
├── results/subset.csv
└── results/full.csv
```

## Installation

Python 3.9 or newer and PyTorch 2.0 or newer are required. The pinned CPU requirements are suitable for reproducing the code path; install a CUDA build of PyTorch separately when needed.

The checkpoint is stored with Git LFS. Install Git LFS before cloning so that the checkpoint is downloaded instead of only its small pointer file:

```bash
git lfs install
git clone https://github.com/haogch99-pixel/PARSeq-BADT.git
cd PARSeq-BADT
```

```bash
python -m pip install -r requirements/core.txt
python -m pip install -r requirements/train.txt -r requirements/test.txt
python -m pip install -e .
```

## Pretrained inference

After cloning with Git LFS, `checkpoints/badt_vit8_best.ckpt` is available locally and is loaded directly by `test.py`. If the repository was cloned before Git LFS was installed, run `git lfs pull` first:

```bash
git lfs pull
```

```bash
python test.py checkpoints/badt_vit8_best.ckpt \
  --data_root data \
  --device cuda \
  --output_dir results
```

Use `--device cpu` for CPU inference. By default, evaluation follows the repository's lowercase-alphanumeric comparison protocol. Add `--cased --punctuation` when evaluating case-sensitive, punctuation-preserving labels.

The command evaluates both benchmark protocols and writes:

- `results/subset.csv`: IIIT5k, SVT, IC13_857, IC15_1811, SVTP, CUTE80.
- `results/full.csv`: IIIT5k, SVT, IC13_1015, IC15_2077, SVTP, CUTE80.

## Reproducing training

The public training data are not redistributed. Download and preprocess the same LMDB datasets described in [Datasets.md](Datasets.md), placing them below `data/`.

To run the formal BADT ViT8 experiment:

```bash
python train.py +experiment=parseq-badt-vit8 \
  data.root_dir=data \
  trainer.accelerator=gpu \
  trainer.devices=1
```

For a CPU smoke run, override the trainer and batch size:

```bash
python train.py +experiment=parseq-badt-vit8 \
  data.root_dir=data \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32-true \
  model.batch_size=16 \
  data.num_workers=2
```

The exact model and BADT hyperparameters are in [configs/model/parseq_badt_vit8.yaml](configs/model/parseq_badt_vit8.yaml); the formal experiment selector is [configs/experiment/parseq-badt-vit8.yaml](configs/experiment/parseq-badt-vit8.yaml).

## Reported results

The included checkpoint was selected from the formal ViT8 BADT run by validation accuracy. The checked-in summaries are:

| Protocol | Samples | Accuracy | 1 - NED |
|---|---:|---:|---:|
| Subset | 7,248 | 96.29 | 98.95 |
| Full | 7,672 | 95.74 | 98.70 |

The CSV files contain per-dataset accuracy, normalized edit distance, confidence, and mean predicted label length.

## License and attribution

The PARSeq source is retained because BADT builds on it. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the applicable notices and attribution.

