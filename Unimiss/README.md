# UniMiss

## Layout

| Path | Role |
|------|------|
| `Data/` | Raw CSV and TSDB cache (ETT, IAQ) |
| `preprocessing/` | BenchPOTS loading and missingness masks (MAR / MNAR / mix) |
| `dataset/` | PyTorch `SequenceDataset`|
| `models/` | `UniMissModel` |
| `layers/` | `unimiss_layers.py`, `unimiss_modules.py` |
| `common/` | Paths, metrics, I/O, result logging |
| `train.py` | Training loop and CLI |
| `run.py` | Entry point |

Unimiss/
├── Data/
├── preprocessing/
├── dataset/
├── models/
├── layers/
├── common/
├── train.py
├── run.py
└── pixi.toml

## Environment

cd Unimiss
pixi install

## Example (IAQ · mix · 20% missing)

pixi run run-iaq-mix-20


ETT: `mar` / `mnar_t` / `mix` · IAQ: `mar` / `mnar_x` / `mix` · missing rates `0.2` / `0.3` / `0.4`.

## Outputs

- Per run: `outputs/unimiss/main/<dataset>/<mask>/mr_<rate>/full/`
- Summary: `results_main.md` (`UNIMISS_AGGREGATE_RESULT_MD` to override)
