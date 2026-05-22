import json
import random
from pathlib import Path

import numpy as np
import torch


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(seed_arg: str) -> list[int]:
    return [int(s.strip()) for s in seed_arg.split(",") if s.strip()]
