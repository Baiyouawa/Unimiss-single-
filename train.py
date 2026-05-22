import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import parse_seeds, save_json, set_seed
from common.metrics import summarize_metrics
from common.paths import attach_mix_ratio_fields, mix_mar_mnar_label, resolve_output_dir
from common.result_writer import write_result_md
from dataset.sequence import SequenceDataset
from models.unimiss_model import UniMissModel
from preprocessing.constants import MISSING_LABEL_MAR, MISSING_LABEL_MNAR
from preprocessing.loader import load_dataset_splits
from preprocessing.masking import apply_mask_with_labels
from pygrinder import calc_missing_rate


def reconstruction_loss(x_hat: torch.Tensor, raw_x: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    denom = target_mask.sum().clamp_min(1.0)
    return (((x_hat - raw_x) ** 2) * target_mask).sum() / denom


def gate_regulation_loss(
    beta_om: torch.Tensor,
    beta_mm: torch.Tensor,
    mech_labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    label_mask = target_mask.float()
    mar_target = ((mech_labels == MISSING_LABEL_MAR).float() * label_mask)
    mnar_target = ((mech_labels == MISSING_LABEL_MNAR).float() * label_mask)
    supervise = mar_target + mnar_target
    denom = supervise.sum().clamp_min(1.0)
    beta_om = beta_om.clamp_min(1e-8)
    beta_mm = beta_mm.clamp_min(1e-8)
    loss = -(mar_target * torch.log(beta_om) + mnar_target * torch.log(beta_mm)).sum() / denom
    return loss


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def align_d_model_for_rope_mha(d_model: int, n_heads: int) -> int:
    if n_heads <= 0:
        return d_model
    step = 2 * n_heads
    lo = (d_model // step) * step
    hi = lo + step
    candidates = []
    if lo >= step:
        candidates.append(lo)
    candidates.append(hi)
    return min(candidates, key=lambda x: (abs(x - d_model), x))


def resolve_model_scale(args: argparse.Namespace) -> dict:
    d_model = align_d_model_for_rope_mha(args.d_model, args.n_heads)
    return {
        "d_model": d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
    }


def train_one_seed(
    args: argparse.Namespace,
    model_kwargs: dict,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    seed: int,
    output_dir: Path,
) -> dict:
    set_seed(seed)
    train_masked, train_labels = apply_mask_with_labels(train_raw, args.dataset, args.mask_type, args.missing_rate, seed, mar_ratio=args.mar_ratio)
    val_masked, val_labels = apply_mask_with_labels(val_raw, args.dataset, args.mask_type, args.missing_rate, seed + 1000, mar_ratio=args.mar_ratio)
    test_masked, test_labels = apply_mask_with_labels(test_raw, args.dataset, args.mask_type, args.missing_rate, seed + 2000, mar_ratio=args.mar_ratio)

    print(f"[seed {seed}] After {args.mask_type} masking train: {calc_missing_rate(train_masked):.2%}")
    print(f"[seed {seed}] After {args.mask_type} masking val  : {calc_missing_rate(val_masked):.2%}")
    print(f"[seed {seed}] After {args.mask_type} masking test : {calc_missing_rate(test_masked):.2%}")

    train_ds = SequenceDataset(train_masked, train_raw, train_labels, args.period_len)
    val_ds = SequenceDataset(val_masked, val_raw, val_labels, args.period_len)
    test_ds = SequenceDataset(test_masked, test_raw, test_labels, args.period_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if args.cuda_device is not None:
            torch.cuda.set_device(args.cuda_device)
        torch.cuda.reset_peak_memory_stats()

    model = UniMissModel(n_features=train_raw.shape[-1], **model_kwargs).to(device)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    save_json(seed_dir / "config.json", {"seed": seed, **vars(args), **model_kwargs, "param_count": count_parameters(model)})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    train_wall_begin = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                x=batch["x"],
                mask=batch["obs_mask"],
                phase=batch["phase"],
                density=batch["density"],
                raw_x=batch["raw_x"],
                target_mask=batch["target_mask"],
                time_index=batch["time_index"],
            )
            loss_recon = reconstruction_loss(outputs["x_hat"], batch["raw_x"], batch["target_mask"])
            loss_gate = gate_regulation_loss(
                outputs["beta_om"],
                outputs["beta_mm"],
                batch["mech_labels"],
                batch["target_mask"],
            )
            loss = (
                loss_recon
                + args.lambda_sep * outputs["sep_loss"]
                + args.lambda_gate * loss_gate
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds = []
        val_targets = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    x=batch["x"],
                    mask=batch["obs_mask"],
                    phase=batch["phase"],
                    density=batch["density"],
                    raw_x=batch["raw_x"],
                    target_mask=batch["target_mask"],
                    time_index=batch["time_index"],
                )
                val_preds.append(outputs["x_hat"].cpu().numpy())
                val_targets.append(batch["raw_x"].cpu().numpy())
                val_masks.append(batch["target_mask"].cpu().numpy().astype(bool))
        val_pred = np.concatenate(val_preds, axis=0)
        val_target = np.concatenate(val_targets, axis=0)
        val_mask = np.concatenate(val_masks, axis=0)
        val_metrics = summarize_metrics(val_pred, val_target, val_mask)
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[seed {seed}] epoch={epoch + 1}/{args.epochs} "
            f"train_loss={running_loss / max(len(train_loader), 1):.6f} "
            f"val_rmse={val_metrics['rmse']:.6f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_sec = time.perf_counter() - train_wall_begin

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), seed_dir / "best.pt")

    model.eval()
    test_preds = []
    test_targets = []
    test_masks = []
    infer_begin = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                x=batch["x"],
                mask=batch["obs_mask"],
                phase=batch["phase"],
                density=batch["density"],
                raw_x=batch["raw_x"],
                target_mask=batch["target_mask"],
                time_index=batch["time_index"],
            )
            test_preds.append(outputs["x_hat"].cpu().numpy())
            test_targets.append(batch["raw_x"].cpu().numpy())
            test_masks.append(batch["target_mask"].cpu().numpy().astype(bool))
    infer_time_sec = time.perf_counter() - infer_begin

    test_pred = np.concatenate(test_preds, axis=0)
    test_target = np.concatenate(test_targets, axis=0)
    test_mask = np.concatenate(test_masks, axis=0)
    metrics = summarize_metrics(test_pred, test_target, test_mask)
    metrics["train_time_sec"] = train_time_sec
    metrics["infer_time_sec"] = infer_time_sec
    metrics["peak_gpu_mem_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    metrics["n_parameters"] = int(count_parameters(model)["total"])
    save_json(seed_dir / "metrics.json", metrics)
    return {"seed": seed, **metrics}


def aggregate_runs(args: argparse.Namespace, runs: list[dict], output_dir: Path) -> dict:
    def stat(key: str) -> dict:
        values = np.array([run[key] for run in runs], dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    agg = {
        "dataset": args.dataset,
        "mask_type": args.mask_type,
        "missing_rate": args.missing_rate,
        "mar_ratio": float(args.mar_ratio),
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "mae": stat("mae"),
        "rmse": stat("rmse"),
        "mre": stat("mre"),
        "nrmse": stat("nrmse"),
        "n_points": stat("n_points"),
        "n_parameters": int(runs[0]["n_parameters"]) if "n_parameters" in runs[0] else None,
        "run_name": "full",
        "training_epochs": args.epochs,
    }
    attach_mix_ratio_fields(agg, mar_ratio=args.mar_ratio)
    save_json(output_dir / "metrics_avg.json", agg)
    write_result_md("UniMiss", agg, result_path=output_dir / "result.md")
    return agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["electricity_transformer_temperature", "italy_air_quality"],
        default="electricity_transformer_temperature",
    )
    parser.add_argument("--mask_type", choices=["mar", "mnar_x", "mnar_t", "mix"], default="mar")
    parser.add_argument("--missing_rate", type=float, choices=[0.2, 0.3, 0.4], default=0.2)
    parser.add_argument("--mar_ratio", type=float, default=0.5,
                        help="MAR fraction in mix scenario: 0.2=MNAR-dominant, 0.5=balanced, 0.8=MAR-dominant")
    parser.add_argument("--prep_n_steps", type=int, default=48)
    parser.add_argument("--cuda_device", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60, help="training epochs (main default 60)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--period_len", type=int, default=24, help="period length (main default 24)")
    parser.add_argument("--lambda_sep", type=float, default=0.05)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--gate_temperature", type=float, default=1.0, help="Stage-II gate temperature (main default 1)")
    parser.add_argument("--seeds", type=str, default="3407,3408,3409")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    allowed = {
        "electricity_transformer_temperature": {"mar", "mnar_t", "mix"},
        "italy_air_quality": {"mar", "mnar_x", "mix"},
    }
    if args.mask_type not in allowed[args.dataset]:
        raise ValueError(f"{args.dataset} does not support {args.mask_type}; choose from {allowed[args.dataset]}")

    if args.mask_type == "mix":
        ratio_label = mix_mar_mnar_label(args.mar_ratio)
        print(
            f"[Config] Mix: MAR:MNAR={ratio_label} (mar_ratio={args.mar_ratio:.2f}) | "
            f"missing_rate={args.missing_rate}"
        )

    data = load_dataset_splits(args.dataset, args.prep_n_steps)
    train_raw, val_raw, test_raw = data["train_X"], data["val_X"], data["test_X"]
    print(f"Base missing rate train: {calc_missing_rate(train_raw):.2%}")
    print(f"Base missing rate val  : {calc_missing_rate(val_raw):.2%}")
    print(f"Base missing rate test : {calc_missing_rate(test_raw):.2%}")

    scale_kwargs = resolve_model_scale(args)
    model_kwargs = {
        **scale_kwargs,
        "phase_dim": 2,
        "period_len": args.period_len,
        "dropout": args.dropout,
        "use_oo": True,
        "use_om": True,
        "use_mm": True,
        "use_stage2_gate": True,
        "use_sep_loss": True,
        "use_srne": True,
        "use_topology_expert": True,
        "use_periodic_expert": True,
        "use_extreme_expert": True,
        "gate_temperature": args.gate_temperature,
    }

    output_dir = Path(args.output_dir) if args.output_dir else resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_manifest.json", {**vars(args), **model_kwargs})

    runs = []
    for seed in parse_seeds(args.seeds):
        run = train_one_seed(args, model_kwargs, train_raw, val_raw, test_raw, seed, output_dir)
        runs.append(run)

    aggregate_runs(args, runs, output_dir)


if __name__ == "__main__":
    main()
