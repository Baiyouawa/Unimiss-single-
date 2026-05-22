from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"


def mix_mar_mnar_label(mar_ratio: float) -> str:
    mar_pct = int(round(float(mar_ratio) * 100))
    mnar_pct = 100 - mar_pct
    if mar_pct % 10 == 0 and mnar_pct % 10 == 0:
        return f"{mar_pct // 10}:{mnar_pct // 10}"
    return f"{mar_pct}:{mnar_pct}"


def attach_mix_ratio_fields(agg: dict, *, mar_ratio: float = 0.5) -> dict:
    if agg.get("mask_type") != "mix":
        return agg
    mr = float(mar_ratio if "mar_ratio" not in agg else agg["mar_ratio"])
    agg["mar_ratio"] = mr
    agg["mix_ratio_label"] = mix_mar_mnar_label(mr)
    agg["mix_output_tag"] = f"mix_mar{int(round(mr * 100))}"
    return agg


def resolve_output_dir(args) -> Path:
    mar_ratio_tag = f"_mar{int(args.mar_ratio * 100)}" if args.mask_type == "mix" and args.mar_ratio != 0.5 else ""
    return (
        PROJECT_ROOT
        / "outputs"
        / "unimiss"
        / "main"
        / args.dataset
        / f"{args.mask_type}{mar_ratio_tag}"
        / f"mr_{args.missing_rate}"
        / "full"
    )
