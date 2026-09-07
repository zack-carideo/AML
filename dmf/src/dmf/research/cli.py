"""
Command-line entry point: a printing shell over :mod:`dmf.research.api`.

Three subcommands. ``train`` is the default, so the original invocation still
works::

    dmf --config configs/dispute_fraud.yaml
    dmf train --config configs/dispute_fraud.yaml --ordering rfe --tune
    dmf sweep --configs a.yaml b.yaml --output-dir artifacts
    dmf score --model artifacts/dispute_fraud_v1/model.joblib \\
              --data data/disputes_next_month.csv --out scored.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ..config import Config
from ..inference import ProductionScorer
from .api import TRAIN_OVERRIDES, apply_overrides, run_sweep, score, train

_SUBCOMMANDS = {"train", "score", "sweep"}


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dmf",
                                description="Variable selection, model specification search, and scoring.")
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Run the selection harness and persist the winning model.")
    t.add_argument("--config", required=True, help="Path to the YAML configuration.")
    t.add_argument("--data", default=None, help="Override data.path.")
    t.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    t.add_argument("--name", default=None, help="Override run.name.")
    t.add_argument("--ordering", default=None, choices=["importance", "rfe"],
                   help="Override selection.ordering_strategy.")
    t.add_argument("--k-max", type=int, default=None, help="Override selection.k_max.")
    t.add_argument("--top-n", type=int, default=None, help="Override selection.top_n.")
    t.add_argument("--distinct-models", action="store_true",
                   help="Make the top-N list one entry per model rather than the N best cells.")
    t.add_argument("--cv-splits", type=int, default=None, help="Override split.cv.n_splits.")
    t.add_argument("--metric", default=None, help="Override metrics.primary.")
    t.add_argument("--tune", action="store_true", help="Enable hyper-parameter tuning.")
    t.add_argument("--no-tune", action="store_true", help="Disable hyper-parameter tuning.")
    t.add_argument("--n-jobs", type=int, default=None, help="Override run.n_jobs.")
    t.add_argument("--seed", type=int, default=None, help="Override run.random_state.")
    t.add_argument("--quiet", action="store_true", help="Suppress the step report.")

    w = sub.add_parser("sweep", help="Run several configs over the same data and compare on holdout.")
    w.add_argument("--configs", required=True, nargs="+", help="YAML configuration files to sweep.")
    w.add_argument("--data", default=None, help="Override data.path for every config.")
    w.add_argument("--output-dir", default=None, help="Shared run.output_dir for all runs.")
    w.add_argument("--quiet", action="store_true", help="Suppress per-run step reports.")

    s = sub.add_parser("score", help="Load a persisted model and score a new file.")
    s.add_argument("--model", required=True, help="Path to model.joblib written by a training run.")
    s.add_argument("--data", required=True, help="CSV/Parquet of records to score.")
    s.add_argument("--out", default="scored.csv", help="Where to write the scored output.")
    s.add_argument("--id-column", default=None, help="Column to carry through as the record key.")
    s.add_argument("--threshold", type=float, default=None, help="Absolute probability cut.")
    s.add_argument("--top-pct", type=float, default=0.05,
                   help="Relative cut: flag this share of the batch (ignored if --threshold).")
    s.add_argument("--guard-report", default=None, help="Optional path for the JSON data-quality report.")
    s.add_argument("--strict", action="store_true",
                   help="Exit non-zero when the guard marks the batch as review_recommended.")
    return p


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    overrides = {k: v for k, v in vars(args).items() if k in TRAIN_OVERRIDES}
    result = train(args.config, **overrides)
    cfg = result.config

    primary = cfg.metrics.primary
    pd.set_option("display.width", 160, "display.max_columns", 40)
    cols = ["rank", "model", "k", f"cv_{primary}_mean", f"cv_{primary}_se"]
    cols += [c for c in ("overfit_gap", "fit_seconds") if c in result.leaderboard.columns]

    print(f"\n--- leaderboard head (out-of-sample {primary}) ---")
    print(result.leaderboard[cols].head(max(cfg.selection.top_n, 5)).to_string(index=False))

    print(f"\n--- top {cfg.selection.top_n} specifications ---")
    for s in result.top_specs:
        print(f"  {s['rank']}. {s['model']:<16s} k={s['k']:<3d} "
              f"{primary}={s[f'cv_{primary}_mean']:.5f} (+/-{s[f'cv_{primary}_se']:.5f})")

    print(f"\n--- selected specification ({result.selected['rule']}) ---")
    print(f"model     : {result.selected_model}")
    print(f"variables : {result.selected['k']} -> {', '.join(result.selected_features)}")
    print(f"cv {primary}: {result.selected[f'cv_{primary}_mean']:.5f} "
          f"(+/- {result.selected[f'cv_{primary}_se']:.5f} se)")
    print(f"holdout   : {result.holdout_metrics.get(primary)}")
    print(f"\nartifacts : {Path(cfg.run.output_dir) / cfg.run.name}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run each config, then print the holdout comparison table."""
    cfgs = [apply_overrides(Config.from_yaml(path), data=args.data, quiet=args.quiet)
            for path in args.configs]
    comparison, _ = run_sweep(cfgs, output_dir=args.output_dir)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols = [c for c in comparison.columns if c != "features"]
    print("\n--- sweep comparison (ranked on holdout when primaries match) ---")
    print(comparison[cols].to_string(index=False))
    if not comparison["comparable"].all():
        print("\nWARNING: runs are not like-for-like (see RuntimeWarning above); "
              "treat the ranking as indicative only.")
    print(f"\nwritten  : {cfgs[0].run.output_dir}/sweep_comparison.csv")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Load a persisted model and score a file of new records."""
    scorer = ProductionScorer.from_joblib(args.model, threshold=args.threshold, top_pct=args.top_pct)
    scored, report = score(scorer, args.data, out=args.out, id_column=args.id_column)

    print(f"model      : {scorer.metadata.get('model', '(unnamed)')}")
    print(f"variables  : {len(scorer.features)} -> {', '.join(scorer.features)}")
    print(f"scored     : {len(scored)} records -> {args.out}")
    print(f"flagged    : {report['n_flagged']} ({report['flag_rate']:.2%}) at cut "
          f"{report['decision_cut']:.4f}")
    print(f"data check : verdict={report['verdict']}  "
          f"rows_flagged={report.get('n_rows_flagged', 0)}  "
          f"guarded_cell_rate={report.get('guarded_cell_rate', 0):.4f}")
    if report.get("escalation_reason"):
        print(f"escalation : {report['escalation_reason']}")
    if report.get("missing_columns"):
        print(f"missing    : {report['missing_columns']}")
    print(f"review q'd : {report['n_manual_review']} records routed to manual review")

    if args.guard_report:
        Path(args.guard_report).write_text(json.dumps(report, indent=2, default=str))
        print(f"guard json : {args.guard_report}")

    if args.strict and report.get("verdict") != "ok":
        print("strict mode: batch did not pass the data-quality check", file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 1
    if argv[0] not in _SUBCOMMANDS:
        argv = ["train"] + argv          # backwards compatible: a bare --config trains
    args = build_parser().parse_args(argv)
    dispatch = {"score": cmd_score, "sweep": cmd_sweep}
    return dispatch.get(args.command, cmd_train)(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
