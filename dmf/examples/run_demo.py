"""
End-to-end demonstration.

    python examples/run_demo.py

Runs the whole loop on synthetic debit-card dispute data:

    1. generate the dataset (known ground truth, planted noise variables)
    2. search the model x variable-count grid with the selection harness
    3. read off the top-3 specifications and the marginal value of each variable
    4. confirm the winner on the untouched holdout
    5. persist the winning pipeline, reload it as a production scorer, and
       score a *drifted* batch containing unseen categories, out-of-range
       numerics, a text value in a numeric column and a missing column
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

warnings.filterwarnings("ignore", category=FutureWarning)

from dmf import Config, ProductionScorer
from dmf.research import ModelSelectionHarness  # noqa: E402
from generate_synthetic_disputes import main as generate  # noqa: E402


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    pd.set_option("display.width", 170, "display.max_columns", 50)

    hr("1. synthetic data")
    train_path, drift_path = generate(out_dir=str(ROOT / "data"), n=12_000, seed=7)

    hr("2. selection harness")
    cfg = Config.from_yaml(ROOT / "configs" / "dispute_fraud.yaml")
    cfg.data.path = train_path
    cfg.run.output_dir = str(ROOT / "artifacts")
    cfg.run.verbose = 0
    result = ModelSelectionHarness(cfg).run()
    print(result.report.render(max_width=170))

    primary = cfg.metrics.primary

    hr(f"3. leaderboard (top 10 of {len(result.leaderboard)} model x k cells)")
    cols = ["rank", "model", "k", f"cv_{primary}_mean", f"cv_{primary}_se",
            "cv_roc_auc_mean", "cv_ks_statistic_mean", "cv_lift_at_top_pct_mean",
            "overfit_gap", "fit_seconds"]
    cols = [c for c in cols if c in result.leaderboard.columns]
    print(result.leaderboard[cols].head(10).to_string(index=False))

    hr("4. best variant per model architecture")
    print(result.best_per_model[[c for c in cols if c != "rank"]].to_string(index=False))

    hr(f"5. top {cfg.selection.top_n} specifications")
    for s in result.top_specs:
        print(f"  {s['rank']}. {s['model']:<16s} k={s['k']:<3d} "
              f"{primary}={s[f'cv_{primary}_mean']:.5f} +/- {s[f'cv_{primary}_se']:.5f}"
              f"   overfit_gap={s['overfit_gap']}")
        print(f"      {', '.join(s['features'])}")

    hr("6. marginal value of the k-th variable (winning architecture)")
    g = result.marginal_gains
    g = g[g["model"] == result.selected_model]
    print(g[["from_k", "to_k", "added_variable", "mean_delta", "relative_delta_pct",
             "n_folds_improved", "n_folds", "p_value", "verdict"]].to_string(index=False))

    hr("7. selected specification and holdout confirmation")
    print(json.dumps(result.selected, indent=2))
    print("\nholdout metrics:")
    print(json.dumps(result.holdout_metrics, indent=2))
    print("\nholdout gains table:")
    print(result.holdout_deciles.to_string(index=False))

    hr("8. what the fitted feature pipeline learned")
    feats = result.fitted_model.named_steps["features"]
    print(feats.summary().to_string(index=False))
    print("\ntraining envelope (the support the guard enforces at inference):")
    print(feats.training_envelope().to_string(index=False))
    iv = feats.information_value()
    if iv is not None and len(iv):
        print("\ninformation value per categorical variable:")
        print(iv.to_string())

    hr("9. production scoring of a drifted batch")
    model_path = Path(cfg.run.output_dir) / cfg.run.name / "model.joblib"
    scorer = ProductionScorer.from_joblib(model_path, top_pct=0.05)
    new = pd.read_csv(drift_path)
    print(f"loaded {model_path.name}: model={scorer.metadata.get('model')}, "
          f"{len(scorer.features)} variables")
    print("schema check:", json.dumps(scorer.check_schema(new), indent=2)[:600])

    scored, report = scorer.score(new)
    print("\nscored head:")
    print(scored.head(8).to_string())
    print("\nbatch data-quality report:")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_column", "schema")}, indent=2))
    guard = scorer.explain_guard(new)
    if len(guard):
        print("\nguard interventions by variable:")
        print(guard.to_string(index=False))

    hr("10. the same batch scored by a model that uses every variable")
    # The winning spec happens to be parsimonious, so most of the injected drift
    # never reaches it. Refitting on the full variable set shows the guard doing
    # its full job -- unseen merchants, a new channel code, a missing column.
    from dmf import build_model_pipeline
    from dmf.research.zoo import build_estimator, config_for_model

    full = pd.read_csv(train_path).drop(columns=["dispute_id"])
    y = full.pop("is_fraudulent_dispute").to_numpy()
    spec = cfg.models["logistic_l2"]
    pipe = build_model_pipeline(config_for_model(cfg, spec), cfg.declared_features,
                                build_estimator(spec, 42, 1, y))
    pipe.fit(full, y)
    wide = ProductionScorer(pipe, top_pct=0.05)
    scored_wide, report_wide = wide.score(new)
    print(json.dumps({k: v for k, v in report_wide.items() if k not in ("by_column", "schema")}, indent=2))
    print("\nguard interventions by variable:")
    print(wide.explain_guard(new).to_string(index=False))
    print(f"\nrows routed to manual review: {(scored_wide['action'] == 'manual_review').sum()} "
          f"of {len(scored_wide)}")

    hr("done")
    print(f"artifacts written to {Path(cfg.run.output_dir) / cfg.run.name}")


if __name__ == "__main__":
    main()
