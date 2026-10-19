"""Train + evaluate DQN for multiple project seeds; save per-seed artifacts and a summary."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.evaluate import evaluate_all
from src.rl.train import train_agent

FORECAST_RUN_ID = "78a2063525284f8cae88094bf8873ebf"
SEEDS = [42, 43, 44]
OUT_DIR = ROOT / "artifacts" / "rl" / "seed_replication"


def _config_for_seed(seed: int) -> str:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["project"]["seed"] = seed
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix=f"dqn_seed{seed}_")
    with open(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rl_dir = ROOT / "artifacts" / "rl"

    for seed in SEEDS:
        print(f"\n========== seed={seed} ==========", flush=True)
        cfg_path = _config_for_seed(seed)
        try:
            summary = train_agent(
                "dqn",
                config_path=cfg_path,
                forecast_run_id=FORECAST_RUN_ID,
            )
            eval_out = evaluate_all(
                cfg_path,
                agents=["dqn"],
                forecast_run_id=FORECAST_RUN_ID,
            )
        finally:
            Path(cfg_path).unlink(missing_ok=True)

        deployed = summary.get("deployed_policy", "unknown")
        best_val = summary.get("best_heldout_reward")
        best_step = summary.get("best_heldout_step")
        base_val = summary.get("heldout_base_stock_reward")

        by_policy = {r["policy"]: r for r in eval_out["results"]}
        dqn = by_policy.get("dqn", {})
        bs = by_policy.get("base_stock", {})

        for name in ("dqn_inventory.zip", "dqn_inventory.json", "dqn_inventory_best.zip"):
            src = rl_dir / name
            if src.exists():
                shutil.copy2(src, OUT_DIR / f"seed{seed}_{name}")

        row = {
            "seed": seed,
            "deployed_policy": deployed,
            "val_best_heldout_reward": best_val,
            "val_best_heldout_step": best_step,
            "val_base_stock_reward": base_val,
            "test_dqn_mean_reward_per_sku": dqn.get("mean_reward_per_sku"),
            "test_base_stock_mean_reward_per_sku": bs.get("mean_reward_per_sku"),
            "test_dqn_fill_rate": dqn.get("fill_rate"),
            "test_base_stock_fill_rate": bs.get("fill_rate"),
            "test_dqn_holding_cost_per_sku": dqn.get("holding_cost_per_sku"),
            "test_dqn_shortage_cost_per_sku": dqn.get("shortage_cost_per_sku"),
            "test_base_stock_holding_cost_per_sku": bs.get("holding_cost_per_sku"),
            "test_base_stock_shortage_cost_per_sku": bs.get("shortage_cost_per_sku"),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
