"""Altair charts of RL training runs, the same for every agent and method (Inventory & RL page)."""

from __future__ import annotations

import altair as alt
import pandas as pd

# The fixed training-sample curves (step = training windows replayed), as metric → chart row.
LEARNING = {"insample_vs_anchor_pct": "cost", "insample_units_vs_anchor_pct": "units"}
# Training-health series: key → (title, reference lines, how to read). Only logged ones are drawn.
PANELS: dict[str, tuple[str, list[float], str]] = {
    "train_vs_anchor_pct": (
        "Sampled batch cost vs anchor (%)", [0.0],
        "Fresh windows and exploration noise every iteration, so read the trend, not single points."),
    "policy_std": (
        "Exploration: policy std", [],
        "Should fall, then flatten. An early collapse stops exploration; no fall at all means no learning signal."),
    "exploration_rate": (
        "Exploration: ε (DQN)", [],
        "Share of random actions; it follows DQN's schedule."),
    "approx_kl": (
        "Update size: KL old → new", [0.02],
        "How far one iteration moved the policy; usually below 0.02. Spikes mean steps are too big."),
    "kl_to_start": (
        "Distance from the start: KL", [],
        "How far training moved from the starting policy (the anchor for residual). Rising while the "
        "training-sample cost is flat means drifting without gain."),
    "clip_frac": (
        "Clipped share", [],
        "Share of decisions where the clipped objective stopped the update. A few % is normal; above ~20% for "
        "long means steps are too big."),
    "grad_norm": (
        "Gradient norm before clipping", [],
        "Above the dashed line (max_grad_norm) the step was clipped to that size."),
    "explained_variance": (
        "PPO critic fit: explained variance", [0.0, 1.0],
        "1 means the critic predicts the returns; 0 or below means no better than a constant, so the "
        "advantages are noise."),
    "group_spread_eur": (
        "GRPO: cost spread within groups (€ / window)", [],
        "Euro differences between the replays of one window, which is all GRPO learns from. It falls as the "
        "policy std falls."),
    "zero_spread_share": (
        "GRPO: groups with identical costs", [],
        "Windows whose replays all cost the same teach nothing (often slow movers, where integer orders coincide)."),
}
# Logged once per PPO / GRPO iteration (step = iteration); the rest use training windows as the step.
PER_ITERATION = frozenset(PANELS) - {"exploration_rate"}
KEYS = (*LEARNING, *PANELS)
# The split view's training and validation colours.
BLOCKS = {"training sample": "#F58518", "validation": "#54A24B"}
_BLOCK_SCALE = alt.Scale(domain=list(BLOCKS), range=list(BLOCKS.values()))


def _x(x_max: float | None, title: str | None = "Training windows replayed") -> alt.X:
    """Linear axis of training windows; passing the same ``x_max`` to every chart keeps them aligned."""
    scale = alt.Scale(domain=[0, x_max], nice=False) if x_max else alt.Scale(zero=True)
    return alt.X("windows:Q", title=title, scale=scale, axis=alt.Axis(format="~s"))


def _rule(y: float) -> alt.Chart:
    return alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")


def _config_color(domain: list[str], legend: bool = True) -> alt.Color:
    return alt.Color("configuration:N", title=None, scale=alt.Scale(domain=domain),
                     legend=alt.Legend(orient="bottom", labelLimit=600) if legend else None)


def learning_chart(curves: pd.DataFrame, marks: pd.DataFrame, y_title: str, domain: list[str],
                   x_max: float | None = None) -> alt.LayerChart:
    """Training-sample curve per seed, with each seed's validation result as a diamond at its final step."""
    color = _config_color(domain)
    layers = [_rule(0.0)]
    if not curves.empty:
        layers.append(alt.Chart(curves).mark_line(point=True).encode(
            x=_x(x_max), y=alt.Y("value:Q", title=y_title), color=color, detail="seed:N",
            tooltip=["configuration:N", "seed:N", alt.Tooltip("windows:Q", format=","),
                     alt.Tooltip("value:Q", title="training sample", format="+.2f")],
        ))
    if not marks.empty:
        layers.append(alt.Chart(marks).mark_point(shape="diamond", size=150, filled=True, opacity=1).encode(
            x=_x(x_max), y=alt.Y("value:Q", title=y_title), color=color,
            tooltip=["configuration:N", "seed:N", alt.Tooltip("value:Q", title="validation", format="+.2f")],
        ))
    return alt.layer(*layers).properties(height=260, width="container")


def panel_chart(df: pd.DataFrame, title: str, refs: list[float], domain: list[str],
                x_max: float | None = None) -> alt.LayerChart:
    """One training-health series: faint raw values and a rolling mean per seed, coloured by configuration."""
    df = df.sort_values("windows").copy()
    df["smooth"] = df.groupby("run")["value"].transform(
        lambda s: s.rolling(max(1, len(s) // 10), center=True, min_periods=1).mean())
    base = alt.Chart(df).encode(x=_x(x_max, None), color=_config_color(domain, legend=False), detail="run:N")
    layers = [
        base.mark_line(opacity=0.25, strokeWidth=1).encode(y=alt.Y("value:Q", title=None)),
        base.mark_line(strokeWidth=2).encode(
            y="smooth:Q",
            tooltip=["configuration:N", "seed:N", alt.Tooltip("windows:Q", format=","),
                     alt.Tooltip("value:Q", format=".4g"), alt.Tooltip("smooth:Q", title="rolling mean", format=".4g")],
        ),
        *[_rule(r) for r in refs],
    ]
    return alt.layer(*layers).properties(title=title, height=170, width="container")


def class_chart(df: pd.DataFrame, value: str, title: str, order: list[str]) -> alt.Chart:
    """Training sample vs validation per demand class (grouped bars)."""
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("demand_class:N", title=None, sort=order),
        xOffset=alt.XOffset("block:N", sort=list(BLOCKS)),
        y=alt.Y(f"{value}:Q", title=title),
        color=alt.Color("block:N", title=None, scale=_BLOCK_SCALE, legend=alt.Legend(orient="bottom")),
        tooltip=["block:N", "demand_class:N", "windows:Q", alt.Tooltip(f"{value}:Q", format="+.1f")],
    ).properties(height=220, width="container")


def time_chart(df: pd.DataFrame, value: str, title: str) -> alt.LayerChart:
    """Per window-start month: the training sample as a line, the validation window as a diamond."""
    enc = {
        "x": alt.X("month:T", title="Window start"),
        "y": alt.Y(f"{value}:Q", title=title),
        "color": alt.Color("block:N", title=None, scale=_BLOCK_SCALE, legend=alt.Legend(orient="bottom")),
        "tooltip": ["block:N", alt.Tooltip("month:T", format="%b %Y"), "windows:Q",
                    alt.Tooltip(f"{value}:Q", format="+.1f")],
    }
    train, val = df[df["block"] == "training sample"], df[df["block"] == "validation"]
    layers = [_rule(0.0), alt.Chart(train).mark_line(point=True).encode(**enc)]
    if not val.empty:
        layers.append(alt.Chart(val).mark_point(shape="diamond", size=150, filled=True, opacity=1).encode(**enc))
    return alt.layer(*layers).properties(height=220, width="container")


def sku_chart(history: pd.DataFrame, points: pd.DataFrame) -> alt.LayerChart:
    """Weekly sales before the decision week, and the forecasts the anchor used."""
    sales = alt.Chart(history).mark_line(point=True, color="#4C78A8").encode(
        x=alt.X("date:T", title=None), y=alt.Y("sales:Q", title="Units per week"),
        tooltip=[alt.Tooltip("date:T", title="week"), alt.Tooltip("sales:Q", title="sales")],
    )
    forecasts = alt.Chart(points).mark_point(filled=True, size=110).encode(
        x="date:T", y="value:Q",
        color=alt.Color("label:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=400)),
        shape=alt.Shape("label:N", legend=None),
        tooltip=["label:N", alt.Tooltip("date:T", title="week"), alt.Tooltip("value:Q", format=".1f")],
    )
    return alt.layer(sales, forecasts).properties(height=230, width="container")


def drivers_chart(df: pd.DataFrame, x_title: str) -> alt.Chart:
    """Signed effect of each input on the policy's correction (units), largest first."""
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("effect_units:Q", title=x_title),
        y=alt.Y("label:N", title=None, sort=None),
        color=alt.when(alt.datum.effect_units > 0).then(alt.value(BLOCKS["training sample"]))
        .otherwise(alt.value("#4C78A8")),
        tooltip=["input:N", alt.Tooltip("value:Q", title="current value (units)", format=".1f"),
                 alt.Tooltip("change:Q", title="raised by (units)", format=".1f"),
                 alt.Tooltip("effect_units:Q", title="effect (units)", format="+.2f")],
    ).properties(height=36 * len(df) + 30, width="container")
