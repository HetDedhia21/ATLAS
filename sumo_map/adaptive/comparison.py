import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

s1 = pd.read_csv("stage1_summary.csv").iloc[0]
s2 = pd.read_csv("stage2_summary.csv").iloc[0]
s3 = pd.read_csv("stage3_summary.csv").iloc[0]

# Stage 3's own training curve (for the reward chart only)
train3 = pd.read_csv("stage3_metrics.csv")

stages = ["Fixed-Time", "Rule-Based", "Independent Q-Learning"]
colors = ["#636EFA", "#EF553B", "#00CC96"]  # Stage 1 / Stage 2 / Stage 3

# -------------------------------------------------------
# CHART 1 — LOLLIPOP: Avg waiting time across stages
# -------------------------------------------------------

wait_times = [s1["avg_waiting_time"], s2["avg_waiting_time"], s3["avg_waiting_time"]]

fig1 = go.Figure()

# stems
for stage, val, color in zip(stages, wait_times, colors):
    fig1.add_trace(go.Scatter(
        x=[stage, stage], y=[0, val],
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo="skip"
    ))

# lollipop heads
fig1.add_trace(go.Scatter(
    x=stages, y=wait_times,
    mode="markers+text",
    marker=dict(size=20, color=colors, line=dict(color="white", width=2)),
    text=[f"{v:.1f}s" for v in wait_times],
    textposition="top center",
    textfont=dict(size=13, color="black"),
    showlegend=False
))

fig1.update_layout(
    title="Average Waiting Time Across Stages",
    xaxis_title="Control Strategy",
    yaxis_title="Avg Waiting Time (s)",
    template="plotly_white",
    yaxis=dict(rangemode="tozero"),
    width=700, height=480
)
fig1.show()

# -------------------------------------------------------
# CHART 2 — STACKED BAR: Avg queue length across stages
# (Note: each stage only has one queue-length value, so this
# renders as a single-segment stack per bar; the stacked
# bar structure is kept in case you later split queue length
# by approach/junction and want to layer those segments.)
# -------------------------------------------------------

queue_lengths = [s1["avg_queue_length"], s2["avg_queue_length"], s3["avg_queue_length"]]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=stages, y=queue_lengths,
    name="Avg Queue Length",
    marker_color=colors,
    text=[f"{v:.1f}" for v in queue_lengths],
    textposition="auto"
))

fig2.update_layout(
    barmode="stack",
    title="Average Queue Length by Control Strategy",
    xaxis_title="Control Strategy",
    yaxis_title="Avg Queue Length (vehicles)",
    template="plotly_white",
    width=700, height=480
)
fig2.show()

# -------------------------------------------------------
# CHART 3 — SMOOTHED LINE: Stage 3 reward vs training episode
# -------------------------------------------------------

window = max(1, min(5, len(train3) // 3))
train3["reward_smooth"] = (
    train3["total_reward"].rolling(window=window, min_periods=1, center=True).mean()
)

fig3 = go.Figure()

# faint raw curve for context
fig3.add_trace(go.Scatter(
    x=train3["episode"], y=train3["total_reward"],
    mode="lines",
    line=dict(color="lightgray", width=1),
    name="Raw reward"
))

# smoothed curve
fig3.add_trace(go.Scatter(
    x=train3["episode"], y=train3["reward_smooth"],
    mode="lines",
    line=dict(color="#d62728", width=3, shape="spline"),
    name=f"Smoothed (rolling window = {window})"
))

fig3.update_layout(
    title="Stage 3: Episode Reward vs Training Episode",
    xaxis_title="Training Episode",
    yaxis_title="Episode Reward",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    width=800, height=480
)
fig3.show()

# -------------------------------------------------------
# CHART 4 — PROGRESS-BAR STYLE: Fuel consumption across stages
# -------------------------------------------------------

fuel_vals = [s1["fuel_consumption"], s2["fuel_consumption"], s3["fuel_consumption"]]
max_fuel = max(fuel_vals)
baseline = fuel_vals[0]  # Stage 1 (fixed-time) treated as the 100% baseline

fig4 = go.Figure()

# background track (full width = the largest fuel value observed)
fig4.add_trace(go.Bar(
    y=stages, x=[max_fuel] * 3,
    orientation="h",
    marker=dict(color="#e8e8e8"),
    showlegend=False,
    hoverinfo="skip"
))

# filled portion = actual fuel value, with % of baseline as label
fig4.add_trace(go.Bar(
    y=stages, x=fuel_vals,
    orientation="h",
    marker=dict(color=colors),
    showlegend=False,
    text=[f"{v/1e6:.1f}M mL  ({v/baseline*100:.0f}% of Stage 1)" for v in fuel_vals],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(color="white", size=12)
))

fig4.update_layout(
    barmode="overlay",
    title="Estimated Fuel Consumption Across Stages",
    xaxis_title="Fuel Consumption (mL)",
    yaxis_title="Control Strategy",
    template="plotly_white",
    bargap=0.45,
    width=800, height=420
)
fig4.show()