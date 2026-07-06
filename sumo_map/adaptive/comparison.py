import pandas as pd
import plotly.express as px

# Load data
df1 = pd.read_csv("stage1_metrics.csv")
df2 = pd.read_csv("stage2_metrics.csv")
df3 = pd.read_csv("stage3_metrics.csv")

# Add stage labels
df1["stage"] = "Stage 1"
df2["stage"] = "Stage 2"
df3["stage"] = "Stage 3"

df = pd.concat([df1, df2, df3])

# 🔥 1. Line Chart (BEST GRAPH)
fig = px.line(
    df,
    x="time",
    y="queue_length",
    color="stage",
    title="Queue Length Over Time"
)
fig.show()

# 🔥 2. Box Plot (VERY IMPRESSIVE)
fig2 = px.box(
    df,
    x="stage",
    y="avg_waiting_time",
    title="Waiting Time Distribution"
)
fig2.show()

# 🔥 3. Heatmap-style density
fig3 = px.density_heatmap(
    df,
    x="queue_length",
    y="avg_waiting_time",
    facet_col="stage",
    title="Traffic Density Analysis"
)
fig3.show()