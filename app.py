"""
EvoArena Dashboard — Streamlit UI
Visualises results from a completed EvoArena evolutionary run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EvoArena",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GENOME_TRAITS = [
    "planning_depth",
    "reasoning_steps",
    "cooperation_bias",
    "exploration_rate",
    "verification_level",
]

TRAIT_RANGES = {
    "planning_depth": (1, 5),
    "reasoning_steps": (1, 10),
    "cooperation_bias": (0, 1),
    "exploration_rate": (0, 1),
    "verification_level": (0, 1),
}

TRAIT_LABELS = {
    "planning_depth": "Planning Depth",
    "reasoning_steps": "Reasoning Steps",
    "cooperation_bias": "Cooperation Bias",
    "exploration_rate": "Exploration Rate",
    "verification_level": "Verification Level",
}

TASK_COLORS = {
    "knapsack": "#636EFA",
    "optimization": "#EF553B",
    "prisoners_dilemma": "#00CC96",
}

PLOTLY_TEMPLATE = "plotly_dark"


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


def flatten_agents(data: dict) -> pd.DataFrame:
    rows = []
    for gen_data in data["history"]:
        gen = gen_data["generation"]
        for agent in gen_data["agents"]:
            g = agent["genome"]
            rows.append({
                "generation": gen,
                "agent_id": g["agent_id"],
                "fitness": g["fitness"],
                "mean_accuracy": agent["mean_accuracy"],
                "mean_efficiency": agent["mean_efficiency"],
                "planning_depth": g["planning_depth"],
                "reasoning_steps": g["reasoning_steps"],
                "cooperation_bias": g["cooperation_bias"],
                "exploration_rate": g["exploration_rate"],
                "verification_level": g["verification_level"],
                "parent_ids": g.get("parent_ids", []),
                "n_parents": len(g.get("parent_ids", [])),
            })
    return pd.DataFrame(rows)


def origin_label(parent_ids: list) -> str:
    if not parent_ids:
        return "Seed"
    if len(parent_ids) == 1:
        return "Mutation"
    return "Crossover"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 EvoArena")
    st.caption("Evolutionary Multi-Agent Problem Solving")
    st.divider()

    results_dir = Path("results")
    json_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    final_file = results_dir / "final_results.json"

    file_options = [str(f) for f in json_files]
    default_idx = file_options.index(str(final_file)) if str(final_file) in file_options else 0

    if not file_options:
        st.warning("No results files found in `results/`. Run `uv run main.py` first.")
        st.stop()

    selected_file = st.selectbox("Results file", file_options, index=default_idx)
    data = load_results(selected_file)

    cfg = data.get("config", {})
    run_at = data.get("run_at", "unknown")[:19]
    generations_count = len(data.get("history", []))
    st.markdown(f"""
**Run config**
- Population: `{cfg.get('population_size', '—')}` agents
- Generations: `{cfg.get('generations', generations_count)}`
- Concurrency: `{cfg.get('max_concurrency', '—')}`
- Run at: `{run_at}`
""")
    st.divider()
    st.caption("Tabs: Overview · Leaderboard · Genome · Lineage · Best Agent")

# ── Prepare data ──────────────────────────────────────────────────────────────

df = flatten_agents(data)
df["origin"] = df["parent_ids"].apply(origin_label)

generations = sorted(df["generation"].unique())
best_agent = data.get("best_agent", {})
trajectory = data.get("fitness_trajectory", [])

# per-generation best/mean/worst
gen_stats = (
    df.groupby("generation")["fitness"]
    .agg(["max", "mean", "min"])
    .reset_index()
    .rename(columns={"max": "Best", "mean": "Average", "min": "Worst"})
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_leaderboard, tab_genome, tab_lineage, tab_best = st.tabs([
    "📊 Overview",
    "🏆 Leaderboard",
    "🧬 Genome Analysis",
    "🌳 Lineage",
    "⭐ Best Agent",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Run Overview")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Fitness", f"{max(trajectory):.4f}")
    col2.metric(
        "Fitness Improvement",
        f"{trajectory[-1] - trajectory[0]:+.4f}",
        f"{(trajectory[-1] - trajectory[0]) / trajectory[0] * 100:+.1f}%",
    )
    col3.metric("Generations", len(trajectory))
    col4.metric("Total Agents Evaluated", len(df))

    st.divider()

    # Fitness trajectory
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Fitness Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(trajectory) + 1)),
            y=trajectory,
            mode="lines+markers",
            name="Best Fitness",
            line=dict(color="#00CC96", width=3),
            marker=dict(size=10),
        ))
        fig.add_trace(go.Scatter(
            x=gen_stats["generation"],
            y=gen_stats["Average"],
            mode="lines+markers",
            name="Average Fitness",
            line=dict(color="#636EFA", width=2, dash="dot"),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=gen_stats["generation"],
            y=gen_stats["Worst"],
            mode="lines+markers",
            name="Worst Fitness",
            line=dict(color="#EF553B", width=1, dash="dash"),
            marker=dict(size=5),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Generation",
            yaxis_title="Fitness",
            xaxis=dict(tickmode="linear", dtick=1),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Accuracy vs Efficiency")
        fig2 = px.scatter(
            df,
            x="mean_efficiency",
            y="mean_accuracy",
            color="generation",
            size="fitness",
            hover_data=["agent_id", "fitness", "planning_depth", "reasoning_steps"],
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
            labels={
                "mean_efficiency": "Mean Efficiency",
                "mean_accuracy": "Mean Accuracy",
                "generation": "Generation",
            },
            height=350,
        )
        fig2.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="white")))
        st.plotly_chart(fig2, width="stretch")

    # Fitness distribution per generation
    st.subheader("Fitness Distribution by Generation")
    fig3 = go.Figure()
    colors = px.colors.sequential.Viridis
    for i, gen in enumerate(generations):
        gen_df = df[df["generation"] == gen]
        color = colors[int(i / max(len(generations) - 1, 1) * (len(colors) - 1))]
        fig3.add_trace(go.Violin(
            x=[f"Gen {gen}"] * len(gen_df),
            y=gen_df["fitness"],
            name=f"Gen {gen}",
            box_visible=True,
            points="all",
            meanline_visible=True,
            line_color=color,
            fillcolor=color,
            opacity=0.7,
            spanmode="hard",
        ))
    fig3.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="Fitness",
        showlegend=False,
        height=320,
    )
    st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_leaderboard:
    st.header("Leaderboard")

    gen_filter = st.selectbox(
        "Generation",
        ["All"] + [f"Generation {g}" for g in generations],
        key="lb_gen",
    )
    gen_num = None if gen_filter == "All" else int(gen_filter.split()[-1])
    lb_df = df if gen_num is None else df[df["generation"] == gen_num]

    lb_df = lb_df.sort_values("fitness", ascending=False).reset_index(drop=True)
    lb_df.index += 1

    # Format for display
    display_df = lb_df[[
        "agent_id", "generation", "fitness", "mean_accuracy", "mean_efficiency",
        "planning_depth", "reasoning_steps", "cooperation_bias",
        "exploration_rate", "verification_level", "origin",
    ]].copy()
    display_df.columns = [
        "Agent ID", "Gen", "Fitness", "Accuracy", "Efficiency",
        "Plan Depth", "Reas. Steps", "Coop. Bias",
        "Explor. Rate", "Verif. Level", "Origin",
    ]

    # Highlight top 3
    def highlight_top(s):
        styles = []
        for i, _ in enumerate(s):
            if i == 0:
                styles.append("background-color: #2d4a1e; font-weight: bold")
            elif i == 1:
                styles.append("background-color: #1e3a4a; font-weight: bold")
            elif i == 2:
                styles.append("background-color: #3a2a1e")
            else:
                styles.append("")
        return styles

    def color_fitness(val):
        """Green scale based on fitness value."""
        norm = max(0.0, min(1.0, (val - 0.5) / 0.5))
        g = int(80 + norm * 120)
        return f"background-color: rgba(0,{g},60,0.4); color: white"

    def color_accuracy(val):
        b = int(80 + max(0.0, min(1.0, val)) * 120)
        return f"background-color: rgba(0,60,{b},0.4); color: white"

    def color_efficiency(val):
        p = int(80 + max(0.0, min(1.0, val)) * 100)
        return f"background-color: rgba({p},0,{p},0.4); color: white"

    styled = (
        display_df.style
        .format({
            "Fitness": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Efficiency": "{:.4f}",
            "Coop. Bias": "{:.3f}",
            "Explor. Rate": "{:.3f}",
            "Verif. Level": "{:.3f}",
        })
        .apply(highlight_top, axis=0, subset=["Fitness"])
        .map(color_fitness, subset=["Fitness"])
        .map(color_accuracy, subset=["Accuracy"])
        .map(color_efficiency, subset=["Efficiency"])
    )
    st.dataframe(styled, width="stretch", height=500)

    st.divider()
    st.subheader("Rank Evolution Across Generations")

    # Track agents that appear in multiple generations
    agent_counts = df.groupby("agent_id")["generation"].nunique()
    persistent = agent_counts[agent_counts > 1].index.tolist()

    if persistent:
        rank_rows = []
        for gen in generations:
            gen_df = df[df["generation"] == gen].sort_values("fitness", ascending=False).reset_index(drop=True)
            gen_df.index += 1
            for rank, row in gen_df.iterrows():
                if row["agent_id"] in persistent:
                    rank_rows.append({"generation": gen, "agent_id": row["agent_id"], "rank": rank, "fitness": row["fitness"]})

        rank_df = pd.DataFrame(rank_rows)
        fig_rank = px.line(
            rank_df,
            x="generation",
            y="rank",
            color="agent_id",
            markers=True,
            template=PLOTLY_TEMPLATE,
            labels={"rank": "Rank (lower = better)", "generation": "Generation"},
            height=320,
        )
        fig_rank.update_yaxes(autorange="reversed")
        fig_rank.update_layout(xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig_rank, width="stretch")
    else:
        st.info("No agents persisted across multiple generations to track rank changes.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Genome Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_genome:
    st.header("Genome Analysis")

    # Average trait values per generation
    st.subheader("Trait Evolution Across Generations")

    trait_means = df.groupby("generation")[GENOME_TRAITS].mean().reset_index()

    fig_traits = make_subplots(
        rows=2, cols=3,
        subplot_titles=[TRAIT_LABELS[t] for t in GENOME_TRAITS],
        vertical_spacing=0.15,
    )
    trait_colors = px.colors.qualitative.Plotly
    for idx, trait in enumerate(GENOME_TRAITS):
        row, col = divmod(idx, 3)
        lo, hi = TRAIT_RANGES[trait]
        fig_traits.add_trace(
            go.Scatter(
                x=trait_means["generation"],
                y=trait_means[trait],
                mode="lines+markers",
                name=TRAIT_LABELS[trait],
                line=dict(color=trait_colors[idx], width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=row + 1, col=col + 1,
        )
        # Add range band
        fig_traits.add_hrect(
            y0=lo, y1=hi,
            fillcolor=trait_colors[idx], opacity=0.05,
            row=row + 1, col=col + 1,
        )
    fig_traits.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=False,
    )
    for i in range(1, 7):
        fig_traits.update_xaxes(tickmode="linear", dtick=1, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
    st.plotly_chart(fig_traits, width="stretch")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fitness Correlation with Traits")
        corr = df[GENOME_TRAITS + ["fitness"]].corr()["fitness"].drop("fitness")
        fig_corr = go.Figure(go.Bar(
            x=corr.values,
            y=[TRAIT_LABELS[t] for t in corr.index],
            orientation="h",
            marker=dict(
                color=corr.values,
                colorscale="RdYlGn",
                cmin=-1, cmax=1,
            ),
        ))
        fig_corr.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Pearson Correlation with Fitness",
            height=320,
        )
        st.plotly_chart(fig_corr, width="stretch")

    with col_b:
        st.subheader("Trait vs Fitness (select trait)")
        selected_trait = st.selectbox(
            "Trait",
            GENOME_TRAITS,
            format_func=lambda t: TRAIT_LABELS[t],
        )
        fig_scatter = px.scatter(
            df,
            x=selected_trait,
            y="fitness",
            color="generation",
            hover_data=["agent_id", "mean_accuracy", "mean_efficiency"],
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
            labels={
                selected_trait: TRAIT_LABELS[selected_trait],
                "fitness": "Fitness",
            },
            height=320,
        )
        st.plotly_chart(fig_scatter, width="stretch")

    st.divider()
    st.subheader("Genome Trait Heatmap (all agents, ranked by fitness)")

    heatmap_df = df.sort_values(["generation", "fitness"], ascending=[True, False])
    labels = [f"Gen{r['generation']} {r['agent_id']}" for _, r in heatmap_df.iterrows()]

    # Normalise each trait to 0-1 for the heatmap
    norm_df = heatmap_df[GENOME_TRAITS].copy()
    for t in GENOME_TRAITS:
        lo, hi = TRAIT_RANGES[t]
        norm_df[t] = (norm_df[t] - lo) / (hi - lo)

    fig_hmap = go.Figure(go.Heatmap(
        z=norm_df.values.T,
        x=labels,
        y=[TRAIT_LABELS[t] for t in GENOME_TRAITS],
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(title="Normalised value"),
    ))
    fig_hmap.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_hmap, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Lineage
# ══════════════════════════════════════════════════════════════════════════════

with tab_lineage:
    st.header("Lineage Tree")
    st.caption("Arrows go from parent → child. Node colour = fitness.")

    # Build edges
    edges_x, edges_y = [], []
    node_ids, node_x, node_y, node_fitness, node_gen, node_origin = [], [], [], [], [], []

    # Position: x = generation, y = rank within generation
    pos = {}
    for gen in generations:
        gen_df = df[df["generation"] == gen].sort_values("fitness", ascending=False).reset_index(drop=True)
        for rank, (_, row) in enumerate(gen_df.iterrows()):
            pos[row["agent_id"]] = (gen, -(rank + 1))

    id_to_fitness = dict(zip(df["agent_id"], df["fitness"]))
    id_to_gen = dict(zip(df["agent_id"], df["generation"]))

    for _, row in df.iterrows():
        aid = row["agent_id"]
        x, y = pos[aid]
        node_ids.append(aid)
        node_x.append(x)
        node_y.append(y)
        node_fitness.append(row["fitness"])
        node_gen.append(row["generation"])
        node_origin.append(row["origin"])

        for pid in row["parent_ids"]:
            if pid in pos:
                px_, py_ = pos[pid]
                edges_x += [px_, x, None]
                edges_y += [py_, y, None]

    fig_lin = go.Figure()

    # Edges
    fig_lin.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes
    best_id = best_agent.get("agent_id", "")
    marker_symbols = ["star" if nid == best_id else "circle" for nid in node_ids]
    marker_sizes = [18 if nid == best_id else 12 for nid in node_ids]

    fig_lin.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[nid[:6] for nid in node_ids],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=marker_sizes,
            color=node_fitness,
            colorscale="RdYlGn",
            cmin=df["fitness"].min(),
            cmax=df["fitness"].max(),
            colorbar=dict(title="Fitness"),
            symbol=marker_symbols,
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Fitness: %{marker.color:.4f}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    fig_lin.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis=dict(tickmode="array", tickvals=generations, ticktext=[f"Gen {g}" for g in generations], title="Generation"),
        yaxis=dict(showticklabels=False, title=""),
        height=520,
    )
    st.plotly_chart(fig_lin, width="stretch")

    st.divider()
    st.subheader("Origin Breakdown")
    origin_counts = df.groupby(["generation", "origin"]).size().reset_index(name="count")
    fig_origin = px.bar(
        origin_counts,
        x="generation",
        y="count",
        color="origin",
        barmode="stack",
        color_discrete_map={"Seed": "#636EFA", "Mutation": "#EF553B", "Crossover": "#00CC96"},
        template=PLOTLY_TEMPLATE,
        labels={"generation": "Generation", "count": "# Agents", "origin": "Origin"},
        height=280,
    )
    fig_origin.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig_origin, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Best Agent
# ══════════════════════════════════════════════════════════════════════════════

with tab_best:
    st.header("Best Agent")

    if not best_agent:
        st.warning("No best_agent found in results file.")
    else:
        ba = best_agent
        col1, col2, col3 = st.columns(3)
        col1.metric("Agent ID", ba["agent_id"])
        col2.metric("Fitness", f"{ba['fitness']:.4f}")
        col3.metric("Generation", ba["generation"])

        lineage_str = " → ".join(ba.get("parent_ids", [])) if ba.get("parent_ids") else "Seed (Gen 1)"
        st.caption(f"Lineage: {lineage_str} → **{ba['agent_id']}**")

        st.divider()
        st.subheader("Genome Profile")

        # Radar chart
        trait_vals = [ba[t] for t in GENOME_TRAITS]
        # Normalise to 0-1
        norm_vals = [(ba[t] - TRAIT_RANGES[t][0]) / (TRAIT_RANGES[t][1] - TRAIT_RANGES[t][0])
                     for t in GENOME_TRAITS]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=[TRAIT_LABELS[t] for t in GENOME_TRAITS] + [TRAIT_LABELS[GENOME_TRAITS[0]]],
            fill="toself",
            line_color="#00CC96",
            fillcolor="rgba(0,204,150,0.2)",
            name=ba["agent_id"],
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="rgba(0,0,0,0)",
            ),
            template=PLOTLY_TEMPLATE,
            height=380,
            showlegend=False,
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(fig_radar, width="stretch")

        with col_right:
            st.subheader("Trait Values")
            for trait in GENOME_TRAITS:
                lo, hi = TRAIT_RANGES[trait]
                val = ba[trait]
                norm = (val - lo) / (hi - lo)
                filled = int(norm * 20)
                bar = "█" * filled + "░" * (20 - filled)
                st.markdown(
                    f"`{TRAIT_LABELS[trait]:<20}` **{val}** &nbsp;&nbsp; `[{bar}]`",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Compare Best Agent vs Population Average")

        # Best vs avg across all generations
        pop_avg = df[GENOME_TRAITS].mean()
        norm_avg = [(pop_avg[t] - TRAIT_RANGES[t][0]) / (TRAIT_RANGES[t][1] - TRAIT_RANGES[t][0])
                    for t in GENOME_TRAITS]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Best Agent",
            x=[TRAIT_LABELS[t] for t in GENOME_TRAITS],
            y=norm_vals,
            marker_color="#00CC96",
            opacity=0.8,
        ))
        fig_compare.add_trace(go.Bar(
            name="Population Average",
            x=[TRAIT_LABELS[t] for t in GENOME_TRAITS],
            y=norm_avg,
            marker_color="#636EFA",
            opacity=0.6,
        ))
        fig_compare.update_layout(
            template=PLOTLY_TEMPLATE,
            barmode="group",
            yaxis_title="Normalised Trait Value",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_compare, width="stretch")

        st.divider()
        st.subheader("Strategy Interpretation")

        pd_val = ba["planning_depth"]
        rs_val = ba["reasoning_steps"]
        cb_val = ba["cooperation_bias"]
        er_val = ba["exploration_rate"]
        vl_val = ba["verification_level"]

        insights = []

        if pd_val >= 4:
            insights.append(("🧠 **Deep Planner**", "Uses exact DP for knapsack (max accuracy, high step cost). Long-horizon reasoning in game theory."))
        elif pd_val >= 2:
            insights.append(("⚡ **Efficient Planner**", "Uses greedy heuristics — near-optimal accuracy at a fraction of the step cost."))
        else:
            insights.append(("🎲 **Shallow Planner**", "Relies on random search. Low accuracy but very low step cost."))

        if rs_val >= 7:
            insights.append(("🔁 **High Compute Budget**", f"{rs_val} reasoning steps → {rs_val * 10} hill-climb iterations. High accuracy, lower efficiency."))
        elif rs_val >= 4:
            insights.append(("⚖️ **Balanced Compute**", f"{rs_val} reasoning steps — good accuracy/efficiency trade-off in optimisation tasks."))
        else:
            insights.append(("💨 **Low Compute**", f"Only {rs_val} reasoning steps. Efficient but may miss optimal solutions."))

        if cb_val >= 0.7:
            insights.append(("🤝 **Cooperator**", f"Cooperation bias {cb_val:.2f} → cooperates with Tit-for-Tat → ~3pts/round (near max)."))
        elif cb_val >= 0.4:
            insights.append(("🔀 **Mixed Strategy**", f"Cooperation bias {cb_val:.2f} → mixed behaviour in Prisoner's Dilemma."))
        else:
            insights.append(("🗡️ **Defector**", f"Cooperation bias {cb_val:.2f} → tends to defect. Earns more short-term but loses against TfT."))

        if er_val >= 0.6:
            insights.append(("🔭 **High Explorer**", f"Exploration rate {er_val:.2f} → wide search radius. Can escape local optima but risks random walks."))
        elif er_val >= 0.2:
            insights.append(("🎯 **Focused Explorer**", f"Exploration rate {er_val:.2f} → balanced search. Sweet spot for hill-climbing."))
        else:
            insights.append(("🔒 **Exploiter**", f"Exploration rate {er_val:.2f} → narrow search. May get stuck in local optima."))

        if vl_val >= 0.5:
            insights.append(("✅ **Verifier**", f"Verification level {vl_val:.2f} → triggers polish passes. Boosts accuracy at small step cost."))
        else:
            insights.append(("⏭️ **Non-Verifier**", f"Verification level {vl_val:.2f} → skips polish. Saves steps."))

        for title, body in insights:
            with st.expander(title, expanded=True):
                st.write(body)

        # ── Generalization holdout test ────────────────────────────────────────
        st.divider()
        st.subheader("Generalization Test — Negotiation (Holdout Task)")
        st.caption("This task was never seen during evolution. It tests whether the best genome generalised beyond its training tasks.")

        holdout_path = Path("results") / "holdout_result.json"
        if holdout_path.exists():
            holdout = json.loads(holdout_path.read_text())
            if "error" in holdout:
                st.error(f"Holdout run failed: {holdout['error']}")
            else:
                neg = holdout.get("negotiation", {})
                score = neg.get("accuracy", 0)
                earned = neg.get("total_earned", 0)
                max_score = neg.get("max_score", 30)
                tools = neg.get("tool_calls", [])
                baseline = 0.5

                col1, col2, col3 = st.columns(3)
                col1.metric("Negotiation Score", f"{score:.3f}", delta=f"{score - baseline:+.3f} vs baseline")
                col2.metric("Points Earned", f"{earned} / {max_score}")
                col3.metric("Generalised?", "YES" if score > baseline else "NO",
                            delta="Beat random baseline" if score > baseline else "Below baseline",
                            delta_color="normal" if score > baseline else "inverse")

                if tools:
                    st.markdown("**Tool call sequence:**  `" + "` → `".join(tools) + "`")

                if score > baseline:
                    st.success(f"The best genome scored {score:.1%} on the negotiation task — above the {baseline:.0%} random baseline. Evolution found a strategy that generalises.")
                else:
                    st.warning(f"The best genome scored {score:.1%} — below the {baseline:.0%} baseline. The genome may have overfit to the training tasks.")
        else:
            st.info("No holdout result yet. Run `uv run main.py` to generate it.")
