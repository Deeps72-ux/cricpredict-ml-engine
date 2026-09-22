"""Interactive Streamlit dashboard for CricPredict Match Analytics & Monte Carlo Engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import settings
from app.core.database import MatchAnalyticsDatabase
from app.ml.model_loader import get_model_manager
from app.ml.monte_carlo import MonteCarloSimulator

# Streamlit Page Configuration
st.set_page_config(
    page_title="CricPredict | ML Match Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Stadium Theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .metric-card h4 {
        margin: 0;
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00ff87 0%, #00b4d8 100%);
        color: #0b0f19;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 135, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading cricket database & ML engine...")
def load_resources():
    """Cache database and ML model singletons."""
    db = MatchAnalyticsDatabase.get_instance()
    mgr = get_model_manager()
    sim = MonteCarloSimulator()
    return db, mgr, sim


db, model_mgr, simulator = load_resources()

# Sidebar: Match Configuration
st.sidebar.markdown("### ⚙️ Match Context")

# Preset Selector
preset = st.sidebar.selectbox(
    "Load Match Scenario",
    [
        "Custom Scenario",
        "Wankhede Thriller: MI vs CSK (Need 38 off 18 balls)",
        "Chinnaswamy Chase: RCB vs KKR (105/2 in 10 overs, Chasing 195)",
        "Chepauk Defense: CSK vs GT (Spin choke: 85/5 in 13 overs)",
    ],
)

default_batting = "Royal Challengers Bengaluru"
default_bowling = "Chennai Super Kings"
default_venue = "M Chinnaswamy Stadium, Bengaluru"
default_target = 195
default_score = 105
default_overs = 10.0
default_wickets = 2

if preset == "Wankhede Thriller: MI vs CSK (Need 38 off 18 balls)":
    default_batting = "Mumbai Indians"
    default_bowling = "Chennai Super Kings"
    default_venue = "Wankhede Stadium, Mumbai"
    default_target = 185
    default_score = 147
    default_overs = 17.0
    default_wickets = 4
elif preset == "Chinnaswamy Chase: RCB vs KKR (105/2 in 10 overs, Chasing 195)":
    default_batting = "Royal Challengers Bengaluru"
    default_bowling = "Kolkata Knight Riders"
    default_venue = "M Chinnaswamy Stadium, Bengaluru"
    default_target = 195
    default_score = 105
    default_overs = 10.0
    default_wickets = 2
elif preset == "Chepauk Defense: CSK vs GT (Spin choke: 85/5 in 13 overs)":
    default_batting = "Gujarat Titans"
    default_bowling = "Chennai Super Kings"
    default_venue = "MA Chidambaram Stadium, Chennai"
    default_target = 165
    default_score = 85
    default_overs = 13.0
    default_wickets = 5

teams = db.get_teams() or [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Gujarat Titans",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
]
venues = db.get_venues() or ["Wankhede Stadium, Mumbai", "M Chinnaswamy Stadium, Bengaluru"]

team_batting = st.sidebar.selectbox("Batting Team (Chasing)", teams, index=teams.index(default_batting) if default_batting in teams else 0)
available_bowling = [t for t in teams if t != team_batting]
team_bowling = st.sidebar.selectbox("Bowling Team (Defending)", available_bowling, index=available_bowling.index(default_bowling) if default_bowling in available_bowling else 0)
venue = st.sidebar.selectbox("Venue", venues, index=venues.index(default_venue) if default_venue in venues else 0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 In-Play Scoreboard")

target_runs = st.sidebar.number_input("Target Runs (1st Innings Total + 1)", min_value=20, max_value=350, value=default_target, step=1)
current_score = st.sidebar.number_input("Current Score", min_value=0, max_value=target_runs, value=min(default_score, target_runs), step=1)
overs_completed = st.sidebar.slider("Overs Completed", min_value=0.0, max_value=20.0, value=float(default_overs), step=0.1)
wickets_fallen = st.sidebar.slider("Wickets Lost", min_value=0, max_value=10, value=default_wickets, step=1)
dew_factor = st.sidebar.slider("Dew Factor (0 = None, 1 = Heavy Dew)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

st.sidebar.markdown("---")
iterations = st.sidebar.select_slider(
    "Monte Carlo Iterations",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000,
)

# Header Section
st.markdown("<div class='main-title'>🏏 CricPredict Match Analytics & Monte Carlo Engine</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='subtitle'>Live match state analysis for <b>{team_batting}</b> vs <b>{team_bowling}</b> at <b>{venue}</b></div>",
    unsafe_allow_html=True,
)

# Compute Features & Win Probability
pred_result = model_mgr.predict_win_probability(
    target_runs=target_runs,
    current_score=current_score,
    overs_completed=overs_completed,
    wickets_fallen=wickets_fallen,
    venue=venue,
    dew_factor=dew_factor,
)

fd = pred_result["feature_dict"]
prob_bat = pred_result["win_probability_batting"]
prob_bowl = pred_result["win_probability_bowling"]

# Scoreboard Metrics Row
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"<div class='metric-card'><h4>Target</h4><p>{target_runs}</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><h4>Score</h4><p>{current_score}/{wickets_fallen}</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><h4>Overs</h4><p>{overs_completed}</p></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><h4>Need</h4><p>{fd['runs_needed']} <span style='font-size:0.9rem;color:#9ca3af;'>off {fd['balls_remaining']}b</span></p></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='metric-card'><h4>Req RR</h4><p style='color:#f59e0b;'>{fd['required_run_rate']:.2f}</p></div>", unsafe_allow_html=True)
with c6:
    st.markdown(f"<div class='metric-card'><h4>Curr RR</h4><p style='color:#10b981;'>{fd['current_run_rate']:.2f}</p></div>", unsafe_allow_html=True)

st.write("")

# Layout: 2 Columns for Gauges & Monte Carlo
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("### 🎯 Live Win Probability (XGBoost)")

    # Plotly Gauge Chart
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_bat * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#00ff87", "family": "Inter"}},
            title={"text": f"{team_batting} Win Chance", "font": {"size": 18, "color": "#f3f4f6"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4b5563"},
                "bar": {"color": "#00ff87", "thickness": 0.3},
                "bgcolor": "#1f2937",
                "borderwidth": 1,
                "bordercolor": "#374151",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [40, 60], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [60, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": prob_bat * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb", "family": "Inter"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Team Share Comparison Bar
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            y=["Win %"],
            x=[prob_bat * 100],
            name=team_batting,
            orientation="h",
            marker=dict(color="#00ff87"),
            text=f"{team_batting}: {prob_bat:.1%}",
            textposition="inside",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            y=["Win %"],
            x=[prob_bowl * 100],
            name=team_bowling,
            orientation="h",
            marker=dict(color="#3b82f6"),
            text=f"{team_bowling}: {prob_bowl:.1%}",
            textposition="inside",
        )
    )
    fig_bar.update_layout(
        barmode="stack",
        height=85,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Context Card
    st.markdown(
        f"""
        <div style='background:#111827; border:1px solid #1f2937; border-radius:8px; padding:12px; font-size:0.9rem;'>
            <div>⚡ <b>Phase:</b> {fd['match_phase'].title()} | <b>Pressure Index:</b> {fd['pressure_index']:.2f}</div>
            <div style='margin-top:4px;'>💧 <b>Dew Factor:</b> {dew_factor:.2f} | <b>Boundary Condition:</b> {pred_result.get('boundary_reason') or 'In-Play ML Inference'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(f"### 🎲 Monte Carlo Engine ({iterations:,} Vectorized Runs)")

    # Run Monte Carlo simulation
    sim_result = simulator.simulate_chase(
        target_runs=target_runs,
        current_score=current_score,
        overs_completed=overs_completed,
        wickets_fallen=wickets_fallen,
        iterations=iterations,
        venue=venue,
    )

    mc_win_p = sim_result["win_probability"]
    sim_ms = sim_result["simulation_time_ms"]
    ci = sim_result["confidence_intervals"]
    hist = sim_result["histogram"]

    # Latency badge
    st.markdown(
        f"<span style='background:#064e3b; color:#34d399; font-weight:600; padding:3px 10px; border-radius:12px; font-size:0.8rem;'>⚡ Parallel Execution: {sim_ms:.2f} ms</span>",
        unsafe_allow_html=True,
    )

    # Plotly Score Distribution Histogram
    fig_hist = go.Figure()

    # Bins
    bin_centers = [
        (hist["bin_edges"][i] + hist["bin_edges"][i + 1]) / 2 for i in range(len(hist["counts"]))
    ]
    fig_hist.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist["counts"],
            width=[(hist["bin_edges"][i + 1] - hist["bin_edges"][i]) * 0.9 for i in range(len(hist["counts"]))],
            marker=dict(
                color=bin_centers,
                colorscale=[[0, "#3b82f6"], [0.5, "#6366f1"], [1, "#10b981"]],
                line=dict(color="#1f2937", width=1),
            ),
            name="Projected Final Scores",
            hovertemplate="Score: ~%{x:.0f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add Target Line
    fig_hist.add_vline(
        x=target_runs,
        line_width=3,
        line_dash="dash",
        line_color="#fbbf24",
        annotation_text=f"Target: {target_runs}",
        annotation_position="top right",
    )

    # Add Mean Projected Line
    fig_hist.add_vline(
        x=sim_result["projected_score_mean"],
        line_width=2,
        line_color="#00ff87",
        annotation_text=f"Mean: {sim_result['projected_score_mean']:.0f}",
        annotation_position="top left",
    )

    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9ca3af", "family": "Inter"},
        height=280,
        margin=dict(l=10, r=10, t=30, b=20),
        xaxis=dict(title="Projected 20-Over Score", gridcolor="#1f2937"),
        yaxis=dict(title="Simulation Trajectories", gridcolor="#1f2937"),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Comparison Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h4>MC Win %</h4><p style='color:#00ff87;'>{mc_win_p:.1%}</p></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h4>Mean Score</h4><p>{sim_result['projected_score_mean']:.0f}</p></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h4>50% CI</h4><p style='font-size:1.2rem;'>{ci['50%'][0]:.0f} - {ci['50%'][1]:.0f}</p></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><h4>95% CI</h4><p style='font-size:1.2rem;'>{ci['95%'][0]:.0f} - {ci['95%'][1]:.0f}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# Section 2: Historical Batter vs Bowler Micro-Matchups
st.markdown("### ⚔️ Micro-Matchup Historical Embeddings (Batter vs Bowler)")

batters_list = db.get_unique_batters()
bowlers_list = db.get_unique_bowlers()

b_col1, b_col2 = st.columns(2)
with b_col1:
    default_bat_idx = batters_list.index("V Kohli") if "V Kohli" in batters_list else 0
    selected_batter = st.selectbox("Select Batter", batters_list, index=default_bat_idx)
with b_col2:
    default_bowl_idx = bowlers_list.index("JJ Bumrah") if "JJ Bumrah" in bowlers_list else 0
    selected_bowler = st.selectbox("Select Bowler", bowlers_list, index=default_bowl_idx)

stats = db.get_matchup_stats(selected_batter, selected_bowler)

if stats:
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        st.metric("Balls Faced", stats["balls_faced"])
    with s2:
        st.metric("Runs Scored", stats["runs_scored"])
    with s3:
        st.metric("Dismissals", stats["dismissals"])
    with s4:
        st.metric("Strike Rate", f"{stats['strike_rate']:.1f}")
    with s5:
        st.metric("Dot Ball %", f"{stats['dot_ball_percentage']:.1f}%")
    with s6:
        avg_str = f"{stats['average']:.1f}" if stats["average"] is not None else "∞"
        st.metric("Batting Average", avg_str)

    # Phase Breakdown Chart
    pb = stats["phase_breakdown"]
    phases = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
    p_balls = [pb["powerplay"]["balls"], pb["middle"]["balls"], pb["death"]["balls"]]
    p_runs = [pb["powerplay"]["runs"], pb["middle"]["runs"], pb["death"]["runs"]]
    p_sr = [pb["powerplay"]["strike_rate"], pb["middle"]["strike_rate"], pb["death"]["strike_rate"]]

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Bar(x=phases, y=p_runs, name="Runs Scored", marker_color="#00ff87"))
    fig_phase.add_trace(go.Bar(x=phases, y=p_balls, name="Balls Faced", marker_color="#3b82f6"))
    fig_phase.add_trace(go.Scatter(x=phases, y=p_sr, name="Strike Rate", yaxis="y2", line=dict(color="#f59e0b", width=3)))

    fig_phase.update_layout(
        title=f"{selected_batter} vs {selected_bowler} Across Match Phases",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb", "family": "Inter"},
        height=300,
        yaxis=dict(title="Runs / Balls", gridcolor="#1f2937"),
        yaxis2=dict(title="Strike Rate", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0.2),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_phase, use_container_width=True)
else:
    st.info(f"No historical head-to-head IPL deliveries found between {selected_batter} and {selected_bowler}.")

st.markdown("---")
st.markdown(
    """
    <div style='display:flex; justify-content:space-between; color:#6b7280; font-size:0.85rem;'>
        <div>CricPredict ML Engine v1.0.0 • FastAPI • XGBoost • NumPy Vectorized</div>
        <div><a href='http://localhost:8000/docs' target='_blank' style='color:#38bdf8; text-decoration:none;'>API Swagger Docs ↗</a></div>
    </div>
    """,
    unsafe_allow_html=True,
)
