import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import sys, os
sys.path.append(".")
from processing.features import load_and_engineer
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1D9E75;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1D9E75;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1D9E75;
        border-bottom: 2px solid #E1F5EE;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached so it doesn't reload on every interaction) ──
@st.cache_data
def load_data():
    conn     = sqlite3.connect("database/eduPulse.db")
    students = pd.read_sql("SELECT * FROM students", conn)
    sessions = pd.read_sql("SELECT * FROM sessions", conn)
    modules  = pd.read_sql("SELECT * FROM modules",  conn)
    conn.close()
    df, _, _ = load_and_engineer()
    return students, sessions, modules, df

students, sessions, modules, df = load_data()
session_modules = sessions.merge(
    modules[["module_id","module_name","difficulty"]], on="module_id"
)


# ── Sidebar filters ──────────────────────────────────────────
st.sidebar.markdown("##  Filters")
st.sidebar.markdown("---")

selected_device = st.sidebar.multiselect(
    "Device Type",
    options=sessions["device_type"].unique().tolist(),
    default=sessions["device_type"].unique().tolist()
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    options=students["gender"].unique().tolist(),
    default=students["gender"].unique().tolist()
)

duration_range = st.sidebar.slider(
    "Study Duration (minutes)",
    min_value=int(sessions["duration_minutes"].min()),
    max_value=int(sessions["duration_minutes"].max()),
    value=(int(sessions["duration_minutes"].min()),
           int(sessions["duration_minutes"].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("**EduPulse v1.0**")
st.sidebar.markdown("DSCE · CSE · 2025-26")

# Apply filters
filtered_sessions = sessions[
    (sessions["device_type"].isin(selected_device)) &
    (sessions["duration_minutes"] >= duration_range[0]) &
    (sessions["duration_minutes"] <= duration_range[1])
]
filtered_students = students[students["gender"].isin(selected_gender)]
filtered_df = df[df["gender"].isin(selected_gender)]


# ── Main title ───────────────────────────────────────────────
st.markdown('<p class="main-title"> EduPulse — Statistical Intelligence Engine</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time analysis of e-learning platform usage · OULAD Dataset · DSCE CSE 2025-26</p>',
            unsafe_allow_html=True)


# ── Page navigation ──────────────────────────────────────────
page = st.selectbox(
    "Navigate to",
    [" Overview", " Statistical Analysis",
     " At-Risk Predictor", " Module Analysis"]
)


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == " Overview":

    # KPI metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students",  f"{len(filtered_students):,}")
    with col2:
        st.metric("Total Sessions",  f"{len(filtered_sessions):,}")
    with col3:
        st.metric("Avg Quiz Score",
                  f"{filtered_sessions['quiz_score'].mean():.1f}")
    with col4:
        st.metric("Completion Rate",
                  f"{filtered_sessions['completed'].mean()*100:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-header">Quiz Score Distribution</p>',
                    unsafe_allow_html=True)
        fig = px.histogram(
            filtered_sessions, x="quiz_score", nbins=30,
            color_discrete_sequence=["#1D9E75"],
            labels={"quiz_score":"Quiz Score","count":"Sessions"}
        )
        fig.add_vline(
            x=filtered_sessions["quiz_score"].mean(),
            line_dash="dash", line_color="red",
            annotation_text=f"Mean: {filtered_sessions['quiz_score'].mean():.1f}"
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">Sessions by Device Type</p>',
                    unsafe_allow_html=True)
        device_counts = filtered_sessions["device_type"].value_counts().reset_index()
        device_counts.columns = ["device","count"]
        fig2 = px.pie(
            device_counts, names="device", values="count",
            color_discrete_sequence=["#1D9E75","#185FA5","#534AB7"]
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Login hour heatmap
    st.markdown('<p class="section-header">When Do Students Study?</p>',
                unsafe_allow_html=True)
    hourly = filtered_sessions.groupby("login_hour")["session_id"].count().reset_index()
    hourly.columns = ["hour","count"]
    fig3 = px.bar(
        hourly, x="hour", y="count",
        color="count", color_continuous_scale="Greens",
        labels={"hour":"Hour of Day","count":"Sessions"}
    )
    fig3.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Gender distribution
    st.markdown('<p class="section-header">Student Demographics</p>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        gender_counts = filtered_students["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender","count"]
        fig4 = px.bar(gender_counts, x="gender", y="count",
                      color_discrete_sequence=["#1D9E75"],
                      labels={"gender":"Gender","count":"Students"})
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        loc_counts = filtered_students["location"].value_counts().reset_index()
        loc_counts.columns = ["location","count"]
        fig5 = px.bar(loc_counts, x="count", y="location",
                      orientation="h",
                      color_discrete_sequence=["#185FA5"],
                      labels={"location":"City","count":"Students"})
        fig5.update_layout(height=300)
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — STATISTICAL ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Statistical Analysis":

    st.markdown('<p class="section-header">T-Test: Completed vs Incomplete Sessions</p>',
                unsafe_allow_html=True)

    completed  = filtered_sessions[filtered_sessions["completed"]==1]["quiz_score"]
    incomplete = filtered_sessions[filtered_sessions["completed"]==0]["quiz_score"]
    t_stat, p_val = stats.ttest_ind(completed, incomplete)

    col1, col2, col3 = st.columns(3)
    col1.metric("Completed Mean",   f"{completed.mean():.2f}")
    col2.metric("Incomplete Mean",  f"{incomplete.mean():.2f}")
    col3.metric("P-Value", f"{p_val:.6f}",
                delta="Significant " if p_val < 0.05 else "Not significant")

    fig_tt = px.box(
        filtered_sessions, x="completed", y="quiz_score",
        color="completed",
        color_discrete_map={0:"#E74C3C", 1:"#1D9E75"},
        labels={"completed":"Completed (1=Yes)","quiz_score":"Quiz Score"}
    )
    fig_tt.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_tt, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">OLS Regression: Predicting Quiz Score</p>',
                unsafe_allow_html=True)

    X = filtered_sessions[["duration_minutes","videos_watched","completed"]].copy()
    y = filtered_sessions["quiz_score"]
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()

    col_r1, col_r2 = st.columns(2)
    col_r1.metric("R-Squared",
                  f"{model.rsquared:.4f}",
                  f"Model explains {model.rsquared*100:.1f}% of variance")
    col_r2.metric("F-Statistic", f"{model.fvalue:.2f}",
                  f"p = {model.f_pvalue:.6f}")

    coef_df = pd.DataFrame({
        "Feature"    : model.params.index,
        "Coefficient": model.params.values.round(4),
        "P-Value"    : model.pvalues.values.round(4),
        "Significant": ["✅" if p < 0.05 else "❌"
                        for p in model.pvalues.values]
    })
    st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">ANOVA: Quiz Score by Device Type</p>',
                unsafe_allow_html=True)

    mob = filtered_sessions[filtered_sessions["device_type"]=="Mobile" ]["quiz_score"]
    des = filtered_sessions[filtered_sessions["device_type"]=="Desktop"]["quiz_score"]
    tab = filtered_sessions[filtered_sessions["device_type"]=="Tablet" ]["quiz_score"]
    f_s, p_a = stats.f_oneway(mob, des, tab)

    col_a1, col_a2 = st.columns(2)
    col_a1.metric("F-Statistic", f"{f_s:.4f}")
    col_a2.metric("P-Value", f"{p_a:.6f}",
                  delta="Significant " if p_a < 0.05 else "Not significant")

    fig_an = px.box(
        filtered_sessions, x="device_type", y="quiz_score",
        color="device_type",
        color_discrete_map={
            "Mobile":"#1D9E75","Desktop":"#185FA5","Tablet":"#534AB7"
        },
        labels={"device_type":"Device","quiz_score":"Quiz Score"}
    )
    fig_an.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_an, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Engagement Score vs Academic Performance</p>',
                unsafe_allow_html=True)

    fig_sc = px.scatter(
        filtered_df, x="engagement_score", y="avg_quiz_score",
        color="risk_flag",
        color_discrete_map={0:"#1D9E75", 1:"#E74C3C"},
        labels={
            "engagement_score":"Engagement Score",
            "avg_quiz_score":"Avg Quiz Score",
            "risk_flag":"At Risk"
        },
        trendline="ols"
    )
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — AT-RISK PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == " At-Risk Predictor":

    st.markdown('<p class="section-header">At-Risk Student Detection</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    at_risk_count = filtered_df["risk_flag"].sum()
    safe_count    = len(filtered_df) - at_risk_count
    col1.metric("Total Students",  len(filtered_df))
    col2.metric("At-Risk ",      at_risk_count)
    col3.metric("Safe ",         safe_count)

    fig_risk = px.pie(
        values=[safe_count, at_risk_count],
        names=["Safe","At-Risk"],
        color_discrete_sequence=["#1D9E75","#E74C3C"]
    )
    fig_risk.update_layout(height=350)
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<p class="section-header">Engagement Score Distribution by Risk</p>',
                unsafe_allow_html=True)
    fig_eng = px.histogram(
        filtered_df, x="engagement_score",
        color="risk_flag",
        color_discrete_map={0:"#1D9E75", 1:"#E74C3C"},
        barmode="overlay", nbins=30,
        labels={"engagement_score":"Engagement Score",
                "risk_flag":"At Risk (1=Yes)"}
    )
    fig_eng.update_layout(height=350)
    st.plotly_chart(fig_eng, use_container_width=True)

    st.markdown('<p class="section-header">At-Risk Students Table</p>',
                unsafe_allow_html=True)
    risk_students = filtered_df[filtered_df["risk_flag"]==1][[
        "student_id","name","engagement_score",
        "avg_quiz_score","completion_rate","total_sessions"
    ]].sort_values("engagement_score").head(20)
    risk_students.columns = [
        "ID","Name","Engagement","Avg Quiz",
        "Completion Rate","Sessions"
    ]
    risk_students = risk_students.round(3)
    st.dataframe(risk_students, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — MODULE ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == " Module Analysis":

    st.markdown('<p class="section-header">Module Popularity</p>',
                unsafe_allow_html=True)

    sm_filtered = filtered_sessions.merge(
        modules[["module_id","module_name","difficulty","category"]],
        on="module_id"
    )
    module_stats = sm_filtered.groupby("module_name").agg(
        sessions      = ("session_id",       "count"),
        avg_score     = ("quiz_score",        "mean"),
        avg_duration  = ("duration_minutes", "mean"),
        completion    = ("completed",         "mean")
    ).reset_index().round(2)

    fig_mod = px.bar(
        module_stats.sort_values("sessions", ascending=True),
        x="sessions", y="module_name",
        orientation="h",
        color="avg_score",
        color_continuous_scale="Greens",
        labels={"sessions":"Sessions","module_name":"Module",
                "avg_score":"Avg Score"}
    )
    fig_mod.update_layout(height=450)
    st.plotly_chart(fig_mod, use_container_width=True)

    st.markdown('<p class="section-header">Module Performance Summary</p>',
                unsafe_allow_html=True)
    module_stats.columns = [
        "Module","Sessions","Avg Score","Avg Duration(min)","Completion Rate"
    ]
    st.dataframe(module_stats, use_container_width=True)

    st.markdown('<p class="section-header">Score vs Duration by Module</p>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        sm_filtered.sample(min(2000, len(sm_filtered))),
        x="duration_minutes", y="quiz_score",
        color="module_name",
        labels={"duration_minutes":"Duration (min)",
                "quiz_score":"Quiz Score",
                "module_name":"Module"},
        opacity=0.6
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)
