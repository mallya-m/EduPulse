import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EduPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size:2.2rem; font-weight:700; color:#1D9E75; }
.sub-title  { font-size:1rem; color:#666; margin-bottom:1.5rem; }
.sec-header { font-size:1.2rem; font-weight:600; color:#1D9E75;
              border-bottom:2px solid #E1F5EE; padding-bottom:4px;
              margin:1.2rem 0 0.8rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    import random
    from faker import Faker
    from datetime import datetime, timedelta
    from sklearn.preprocessing import MinMaxScaler

    fake = Faker("en_IN")
    random.seed(42)
    np.random.seed(42)

    # Load real OULAD student data
    df_raw = pd.read_csv("data_gen/raw/studentInfo.csv")
    df_raw = df_raw[["id_student","gender","age_band"]].head(500).copy()
    df_raw["gender"] = df_raw["gender"].map({"M":"Male","F":"Female"}).fillna("Other")
    age_map = {"0-35":25,"35-55":45,"55<=":60}
    df_raw["age"] = df_raw["age_band"].map(age_map).fillna(25).astype(int)

    cities  = ["Bengaluru","Mumbai","Delhi","Chennai","Hyderabad","Pune","Kolkata","Jaipur"]
    devices = ["Mobile","Desktop","Tablet"]
    df_raw["location"]    = [random.choice(cities)  for _ in range(len(df_raw))]
    df_raw["device_type"] = [random.choice(devices) for _ in range(len(df_raw))]
    df_raw["name"]        = [fake.name()             for _ in range(len(df_raw))]
    df_raw["student_id"]  = range(1, len(df_raw)+1)

    students = df_raw[["student_id","name","age","gender","location","device_type"]]

    modules = pd.DataFrame({
        "module_id"  : range(1,11),
        "module_name": ["Python Basics","Data Science 101","Machine Learning",
                        "Statistics Fundamentals","Deep Learning",
                        "Data Visualization","SQL and Databases",
                        "Web Development","Cloud Computing",
                        "Natural Language Processing"],
        "category"   : ["Programming","Data Science","AI/ML","Mathematics","AI/ML",
                        "Data Science","Programming","Programming","DevOps","AI/ML"],
        "difficulty" : ["Beginner","Beginner","Intermediate","Beginner","Advanced",
                        "Intermediate","Beginner","Intermediate","Intermediate","Advanced"]
    })

    # Synthetic sessions
    n = 8000
    sid      = np.random.randint(1, 501,  size=n)
    mid      = np.random.randint(1, 11,   size=n)
    dur      = np.random.randint(5, 181,  size=n)
    vids     = np.clip((dur/15).astype(int) + np.random.randint(0,3,n), 0, 10)
    scores   = np.clip(dur*0.3 + 20 + np.random.normal(0,15,n), 0, 100).round(2)
    comp     = ((dur > 60) & (np.random.random(n) > 0.3)).astype(int)
    hours    = np.random.randint(0, 24, size=n)
    devs     = np.random.choice(devices, size=n)

    sessions = pd.DataFrame({
        "session_id"      : range(1, n+1),
        "student_id"      : sid,
        "module_id"       : mid,
        "duration_minutes": dur,
        "videos_watched"  : vids,
        "quiz_score"      : scores,
        "completed"       : comp,
        "login_hour"      : hours,
        "device_type"     : devs
    })

    # Feature engineering
    stats_df = sessions.groupby("student_id").agg(
        total_sessions  =("session_id",        "count"),
        avg_duration    =("duration_minutes",  "mean"),
        avg_quiz_score  =("quiz_score",         "mean"),
        total_videos    =("videos_watched",     "sum"),
        completion_rate =("completed",          "mean"),
        unique_modules  =("module_id",          "nunique")
    ).reset_index()

    scaler = MinMaxScaler()
    cols   = ["total_sessions","avg_duration","avg_quiz_score",
              "total_videos","completion_rate","unique_modules"]
    scaled = scaler.fit_transform(stats_df[cols])
    s      = pd.DataFrame(scaled, columns=[f"{c}_s" for c in cols])

    stats_df["engagement_score"] = (
        s["total_sessions_s"]  * 0.15 +
        s["avg_duration_s"]    * 0.20 +
        s["avg_quiz_score_s"]  * 0.30 +
        s["total_videos_s"]    * 0.10 +
        s["completion_rate_s"] * 0.20 +
        s["unique_modules_s"]  * 0.05
    ).round(4)

    stats_df["completion_velocity"] = (
        stats_df["completion_rate"] /
        (stats_df["avg_duration"] / 60 + 0.001)
    ).round(4)

    stats_df["risk_flag"] = (
        (stats_df["avg_quiz_score"]  < 50) |
        (stats_df["completion_rate"] < 0.3) |
        (stats_df["total_sessions"]  < 5)
    ).astype(int)

    df = students.merge(stats_df, on="student_id", how="inner")
    return students, sessions, modules, df


students, sessions, modules, df = load_data()
sm_data = sessions.merge(modules[["module_id","module_name","difficulty"]], on="module_id")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("##  Filters")
sel_device = st.sidebar.multiselect(
    "Device Type", sessions["device_type"].unique().tolist(),
    default=sessions["device_type"].unique().tolist()
)
sel_gender = st.sidebar.multiselect(
    "Gender", students["gender"].unique().tolist(),
    default=students["gender"].unique().tolist()
)
dur_range = st.sidebar.slider(
    "Study Duration (min)",
    int(sessions["duration_minutes"].min()),
    int(sessions["duration_minutes"].max()),
    (int(sessions["duration_minutes"].min()),
     int(sessions["duration_minutes"].max()))
)
st.sidebar.markdown("---")
st.sidebar.markdown("**EduPulse v1.0**  \nDSCE · CSE · 2024-25")

fs = sessions[
    sessions["device_type"].isin(sel_device) &
    (sessions["duration_minutes"] >= dur_range[0]) &
    (sessions["duration_minutes"] <= dur_range[1])
]
fstu = students[students["gender"].isin(sel_gender)]
fdf  = df[df["gender"].isin(sel_gender)]

# ── Title ────────────────────────────────────────────────────
st.markdown('<p class="main-title"> EduPulse — Statistical Intelligence Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time analysis · OULAD Dataset · DSCE CSE 2024-25</p>', unsafe_allow_html=True)

page = st.selectbox("Navigate to", [
    " Overview", " Statistical Analysis",
    " At-Risk Predictor", " Module Analysis"
])


# ── PAGE 1: OVERVIEW ────────────────────────────────────────
if page == " Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Students",  len(fstu))
    c2.metric("Total Sessions",  len(fs))
    c3.metric("Avg Quiz Score",  f"{fs['quiz_score'].mean():.1f}")
    c4.metric("Completion Rate", f"{fs['completed'].mean()*100:.1f}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="sec-header">Quiz Score Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(fs, x="quiz_score", nbins=30,
                           color_discrete_sequence=["#1D9E75"])
        fig.add_vline(x=fs["quiz_score"].mean(), line_dash="dash",
                      line_color="red",
                      annotation_text=f"Mean:{fs['quiz_score'].mean():.1f}")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="sec-header">Sessions by Device</p>', unsafe_allow_html=True)
        dc = fs["device_type"].value_counts().reset_index()
        dc.columns = ["device","count"]
        fig2 = px.pie(dc, names="device", values="count",
                      color_discrete_sequence=["#1D9E75","#185FA5","#534AB7"])
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="sec-header">When Do Students Study?</p>', unsafe_allow_html=True)
    hourly = fs.groupby("login_hour")["session_id"].count().reset_index()
    hourly.columns = ["hour","count"]
    fig3 = px.bar(hourly, x="hour", y="count",
                  color="count", color_continuous_scale="Greens",
                  labels={"hour":"Hour of Day","count":"Sessions"})
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="sec-header">Gender Distribution</p>', unsafe_allow_html=True)
        gc = fstu["gender"].value_counts().reset_index()
        gc.columns = ["gender","count"]
        fig4 = px.bar(gc, x="gender", y="count",
                      color_discrete_sequence=["#1D9E75"])
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        st.markdown('<p class="sec-header">Students by City</p>', unsafe_allow_html=True)
        lc = fstu["location"].value_counts().reset_index()
        lc.columns = ["city","count"]
        fig5 = px.bar(lc, x="count", y="city", orientation="h",
                      color_discrete_sequence=["#185FA5"])
        fig5.update_layout(height=300)
        st.plotly_chart(fig5, use_container_width=True)


# ── PAGE 2: STATISTICAL ANALYSIS ────────────────────────────
elif page == " Statistical Analysis":
    st.markdown('<p class="sec-header">T-Test: Completed vs Incomplete Sessions</p>', unsafe_allow_html=True)
    comp_g   = fs[fs["completed"]==1]["quiz_score"]
    incomp_g = fs[fs["completed"]==0]["quiz_score"]
    t_s, p_v = stats.ttest_ind(comp_g, incomp_g)
    c1,c2,c3 = st.columns(3)
    c1.metric("Completed Mean",  f"{comp_g.mean():.2f}")
    c2.metric("Incomplete Mean", f"{incomp_g.mean():.2f}")
    c3.metric("P-Value", f"{p_v:.6f}",
              delta="✅ Significant" if p_v < 0.05 else "❌ Not significant")
    fig_t = px.box(fs, x="completed", y="quiz_score", color="completed",
                   color_discrete_map={0:"#E74C3C",1:"#1D9E75"})
    fig_t.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec-header">OLS Regression: What Predicts Quiz Score?</p>', unsafe_allow_html=True)
    X     = fs[["duration_minutes","videos_watched","completed"]].copy()
    y     = fs["quiz_score"]
    Xc    = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    c1,c2 = st.columns(2)
    c1.metric("R-Squared", f"{model.rsquared:.4f}",
              f"Explains {model.rsquared*100:.1f}% of variance")
    c2.metric("F-Statistic", f"{model.fvalue:.2f}",
              f"p = {model.f_pvalue:.6f}")
    coef_df = pd.DataFrame({
        "Feature"    : model.params.index,
        "Coefficient": model.params.values.round(4),
        "P-Value"    : model.pvalues.round(4),
        "Significant": ["✅" if p < 0.05 else "❌" for p in model.pvalues]
    })
    st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec-header">ANOVA: Quiz Score by Device Type</p>', unsafe_allow_html=True)
    mob = fs[fs["device_type"]=="Mobile" ]["quiz_score"]
    des = fs[fs["device_type"]=="Desktop"]["quiz_score"]
    tab = fs[fs["device_type"]=="Tablet" ]["quiz_score"]
    f_s2, p_a = stats.f_oneway(mob, des, tab)
    c1,c2 = st.columns(2)
    c1.metric("F-Statistic", f"{f_s2:.4f}")
    c2.metric("P-Value", f"{p_a:.6f}",
              delta=" Significant" if p_a < 0.05 else "Not significant")
    fig_an = px.box(fs, x="device_type", y="quiz_score", color="device_type",
                    color_discrete_map={"Mobile":"#1D9E75","Desktop":"#185FA5","Tablet":"#534AB7"})
    fig_an.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_an, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec-header">Engagement Score vs Performance</p>', unsafe_allow_html=True)
    fig_sc = px.scatter(fdf, x="engagement_score", y="avg_quiz_score",
                        color="risk_flag",
                        color_discrete_map={0:"#1D9E75",1:"#E74C3C"},
                        trendline="ols",
                        labels={"engagement_score":"Engagement Score",
                                "avg_quiz_score":"Avg Quiz Score",
                                "risk_flag":"At Risk"})
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)


# ── PAGE 3: AT-RISK PREDICTOR ────────────────────────────────
elif page == " At-Risk Predictor":
    st.markdown('<p class="sec-header">At-Risk Student Detection</p>', unsafe_allow_html=True)
    ar  = fdf["risk_flag"].sum()
    saf = len(fdf) - ar
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Students", len(fdf))
    c2.metric("At-Risk ",     ar)
    c3.metric("Safe ",        saf)

    fig_p = px.pie(values=[saf, ar], names=["Safe","At-Risk"],
                   color_discrete_sequence=["#1D9E75","#E74C3C"])
    fig_p.update_layout(height=350)
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<p class="sec-header">Engagement Score by Risk Group</p>', unsafe_allow_html=True)
    fig_h = px.histogram(fdf, x="engagement_score", color="risk_flag",
                         barmode="overlay", nbins=30,
                         color_discrete_map={0:"#1D9E75",1:"#E74C3C"})
    fig_h.update_layout(height=350)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<p class="sec-header">At-Risk Students Table</p>', unsafe_allow_html=True)
    risk_tbl = fdf[fdf["risk_flag"]==1][[
        "student_id","name","engagement_score",
        "avg_quiz_score","completion_rate","total_sessions"
    ]].sort_values("engagement_score").head(20).round(3)
    risk_tbl.columns = ["ID","Name","Engagement","Avg Quiz","Completion","Sessions"]
    st.dataframe(risk_tbl, use_container_width=True)


# ── PAGE 4: MODULE ANALYSIS ──────────────────────────────────
elif page == " Module Analysis":
    fs_mod = fs.merge(modules[["module_id","module_name"]], on="module_id")
    mod_stats = fs_mod.groupby("module_name").agg(
        sessions    =("session_id",        "count"),
        avg_score   =("quiz_score",         "mean"),
        avg_duration=("duration_minutes",  "mean"),
        completion  =("completed",          "mean")
    ).reset_index().round(2)

    st.markdown('<p class="sec-header">Module Popularity</p>', unsafe_allow_html=True)
    fig_m = px.bar(
        mod_stats.sort_values("sessions", ascending=True),
        x="sessions", y="module_name", orientation="h",
        color="avg_score", color_continuous_scale="Greens",
        labels={"sessions":"Sessions","module_name":"Module"}
    )
    fig_m.update_layout(height=450)
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown('<p class="sec-header">Module Performance Table</p>', unsafe_allow_html=True)
    mod_stats.columns = ["Module","Sessions","Avg Score","Avg Duration","Completion"]
    st.dataframe(mod_stats, use_container_width=True)

    st.markdown('<p class="sec-header">Score vs Duration</p>', unsafe_allow_html=True)
    fig_sc2 = px.scatter(
        fs_mod.sample(min(2000,len(fs_mod))),
        x="duration_minutes", y="quiz_score",
        color="module_name", opacity=0.5,
        labels={"duration_minutes":"Duration (min)","quiz_score":"Quiz Score"}
    )
    fig_sc2.update_layout(height=450)
    st.plotly_chart(fig_sc2, use_container_width=True)
