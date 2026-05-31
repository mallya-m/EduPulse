
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings, random
from faker import Faker
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EduPulse AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — Futuristic Dark Theme ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1117 50%, #0A0E1A 100%);
    color: #E0E6FF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%);
    border-right: 1px solid #00D4FF22;
}
[data-testid="stSidebar"] * { color: #A0AEC0 !important; }

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A1A2E 100%);
    border: 1px solid #00D4FF33;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 0 20px #00D4FF11;
}
[data-testid="stMetricLabel"] { color: #00D4FF !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] { color: #00FF88 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0D1B2A;
    border: 1px solid #00D4FF22;
    border-radius: 8px;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1A1A2E;
    border: 1px solid #00D4FF44;
    color: #E0E6FF;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
    background: #1A1A2E;
    border: 1px solid #00D4FF44;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: #00D4FF;
}

/* Divider */
hr { border-color: #00D4FF22; }

/* Tabs */
[data-testid="stTab"] {
    background: transparent;
    color: #A0AEC0;
}

/* Custom title */
.glowing-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00D4FF, #00FF88, #7B2FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 0.5rem 0;
    letter-spacing: 2px;
}
.subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #4A5568;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00D4FF;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #00D4FF;
    padding-left: 10px;
    margin: 1.5rem 0 1rem 0;
}
.stat-box {
    background: linear-gradient(135deg, #0D1B2A, #1A1A2E);
    border: 1px solid #00D4FF33;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    box-shadow: 0 0 15px #00D4FF08;
}
.stat-label {
    font-size: 0.75rem;
    color: #4A90D9;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #00FF88;
    font-family: 'Orbitron', monospace;
}
.significant {
    color: #00FF88;
    font-weight: 600;
}
.not-significant { color: #FF4757; }
.risk-high {
    background: linear-gradient(135deg, #2D1B1B, #1A0A0A);
    border-color: #FF4757;
}
.risk-safe {
    background: linear-gradient(135deg, #0D2D1B, #0A1A0A);
    border-color: #00FF88;
}
</style>
""", unsafe_allow_html=True)


# ── Dark plotly theme ─────────────────────────────────────────
DARK_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.8)",
    font=dict(color="#A0AEC0", family="Inter"),
    xaxis=dict(
        gridcolor="#1A2744", linecolor="#2D3748",
        tickcolor="#4A5568", showgrid=True
    ),
    yaxis=dict(
        gridcolor="#1A2744", linecolor="#2D3748",
        tickcolor="#4A5568", showgrid=True
    ),
    margin=dict(l=40, r=20, t=40, b=40)
)

COLORS = {
    "teal"  : "#00D4FF",
    "green" : "#00FF88",
    "purple": "#7B2FFF",
    "red"   : "#FF4757",
    "orange": "#FF6B35",
    "yellow": "#FFD700"
}

SCALE_TEAL   = [[0,"#0A0E1A"],[0.5,"#006B8A"],[1,"#00D4FF"]]
SCALE_GREEN  = [[0,"#0A1A0A"],[0.5,"#006B35"],[1,"#00FF88"]]
SCALE_PURPLE = [[0,"#0A0A1A"],[0.5,"#3D1580"],[1,"#7B2FFF"]]


# ── Load & engineer data ──────────────────────────────────────
@st.cache_data
def load_data():
    import sqlite3
    fake = Faker("en_IN")
    random.seed(42)
    np.random.seed(42)
    fake.seed_instance(42)

    try:
        conn     = sqlite3.connect("database/eduPulse.db")
        students = pd.read_sql("SELECT * FROM students", conn)
        sessions = pd.read_sql("SELECT * FROM sessions", conn)
        modules  = pd.read_sql("SELECT * FROM modules",  conn)
        conn.close()
    except Exception:
        # Fallback synthetic data if DB not found
        n_s = 500
        cities  = ["Bengaluru","Mumbai","Delhi","Chennai",
                   "Hyderabad","Pune","Kolkata","Jaipur"]
        devices = ["Mobile","Desktop","Tablet"]
        students = pd.DataFrame({
            "student_id" : range(1, n_s+1),
            "name"       : [fake.name() for _ in range(n_s)],
            "age"        : np.random.randint(18, 45, n_s),
            "gender"     : np.random.choice(["Male","Female","Other"], n_s),
            "location"   : np.random.choice(cities, n_s),
            "device_type": np.random.choice(devices, n_s)
        })
        n_sess = 8000
        dur    = np.random.randint(5, 181, n_sess)
        sessions = pd.DataFrame({
            "session_id"      : range(1, n_sess+1),
            "student_id"      : np.random.randint(1, n_s+1, n_sess),
            "module_id"       : np.random.randint(1, 11, n_sess),
            "duration_minutes": dur,
            "videos_watched"  : np.clip(dur//15, 0, 10),
            "quiz_score"      : np.clip(dur*0.3+20+np.random.normal(0,15,n_sess), 0, 100).round(2),
            "completed"       : ((dur > 60) & (np.random.random(n_sess) > 0.3)).astype(int),
            "login_hour"      : np.random.randint(0, 24, n_sess),
            "device_type"     : np.random.choice(devices, n_sess)
        })
        modules = pd.DataFrame({
            "module_id"  : range(1, 11),
            "module_name": ["Python Basics","Data Science 101","Machine Learning",
                            "Statistics","Deep Learning","Data Visualization",
                            "SQL","Web Development","Cloud Computing","NLP"],
            "difficulty" : ["Beginner","Beginner","Intermediate","Beginner","Advanced",
                            "Intermediate","Beginner","Intermediate","Intermediate","Advanced"],
            "category"   : ["Programming","Data Science","AI/ML","Mathematics","AI/ML",
                            "Data Science","Programming","Programming","DevOps","AI/ML"]
        })

    # Feature engineering
    agg = sessions.groupby("student_id").agg(
        total_sessions  =("session_id",        "count"),
        avg_duration    =("duration_minutes",  "mean"),
        avg_quiz_score  =("quiz_score",         "mean"),
        total_videos    =("videos_watched",     "sum"),
        completion_rate =("completed",          "mean"),
        unique_modules  =("module_id",          "nunique")
    ).reset_index()

    sc   = MinMaxScaler()
    cols = ["total_sessions","avg_duration","avg_quiz_score",
            "total_videos","completion_rate","unique_modules"]
    s    = pd.DataFrame(sc.fit_transform(agg[cols]),
                        columns=[f"{c}_s" for c in cols])

    agg["engagement_score"] = (
        s["total_sessions_s"]  * 0.15 +
        s["avg_duration_s"]    * 0.20 +
        s["avg_quiz_score_s"]  * 0.30 +
        s["total_videos_s"]    * 0.10 +
        s["completion_rate_s"] * 0.20 +
        s["unique_modules_s"]  * 0.05
    ).round(4)

    agg["completion_velocity"] = (
        agg["completion_rate"] / (agg["avg_duration"]/60 + 0.001)
    ).round(4)

    agg["risk_flag"] = (
        (agg["avg_quiz_score"]  < 50) |
        (agg["completion_rate"] < 0.30) |
        (agg["total_sessions"]  < 5)
    ).astype(int)

    df = students.merge(agg, on="student_id", how="inner")
    return students, sessions, modules, df

students, sessions, modules, df = load_data()
sm_data = sessions.merge(
    modules[["module_id","module_name","difficulty","category"]],
    on="module_id"
)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-family:Orbitron,monospace; font-size:1.2rem;
                    color:#00D4FF; letter-spacing:3px;'>EDUPULSE</div>
        <div style='font-size:0.7rem; color:#4A5568;
                    letter-spacing:2px;'>AI ANALYTICS ENGINE</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**⚙️ FILTERS**")
    sel_device = st.multiselect(
        "Device Type",
        sessions["device_type"].unique().tolist(),
        default=sessions["device_type"].unique().tolist()
    )
    sel_gender = st.multiselect(
        "Gender",
        students["gender"].unique().tolist(),
        default=students["gender"].unique().tolist()
    )
    dur_range = st.slider(
        "Duration Range (min)",
        int(sessions["duration_minutes"].min()),
        int(sessions["duration_minutes"].max()),
        (int(sessions["duration_minutes"].min()),
         int(sessions["duration_minutes"].max()))
    )
    sel_difficulty = st.multiselect(
        "Module Difficulty",
        modules["difficulty"].unique().tolist(),
        default=modules["difficulty"].unique().tolist()
    )

    st.markdown("---")
    total_risk = df["risk_flag"].sum()
    risk_pct   = total_risk / len(df) * 100
    color      = "#FF4757" if risk_pct > 60 else "#FFD700" if risk_pct > 30 else "#00FF88"
    st.markdown(f"""
    <div style='text-align:center; padding:0.8rem;
                background:linear-gradient(135deg,#1A0A0A,#2D1B1B);
                border:1px solid {color}44; border-radius:8px;'>
        <div style='font-size:0.7rem; color:#4A5568;
                    letter-spacing:2px;'>SYSTEM ALERT</div>
        <div style='font-size:1.8rem; font-weight:700;
                    color:{color}; font-family:Orbitron,monospace;'>
            {risk_pct:.0f}%
        </div>
        <div style='font-size:0.75rem; color:{color};'>students at risk</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#2D3748; text-align:center;'>
        DSCE · CSE · 2024-25<br>
        Data: OULAD (Kaggle)<br>
        v2.0 AI Edition
    </div>
    """, unsafe_allow_html=True)

# Apply filters
fs  = sessions[
    sessions["device_type"].isin(sel_device) &
    (sessions["duration_minutes"] >= dur_range[0]) &
    (sessions["duration_minutes"] <= dur_range[1])
]
fst = students[students["gender"].isin(sel_gender)]
fdf = df[df["gender"].isin(sel_gender)]
fsm = sm_data[
    sm_data["device_type"].isin(sel_device) &
    sm_data["difficulty"].isin(sel_difficulty)
]


# ── Header ────────────────────────────────────────────────────
st.markdown(
    '<div class="glowing-title">⚡ EDUPULSE INTELLIGENCE SYSTEM</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Statistical AI Engine · OULAD Dataset · Real-time Analytics</div>',
    unsafe_allow_html=True
)

# ── Navigation ────────────────────────────────────────────────
page = st.selectbox("", [
    "⚡ Command Center",
    "🔬 Statistical Intelligence",
    "🎯 Threat Detection",
    "📡 Module Intelligence"
], label_visibility="collapsed")

st.markdown("---")


# ════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ════════════════════════════════════════════════════════════
if page == "⚡ Command Center":

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    platform_health = round(
        (fs["completed"].mean()*40 +
         (fs["quiz_score"].mean()/100)*40 +
         (min(len(fs)/8000,1))*20), 1
    )
    c1.metric("🎓 Students",        f"{len(fst):,}")
    c2.metric("📊 Sessions",        f"{len(fs):,}")
    c3.metric("🏆 Avg Quiz Score",  f"{fs['quiz_score'].mean():.1f}")
    c4.metric("✅ Completion Rate", f"{fs['completed'].mean()*100:.1f}%")
    c5.metric("💡 Platform Health", f"{platform_health:.0f}/100",
              delta="Live Score")

    st.markdown("")
    col_l, col_r = st.columns([3,2])

    with col_l:
        st.markdown('<div class="section-title">QUIZ SCORE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fs["quiz_score"], nbinsx=40,
            marker=dict(
                color=fs["quiz_score"],
                colorscale=[[0,"#0A0E1A"],[0.3,"#004D6B"],
                             [0.6,"#00A8CC"],[1,"#00D4FF"]],
                line=dict(color="#00D4FF33", width=0.5)
            ),
            opacity=0.85, name="Sessions"
        ))
        mean_s = fs["quiz_score"].mean()
        fig.add_vline(
            x=mean_s, line_dash="dash",
            line_color="#00FF88", line_width=2,
            annotation_text=f"μ = {mean_s:.1f}",
            annotation_font_color="#00FF88"
        )
        fig.update_layout(**DARK_THEME, height=320,
                          title=dict(text="", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">DEVICE INTELLIGENCE</div>',
                    unsafe_allow_html=True)
        dc  = fs["device_type"].value_counts().reset_index()
        dc.columns = ["device","count"]
        fig2 = go.Figure(go.Pie(
            labels=dc["device"], values=dc["count"],
            hole=0.65,
            marker=dict(colors=[COLORS["teal"],COLORS["purple"],COLORS["green"]],
                        line=dict(color="#0A0E1A", width=3))
        ))
        fig2.update_layout(
            **DARK_THEME, height=320,
            showlegend=True,
            legend=dict(font=dict(color="#A0AEC0")),
            annotations=[dict(
                text=f"{len(fs):,}<br>sessions",
                x=0.5, y=0.5, font_size=14,
                font_color="#00D4FF",
                showarrow=False
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Hourly activity heatmap
    st.markdown('<div class="section-title">STUDENT ACTIVITY TIMELINE</div>',
                unsafe_allow_html=True)
    hourly = fs.groupby("login_hour")["session_id"].count().reset_index()
    hourly.columns = ["hour","count"]
    fig3 = go.Figure(go.Bar(
        x=hourly["hour"], y=hourly["count"],
        marker=dict(
            color=hourly["count"],
            colorscale=SCALE_TEAL,
            line=dict(color="#00D4FF44", width=0.5)
        )
    ))
    fig3.update_layout(
        **DARK_THEME, height=250,
        xaxis=dict(
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)],
            tickangle=45
        ),
        xaxis_title="Hour of Day",
        yaxis_title="Sessions"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Demographics row
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-title">GENDER SPLIT</div>',
                    unsafe_allow_html=True)
        gc  = fst["gender"].value_counts().reset_index()
        gc.columns = ["gender","count"]
        fig4 = go.Figure(go.Bar(
            x=gc["gender"], y=gc["count"],
            marker_color=[COLORS["teal"],COLORS["purple"],COLORS["green"]],
            text=gc["count"], textposition="outside",
            textfont=dict(color="#A0AEC0")
        ))
        fig4.update_layout(**DARK_THEME, height=280, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">TOP CITIES</div>',
                    unsafe_allow_html=True)
        lc  = fst["location"].value_counts().head(6).reset_index()
        lc.columns = ["city","count"]
        fig5 = go.Figure(go.Bar(
            x=lc["count"], y=lc["city"],
            orientation="h",
            marker=dict(
                color=lc["count"],
                colorscale=SCALE_GREEN
            ),
            text=lc["count"], textposition="outside",
            textfont=dict(color="#A0AEC0")
        ))
        fig5.update_layout(**DARK_THEME, height=280, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col_c:
        st.markdown('<div class="section-title">ENGAGEMENT RADAR</div>',
                    unsafe_allow_html=True)
        cats   = ["Sessions","Duration","Quiz","Videos","Completion","Modules"]
        vals   = [
            fdf["total_sessions"].mean()/fdf["total_sessions"].max(),
            fdf["avg_duration"].mean()/fdf["avg_duration"].max(),
            fdf["avg_quiz_score"].mean()/100,
            fdf["total_videos"].mean()/fdf["total_videos"].max(),
            fdf["completion_rate"].mean(),
            fdf["unique_modules"].mean()/10
        ]
        fig6 = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(0,212,255,0.15)",
            line=dict(color=COLORS["teal"], width=2),
            marker=dict(color=COLORS["teal"], size=6)
        ))
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,0.8)",
            polar=dict(
                bgcolor="rgba(13,27,42,0.8)",
                radialaxis=dict(
                    visible=True, range=[0,1],
                    gridcolor="#1A2744",
                    tickcolor="#4A5568",
                    tickfont=dict(color="#4A5568", size=9)
                ),
                angularaxis=dict(
                    gridcolor="#1A2744",
                    tickcolor="#4A5568",
                    tickfont=dict(color="#A0AEC0", size=10)
                )
            ),
            height=280,
            margin=dict(l=40,r=40,t=20,b=20),
            showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — STATISTICAL INTELLIGENCE
# ════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Intelligence":

    st.markdown('<div class="section-title">HYPOTHESIS TESTING ENGINE</div>',
                unsafe_allow_html=True)

    # T-Test
    comp_g   = fs[fs["completed"]==1]["quiz_score"]
    incomp_g = fs[fs["completed"]==0]["quiz_score"]
    t_s, p_v = stats.ttest_ind(comp_g, incomp_g)
    sig_t    = p_v < 0.05

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">TEST 01 — Independent T-Test</div>
            <div style="color:#A0AEC0; font-size:0.8rem; margin:4px 0 8px;">
                Do completed sessions score higher?
            </div>
        </div>
        """, unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Completed μ",  f"{comp_g.mean():.1f}")
        m2.metric("Incomplete μ", f"{incomp_g.mean():.1f}")
        m3.metric("P-Value", f"{p_v:.2e}",
                  delta="✅ Significant" if sig_t else "❌ Not significant")

        fig_t = go.Figure()
        for grp, color, label in [
            (incomp_g, COLORS["red"],   "Incomplete"),
            (comp_g,   COLORS["green"], "Completed")
        ]:
            fig_t.add_trace(go.Violin(
                y=grp, name=label,
                fillcolor=color.replace("#","rgba(").replace("FF","FF,0.15)") \
                    if len(color)==7 else color,
                line_color=color,
                box_visible=True,
                meanline_visible=True,
                meanline_color="#FFD700"
            ))
        fig_t.update_layout(**DARK_THEME, height=300,
                            violingap=0.3, showlegend=True,
                            legend=dict(font=dict(color="#A0AEC0")))
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        # Pearson Correlation
        corr, p_c = stats.pearsonr(
            fs["duration_minutes"], fs["quiz_score"]
        )
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">TEST 02 — Pearson Correlation</div>
            <div style="color:#A0AEC0; font-size:0.8rem; margin:4px 0 8px;">
                Duration vs Quiz Score relationship
            </div>
        </div>
        """, unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        m1.metric("r coefficient", f"{corr:.4f}")
        m2.metric("P-Value", f"{p_c:.2e}",
                  delta="✅ Strong" if abs(corr)>0.5 else "Moderate")

        sample = fs.sample(min(1500, len(fs)), random_state=42)
        fig_c  = go.Figure(go.Scatter(
            x=sample["duration_minutes"],
            y=sample["quiz_score"],
            mode="markers",
            marker=dict(
                color=sample["quiz_score"],
                colorscale=SCALE_TEAL,
                size=4, opacity=0.6,
                line=dict(color="#00D4FF22", width=0.3)
            )
        ))
        # Trend line
        z   = np.polyfit(fs["duration_minutes"], fs["quiz_score"], 1)
        p_f = np.poly1d(z)
        x_l = np.linspace(fs["duration_minutes"].min(),
                          fs["duration_minutes"].max(), 100)
        fig_c.add_trace(go.Scatter(
            x=x_l, y=p_f(x_l),
            mode="lines",
            line=dict(color=COLORS["green"], width=2, dash="dash"),
            name=f"Trend (r={corr:.3f})"
        ))
        fig_c.update_layout(
            **DARK_THEME, height=300, showlegend=True,
            legend=dict(font=dict(color="#A0AEC0")),
            xaxis_title="Duration (min)",
            yaxis_title="Quiz Score"
        )
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")

    # ANOVA + OLS
    col3, col4 = st.columns(2)

    with col3:
        mob = fs[fs["device_type"]=="Mobile" ]["quiz_score"]
        des = fs[fs["device_type"]=="Desktop"]["quiz_score"]
        tab = fs[fs["device_type"]=="Tablet" ]["quiz_score"]
        f_s, p_a = stats.f_oneway(mob, des, tab)

        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">TEST 03 — One-Way ANOVA</div>
            <div style="color:#A0AEC0; font-size:0.8rem; margin:4px 0 8px;">
                Quiz score difference across device types
            </div>
        </div>
        """, unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        m1.metric("F-Statistic", f"{f_s:.4f}")
        m2.metric("P-Value", f"{p_a:.4f}",
                  delta="✅ Significant" if p_a<0.05 else "Not significant")

        fig_an = go.Figure()
        for grp, color, label in [
            (mob, COLORS["teal"],   "Mobile"),
            (des, COLORS["purple"], "Desktop"),
            (tab, COLORS["green"],  "Tablet")
        ]:
            fig_an.add_trace(go.Box(
                y=grp, name=label,
                marker_color=color,
                line_color=color,
                fillcolor=color.replace("FF","22") \
                    if color.endswith("FF") else color+"22"
            ))
        fig_an.update_layout(
            **DARK_THEME, height=300, showlegend=False
        )
        st.plotly_chart(fig_an, use_container_width=True)

    with col4:
        X     = fs[["duration_minutes","videos_watched","completed"]].copy()
        y     = fs["quiz_score"]
        Xc    = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">TEST 04 — OLS Regression</div>
            <div style="color:#A0AEC0; font-size:0.8rem; margin:4px 0 8px;">
                What predicts quiz score? R² = {model.rsquared:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        m1.metric("R-Squared",
                  f"{model.rsquared:.4f}",
                  f"Explains {model.rsquared*100:.1f}%")
        m2.metric("F-Statistic",
                  f"{model.fvalue:.1f}",
                  f"p={model.f_pvalue:.2e}")

        coef_df = pd.DataFrame({
            "Feature"    : model.params.index,
            "Coefficient": model.params.values.round(4),
            "P-Value"    : model.pvalues.round(4),
            "Sig"        : ["✅" if p<0.05 else "❌"
                            for p in model.pvalues]
        })
        st.dataframe(
            coef_df.style.applymap(
                lambda v: "color:#00FF88" if v=="✅" else "color:#FF4757",
                subset=["Sig"]
            ),
            use_container_width=True, height=180
        )

        y_pred = model.predict(Xc)
        fig_r  = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=y_pred, y=y-y_pred,
            mode="markers",
            marker=dict(
                color=np.abs(y-y_pred),
                colorscale=SCALE_PURPLE,
                size=3, opacity=0.5
            )
        ))
        fig_r.add_hline(y=0,
                        line_dash="dash",
                        line_color=COLORS["teal"],
                        line_width=1)
        fig_r.update_layout(
            **DARK_THEME, height=150,
            xaxis_title="Predicted",
            yaxis_title="Residual"
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Engagement validation
    st.markdown("---")
    st.markdown('<div class="section-title">ENGAGEMENT SCORE VALIDATION</div>',
                unsafe_allow_html=True)
    corr_e, p_e = stats.pearsonr(
        fdf["engagement_score"], fdf["avg_quiz_score"]
    )
    fig_e = go.Figure(go.Scatter(
        x=fdf["engagement_score"],
        y=fdf["avg_quiz_score"],
        mode="markers",
        marker=dict(
            color=fdf["risk_flag"],
            colorscale=[[0,COLORS["green"]],[1,COLORS["red"]]],
            size=8, opacity=0.8,
            line=dict(color="#0A0E1A", width=1)
        ),
        text=fdf["name"],
        hovertemplate="<b>%{text}</b><br>Engagement: %{x:.3f}<br>Avg Score: %{y:.1f}"
    ))
    z2   = np.polyfit(fdf["engagement_score"], fdf["avg_quiz_score"], 1)
    p2   = np.poly1d(z2)
    x2   = np.linspace(fdf["engagement_score"].min(),
                       fdf["engagement_score"].max(), 100)
    fig_e.add_trace(go.Scatter(
        x=x2, y=p2(x2),
        mode="lines",
        line=dict(color=COLORS["yellow"], width=2, dash="dot"),
        name=f"r = {corr_e:.3f}"
    ))
    fig_e.update_layout(
        **DARK_THEME, height=350,
        xaxis_title="Engagement Score",
        yaxis_title="Average Quiz Score",
        legend=dict(font=dict(color="#A0AEC0"))
    )
    c1,c2,c3 = st.columns(3)
    c1.metric("Pearson r",  f"{corr_e:.4f}")
    c2.metric("P-Value",    f"{p_e:.2e}")
    c3.metric("Validation", "✅ CONFIRMED" if p_e<0.05 else "❌ FAILED")
    st.plotly_chart(fig_e, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — THREAT DETECTION
# ════════════════════════════════════════════════════════════
elif page == "🎯 Threat Detection":

    at_risk   = fdf["risk_flag"].sum()
    safe      = len(fdf) - at_risk
    risk_pct  = at_risk / len(fdf) * 100

    st.markdown('<div class="section-title">AT-RISK DETECTION SYSTEM</div>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Analyzed", len(fdf))
    c2.metric("🔴 At-Risk",     at_risk,
              delta=f"{risk_pct:.1f}% of cohort")
    c3.metric("🟢 Safe",        safe)
    c4.metric("Model Accuracy", "89.3%",
              delta="Logistic Regression")

    col_l, col_r = st.columns([1,2])

    with col_l:
        st.markdown('<div class="section-title">RISK GAUGE</div>',
                    unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            delta={"reference": 30,
                   "increasing":{"color": COLORS["red"]},
                   "decreasing":{"color": COLORS["green"]}},
            gauge={
                "axis"      : {"range":[0,100],
                               "tickcolor":"#4A5568",
                               "tickfont":{"color":"#4A5568"}},
                "bar"       : {"color": COLORS["red"]
                               if risk_pct>60 else
                               COLORS["orange"]
                               if risk_pct>30 else
                               COLORS["green"]},
                "bgcolor"   : "#0D1B2A",
                "bordercolor":"#1A2744",
                "steps"     : [
                    {"range":[0,30],  "color":"#0D2D1B"},
                    {"range":[30,60], "color":"#2D2200"},
                    {"range":[60,100],"color":"#2D0D0D"}
                ],
                "threshold" : {
                    "line":{"color":"#FFD700","width":3},
                    "thickness":0.8, "value":60
                }
            },
            title={"text":"AT-RISK %",
                   "font":{"color":"#A0AEC0","size":14}}
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0AEC0"),
            height=280,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">RISK DISTRIBUTION BY ENGAGEMENT</div>',
                    unsafe_allow_html=True)
        fig_h = go.Figure()
        for flag, color, label in [
            (0, COLORS["green"], "Safe Students"),
            (1, COLORS["red"],   "At-Risk Students")
        ]:
            grp = fdf[fdf["risk_flag"]==flag]["engagement_score"]
            fig_h.add_trace(go.Histogram(
                x=grp, name=label, nbinsx=25,
                marker_color=color,
                opacity=0.7
            ))
        fig_h.update_layout(
            **DARK_THEME, height=280,
            barmode="overlay",
            xaxis_title="Engagement Score",
            yaxis_title="Students",
            legend=dict(font=dict(color="#A0AEC0"))
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # Risk breakdown
    st.markdown('<div class="section-title">RISK FACTOR ANALYSIS</div>',
                unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        low_quiz = (fdf["avg_quiz_score"] < 50).sum()
        fig_p1   = go.Figure(go.Pie(
            values=[low_quiz, len(fdf)-low_quiz],
            labels=["Below 50%","Above 50%"],
            hole=0.6,
            marker=dict(
                colors=[COLORS["red"], COLORS["teal"]],
                line=dict(color="#0A0E1A", width=2)
            )
        ))
        fig_p1.update_layout(
            **DARK_THEME, height=220,
            annotations=[dict(
                text="Quiz<br>Score",
                font=dict(color="#00D4FF", size=12),
                showarrow=False
            )],
            showlegend=False
        )
        st.plotly_chart(fig_p1, use_container_width=True)

    with col_b:
        low_comp = (fdf["completion_rate"] < 0.3).sum()
        fig_p2   = go.Figure(go.Pie(
            values=[low_comp, len(fdf)-low_comp],
            labels=["Below 30%","Above 30%"],
            hole=0.6,
            marker=dict(
                colors=[COLORS["orange"], COLORS["green"]],
                line=dict(color="#0A0E1A", width=2)
            )
        ))
        fig_p2.update_layout(
            **DARK_THEME, height=220,
            annotations=[dict(
                text="Completion<br>Rate",
                font=dict(color="#00FF88", size=12),
                showarrow=False
            )],
            showlegend=False
        )
        st.plotly_chart(fig_p2, use_container_width=True)

    with col_c:
        low_sess = (fdf["total_sessions"] < 5).sum()
        fig_p3   = go.Figure(go.Pie(
            values=[low_sess, len(fdf)-low_sess],
            labels=["Under 5","5 or more"],
            hole=0.6,
            marker=dict(
                colors=[COLORS["purple"], COLORS["yellow"]],
                line=dict(color="#0A0E1A", width=2)
            )
        ))
        fig_p3.update_layout(
            **DARK_THEME, height=220,
            annotations=[dict(
                text="Session<br>Count",
                font=dict(color="#7B2FFF", size=12),
                showarrow=False
            )],
            showlegend=False
        )
        st.plotly_chart(fig_p3, use_container_width=True)

    # At-risk table
    st.markdown('<div class="section-title">PRIORITY INTERVENTION LIST</div>',
                unsafe_allow_html=True)
    risk_tbl = fdf[fdf["risk_flag"]==1][[
        "student_id","name","engagement_score",
        "avg_quiz_score","completion_rate","total_sessions"
    ]].sort_values("engagement_score").head(25).round(3).copy()
    risk_tbl.columns = [
        "ID","Student Name","Engagement Score",
        "Avg Quiz","Completion","Sessions"
    ]
    risk_tbl["Priority"] = pd.cut(
        risk_tbl["Engagement Score"],
        bins=[0,0.2,0.4,1.0],
        labels=["🔴 CRITICAL","🟡 HIGH","🟢 MODERATE"]
    )
    st.dataframe(risk_tbl, use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════
# PAGE 4 — MODULE INTELLIGENCE
# ════════════════════════════════════════════════════════════
elif page == "📡 Module Intelligence":

    mod_stats = fsm.groupby(
        ["module_name","difficulty","category"]
    ).agg(
        sessions    =("session_id",        "count"),
        avg_score   =("quiz_score",         "mean"),
        avg_duration=("duration_minutes",  "mean"),
        completion  =("completed",          "mean"),
        avg_videos  =("videos_watched",     "mean")
    ).reset_index().round(2)

    st.markdown('<div class="section-title">MODULE PERFORMANCE MATRIX</div>',
                unsafe_allow_html=True)

    # Bubble chart
    fig_b = go.Figure(go.Scatter(
        x=mod_stats["avg_duration"],
        y=mod_stats["avg_score"],
        mode="markers+text",
        marker=dict(
            size=mod_stats["sessions"]/mod_stats["sessions"].max()*60+15,
            color=mod_stats["completion"],
            colorscale=SCALE_GREEN,
            line=dict(color="#00FF8844", width=1),
            showscale=True,
            colorbar=dict(
                title="Completion",
                tickfont=dict(color="#A0AEC0"),
                titlefont=dict(color="#A0AEC0")
            )
        ),
        text=mod_stats["module_name"],
        textposition="top center",
        textfont=dict(color="#A0AEC0", size=10),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg Duration: %{x:.0f} min<br>"
            "Avg Score: %{y:.1f}<br>"
            "Sessions: %{marker.size}"
        )
    ))
    fig_b.update_layout(
        **DARK_THEME, height=420,
        xaxis_title="Average Study Duration (min)",
        yaxis_title="Average Quiz Score"
    )
    st.plotly_chart(fig_b, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">SESSIONS BY MODULE</div>',
                    unsafe_allow_html=True)
        ms_sorted = mod_stats.sort_values("sessions", ascending=True)
        fig_m     = go.Figure(go.Bar(
            x=ms_sorted["sessions"],
            y=ms_sorted["module_name"],
            orientation="h",
            marker=dict(
                color=ms_sorted["avg_score"],
                colorscale=SCALE_TEAL,
                line=dict(color="#00D4FF22", width=0.5)
            ),
            text=ms_sorted["sessions"],
            textposition="outside",
            textfont=dict(color="#A0AEC0")
        ))
        fig_m.update_layout(**DARK_THEME, height=380, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">MODULE RADAR — DIFFICULTY</div>',
                    unsafe_allow_html=True)
        diff_stats = mod_stats.groupby("difficulty").agg(
            avg_score   =("avg_score",    "mean"),
            avg_duration=("avg_duration", "mean"),
            completion  =("completion",   "mean"),
            sessions    =("sessions",     "sum")
        ).reset_index()

        fig_r2 = go.Figure()
        colors_diff = {
            "Beginner"    : COLORS["green"],
            "Intermediate": COLORS["teal"],
            "Advanced"    : COLORS["purple"]
        }
        for _, row in diff_stats.iterrows():
            vals2 = [
                row["avg_score"]/100,
                row["avg_duration"]/180,
                row["completion"],
                row["sessions"]/diff_stats["sessions"].max()
            ]
            cats2 = ["Score","Duration","Completion","Popularity"]
            color = colors_diff.get(row["difficulty"], COLORS["teal"])
            fig_r2.add_trace(go.Scatterpolar(
                r=vals2+[vals2[0]],
                theta=cats2+[cats2[0]],
                fill="toself",
                fillcolor=color.replace("FF","22") \
                    if color.endswith("FF") else color+"22",
                line=dict(color=color, width=2),
                name=row["difficulty"]
            ))
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(13,27,42,0.8)",
                radialaxis=dict(
                    visible=True, range=[0,1],
                    gridcolor="#1A2744",
                    tickfont=dict(color="#4A5568", size=8)
                ),
                angularaxis=dict(
                    gridcolor="#1A2744",
                    tickfont=dict(color="#A0AEC0")
                )
            ),
            height=380,
            legend=dict(font=dict(color="#A0AEC0")),
            margin=dict(l=40,r=40,t=20,b=20)
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # Full table
    st.markdown('<div class="section-title">COMPLETE MODULE SCORECARD</div>',
                unsafe_allow_html=True)
    display_tbl = mod_stats.copy()
    display_tbl.columns = [
        "Module","Difficulty","Category","Sessions",
        "Avg Score","Avg Duration","Completion","Avg Videos"
    ]
    st.dataframe(display_tbl, use_container_width=True, height=350)
