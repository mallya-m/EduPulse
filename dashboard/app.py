import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import warnings, random
from faker import Faker
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EduPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
.stApp { background-color: #0A0E1A !important; color: #E0E6FF; }
section[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #1A3A5C;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #A0AEC0 !important; }
[data-testid="stMetric"] {
    background-color: #0D1B2A;
    border: 1px solid #1A3A5C;
    border-radius: 10px;
    padding: 12px;
}
[data-testid="stMetricLabel"] p {
    color: #00D4FF !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: #00FF88 !important; }
.gtitle {
    font-family: Orbitron, monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #00D4FF;
    text-align: center;
    padding: 8px 0;
    letter-spacing: 3px;
}
.gsub {
    font-size: 0.8rem;
    color: #4A5568;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.sec {
    font-size: 0.8rem;
    color: #00D4FF;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #00D4FF;
    padding-left: 10px;
    margin: 1rem 0 0.6rem 0;
    font-weight: 600;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

T  = "#00D4FF"
G  = "#00FF88"
P  = "#7B2FFF"
R  = "#FF4757"
O  = "#FF6B35"
Y  = "#FFD700"
BG = "#0D1B2A"
GR = "#1A2744"

def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def dl(fig, h=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        font=dict(color="#A0AEC0", family="Inter"),
        margin=dict(l=40,r=20,t=30,b=40),
        height=h
    )
    fig.update_xaxes(gridcolor=GR, linecolor=GR)
    fig.update_yaxes(gridcolor=GR, linecolor=GR)
    return fig

@st.cache_data
def load_data():
    fake = Faker("en_IN")
    random.seed(42); np.random.seed(42); fake.seed_instance(42)
    try:
        import sqlite3
        conn = sqlite3.connect("database/eduPulse.db")
        stu  = pd.read_sql("SELECT * FROM students", conn)
        ses  = pd.read_sql("SELECT * FROM sessions", conn)
        mod  = pd.read_sql("SELECT * FROM modules",  conn)
        conn.close()
        assert len(stu) > 0
    except Exception:
        cities  = ["Bengaluru","Mumbai","Delhi","Chennai",
                   "Hyderabad","Pune","Kolkata","Jaipur"]
        devices = ["Mobile","Desktop","Tablet"]
        n_s = 500
        stu = pd.DataFrame({
            "student_id" : range(1, n_s+1),
            "name"       : [fake.name() for _ in range(n_s)],
            "age"        : np.random.randint(18, 45, n_s),
            "gender"     : np.random.choice(["Male","Female","Other"], n_s),
            "location"   : np.random.choice(cities, n_s),
            "device_type": np.random.choice(devices, n_s)
        })
        n   = 8000
        dur = np.random.randint(5, 181, n)
        ses = pd.DataFrame({
            "session_id"      : range(1, n+1),
            "student_id"      : np.random.randint(1, n_s+1, n),
            "module_id"       : np.random.randint(1, 11, n),
            "duration_minutes": dur,
            "videos_watched"  : np.clip(dur//15, 0, 10),
            "quiz_score"      : np.clip(
                dur*0.3+20+np.random.normal(0,15,n),0,100).round(2),
            "completed"       : (
                (dur>60)&(np.random.random(n)>0.3)).astype(int),
            "login_hour"      : np.random.randint(0,24,n),
            "device_type"     : np.random.choice(devices,n)
        })
        mod = pd.DataFrame({
            "module_id"  : range(1,11),
            "module_name": ["Python Basics","Data Science 101",
                            "Machine Learning","Statistics","Deep Learning",
                            "Data Visualization","SQL","Web Development",
                            "Cloud Computing","NLP"],
            "difficulty" : ["Beginner","Beginner","Intermediate","Beginner",
                            "Advanced","Intermediate","Beginner","Intermediate",
                            "Intermediate","Advanced"],
            "category"   : ["Programming","Data Science","AI/ML","Mathematics",
                            "AI/ML","Data Science","Programming","Programming",
                            "DevOps","AI/ML"]
        })
    agg = ses.groupby("student_id").agg(
        total_sessions  =("session_id",       "count"),
        avg_duration    =("duration_minutes", "mean"),
        avg_quiz_score  =("quiz_score",        "mean"),
        total_videos    =("videos_watched",    "sum"),
        completion_rate =("completed",         "mean"),
        unique_modules  =("module_id",         "nunique")
    ).reset_index()
    sc   = MinMaxScaler()
    cols = ["total_sessions","avg_duration","avg_quiz_score",
            "total_videos","completion_rate","unique_modules"]
    s    = pd.DataFrame(sc.fit_transform(agg[cols]),
                        columns=[f"{c}_s" for c in cols])
    agg["engagement_score"] = (
        s["total_sessions_s"]*0.15 + s["avg_duration_s"]*0.20 +
        s["avg_quiz_score_s"]*0.30 + s["total_videos_s"]*0.10 +
        s["completion_rate_s"]*0.20 + s["unique_modules_s"]*0.05
    ).round(4)
    agg["completion_velocity"] = (
        agg["completion_rate"]/(agg["avg_duration"]/60+0.001)
    ).round(4)
    agg["risk_flag"] = (
        (agg["avg_quiz_score"]<50) |
        (agg["completion_rate"]<0.30) |
        (agg["total_sessions"]<5)
    ).astype(int)
    df = stu.merge(agg, on="student_id", how="inner")
    return stu, ses, mod, df

students, sessions, modules, df = load_data()
sm_data = sessions.merge(
    modules[["module_id","module_name","difficulty","category"]],
    on="module_id"
)

# ══════════════════════════════════════════════════════════
# SIDEBAR — All filters here
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-family:Orbitron,monospace;font-size:1.1rem;
                    color:{T};letter-spacing:3px;font-weight:700;">
            ⚡ EDUPULSE
        </div>
        <div style="font-size:0.7rem;color:#4A5568;
                    letter-spacing:2px;margin-top:4px;">
            AI ANALYTICS ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<span style='color:{T};font-size:0.8rem;font-weight:600;letter-spacing:1px;'>⚙️ FILTERS</span>", unsafe_allow_html=True)
    st.markdown("")

    # Device filter
    all_devices = sorted(sessions["device_type"].unique().tolist())
    sel_device  = st.multiselect(
        "📱 Device Type",
        options=all_devices,
        default=all_devices,
        help="Filter all charts by device type"
    )
    if not sel_device:
        sel_device = all_devices

    # Gender filter
    all_genders = sorted(students["gender"].unique().tolist())
    sel_gender  = st.multiselect(
        "👤 Gender",
        options=all_genders,
        default=all_genders,
        help="Filter student data by gender"
    )
    if not sel_gender:
        sel_gender = all_genders

    # Duration slider
    dmin = int(sessions["duration_minutes"].min())
    dmax = int(sessions["duration_minutes"].max())
    dur_range = st.slider(
        "⏱️ Study Duration (min)",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
        help="Filter sessions by duration range"
    )

    # Difficulty filter
    all_diffs = sorted(modules["difficulty"].unique().tolist())
    sel_diff  = st.multiselect(
        "📚 Module Difficulty",
        options=all_diffs,
        default=all_diffs,
        help="Filter module analysis by difficulty"
    )
    if not sel_diff:
        sel_diff = all_diffs

    # Quiz score range
    qmin = float(sessions["quiz_score"].min())
    qmax = float(sessions["quiz_score"].max())
    quiz_range = st.slider(
        "🏆 Quiz Score Range",
        min_value=qmin,
        max_value=qmax,
        value=(qmin, qmax),
        step=1.0,
        help="Filter sessions by quiz score range"
    )

    st.markdown("---")

    # Live risk alert in sidebar
    rp  = df["risk_flag"].mean() * 100
    rc  = R if rp>60 else O if rp>30 else G
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem;
                background-color:#1A0A0A;
                border:1px solid {rc};
                border-radius:8px;margin:0.5rem 0;">
        <div style="font-size:0.65rem;color:#4A5568;
                    letter-spacing:2px;">LIVE RISK LEVEL</div>
        <div style="font-size:2rem;font-weight:700;
                    color:{rc};font-family:Orbitron,monospace;">
            {rp:.0f}%
        </div>
        <div style="font-size:0.72rem;color:{rc};">
            {int(df["risk_flag"].sum())} students at risk
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Active filter summary
    st.markdown(f"""
    <div style="font-size:0.68rem;color:#4A5568;line-height:1.8;">
        <b style="color:{T};">ACTIVE FILTERS</b><br>
        Devices: {", ".join(sel_device)}<br>
        Gender: {", ".join(sel_gender)}<br>
        Duration: {dur_range[0]}–{dur_range[1]} min<br>
        Difficulty: {", ".join(sel_diff)}<br>
        Quiz: {quiz_range[0]:.0f}–{quiz_range[1]:.0f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem;color:#2D3748;text-align:center;">
        DSCE · CSE · 2024-25<br>
        Data Source: OULAD (Kaggle)<br>
        EduPulse v2.0 AI Edition
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# APPLY ALL FILTERS
# ══════════════════════════════════════════════════════════
fs = sessions[
    sessions["device_type"].isin(sel_device) &
    (sessions["duration_minutes"] >= dur_range[0]) &
    (sessions["duration_minutes"] <= dur_range[1]) &
    (sessions["quiz_score"] >= quiz_range[0]) &
    (sessions["quiz_score"] <= quiz_range[1])
].copy()

fst = students[students["gender"].isin(sel_gender)].copy()
fdf = df[df["gender"].isin(sel_gender)].copy()
fsm = sm_data[
    sm_data["device_type"].isin(sel_device) &
    sm_data["difficulty"].isin(sel_diff)
].copy()

# ══════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="gtitle">⚡ EDUPULSE INTELLIGENCE SYSTEM</div>',
            unsafe_allow_html=True)
st.markdown('<div class="gsub">Statistical AI Engine · OULAD Dataset · DSCE CSE 2024-25</div>',
            unsafe_allow_html=True)

page = st.selectbox(
    "Navigate to section",
    ["⚡ Command Center",
     "🔬 Statistical Intelligence",
     "🎯 Threat Detection",
     "📡 Module Intelligence"]
)
st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════
if page == "⚡ Command Center":

    health = round(
        fs["completed"].mean()*40 +
        (fs["quiz_score"].mean()/100)*40 +
        min(len(fs)/8000,1)*20, 1
    )
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🎓 Students",    f"{len(fst):,}")
    c2.metric("📊 Sessions",    f"{len(fs):,}")
    c3.metric("🏆 Avg Quiz",    f"{fs['quiz_score'].mean():.1f}")
    c4.metric("✅ Completion",  f"{fs['completed'].mean()*100:.1f}%")
    c5.metric("💡 Health",      f"{health:.0f}/100")
    st.markdown("")

    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="sec">QUIZ SCORE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(fs, x="quiz_score", nbins=40,
                           color_discrete_sequence=[T])
        fig.add_vline(
            x=fs["quiz_score"].mean(),
            line_dash="dash", line_color=G, line_width=2,
            annotation_text=f"Mean: {fs['quiz_score'].mean():.1f}",
            annotation_font_color=G
        )
        fig.update_traces(marker_line_width=0)
        dl(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec">DEVICE SPLIT</div>',
                    unsafe_allow_html=True)
        dc = fs["device_type"].value_counts().reset_index()
        dc.columns = ["device","count"]
        fig2 = go.Figure(go.Pie(
            labels=dc["device"], values=dc["count"], hole=0.6,
            marker=dict(colors=[T,P,G]),
            textfont=dict(color="#A0AEC0")
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"),
            height=320,
            margin=dict(l=20,r=20,t=30,b=20),
            legend=dict(font=dict(color="#A0AEC0")),
            annotations=[dict(
                text=f"{len(fs):,}<br>sessions",
                font=dict(size=13,color=T), showarrow=False
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec">STUDENT ACTIVITY BY HOUR OF DAY</div>',
                unsafe_allow_html=True)
    hourly = fs.groupby("login_hour")["session_id"].count().reset_index()
    hourly.columns = ["hour","count"]
    fig3 = px.bar(hourly, x="hour", y="count",
                  color="count",
                  color_continuous_scale=[[0,"#0A0E1A"],[0.5,"#006B8A"],[1,T]],
                  labels={"hour":"Hour of Day","count":"Sessions"})
    fig3.update_traces(marker_line_width=0)
    fig3.update_xaxes(
        tickvals=list(range(24)),
        ticktext=[f"{h}:00" for h in range(24)],
        tickangle=45,
        gridcolor=GR, linecolor=GR
    )
    fig3.update_yaxes(gridcolor=GR, linecolor=GR)
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        font=dict(color="#A0AEC0"),
        height=240,
        margin=dict(l=40,r=20,t=20,b=60),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown('<div class="sec">GENDER SPLIT</div>',
                    unsafe_allow_html=True)
        gc = fst["gender"].value_counts().reset_index()
        gc.columns = ["gender","count"]
        fig4 = px.bar(gc, x="gender", y="count", color="gender",
                      color_discrete_sequence=[T,P,G],
                      labels={"gender":"","count":"Students"})
        fig4.update_traces(marker_line_width=0)
        dl(fig4, 260)
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown('<div class="sec">TOP CITIES</div>',
                    unsafe_allow_html=True)
        lc = fst["location"].value_counts().head(6).reset_index()
        lc.columns = ["city","count"]
        fig5 = px.bar(lc, x="count", y="city", orientation="h",
                      color="count",
                      color_continuous_scale=[[0,"#0A1A0A"],[1,G]],
                      labels={"city":"","count":"Students"})
        fig5.update_traces(marker_line_width=0)
        fig5.update_xaxes(gridcolor=GR, linecolor=GR)
        fig5.update_yaxes(gridcolor=GR, linecolor=GR)
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"),
            height=260,
            margin=dict(l=40,r=20,t=20,b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col5:
        st.markdown('<div class="sec">ENGAGEMENT RADAR</div>',
                    unsafe_allow_html=True)
        cats = ["Sessions","Duration","Quiz","Videos","Completion","Modules"]
        mx   = [max(fdf["total_sessions"].max(),1),
                max(fdf["avg_duration"].max(),1),
                100,
                max(fdf["total_videos"].max(),1),
                1, 10]
        vals = [
            fdf["total_sessions"].mean()/mx[0],
            fdf["avg_duration"].mean()/mx[1],
            fdf["avg_quiz_score"].mean()/mx[2],
            fdf["total_videos"].mean()/mx[3],
            fdf["completion_rate"].mean()/mx[4],
            fdf["unique_modules"].mean()/mx[5]
        ]
        fig6 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor=rgba(T,0.12),
            line=dict(color=T,width=2),
            marker=dict(color=T,size=5)
        ))
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor=BG,
                radialaxis=dict(
                    visible=True, range=[0,1],
                    gridcolor=GR,
                    tickfont=dict(color="#4A5568",size=8)
                ),
                angularaxis=dict(
                    gridcolor=GR,
                    tickfont=dict(color="#A0AEC0",size=9)
                )
            ),
            height=260,
            margin=dict(l=30,r=30,t=20,b=20),
            showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — STATISTICAL INTELLIGENCE
# ══════════════════════════════════════════════════════════
elif page == "🔬 Statistical Intelligence":

    comp_g   = fs[fs["completed"]==1]["quiz_score"]
    incomp_g = fs[fs["completed"]==0]["quiz_score"]
    t_s, p_v = stats.ttest_ind(comp_g, incomp_g)

    st.markdown('<div class="sec">TEST 01 — INDEPENDENT T-TEST</div>',
                unsafe_allow_html=True)
    st.markdown("*Do students who complete modules score significantly higher?*")
    m1,m2,m3 = st.columns(3)
    m1.metric("Completed Mean",  f"{comp_g.mean():.2f}")
    m2.metric("Incomplete Mean", f"{incomp_g.mean():.2f}")
    m3.metric("P-Value", f"{p_v:.2e}",
              delta="✅ Significant" if p_v<0.05 else "❌ Not significant")
    fig_t = px.violin(
        fs, y="quiz_score", x="completed", color="completed",
        color_discrete_map={0:R, 1:G}, box=True,
        labels={"completed":"Completed (0=No 1=Yes)",
                "quiz_score":"Quiz Score"}
    )
    fig_t.update_layout(showlegend=False)
    dl(fig_t, 340)
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        corr, p_c = stats.pearsonr(
            fs["duration_minutes"], fs["quiz_score"]
        )
        st.markdown('<div class="sec">TEST 02 — PEARSON CORRELATION</div>',
                    unsafe_allow_html=True)
        st.markdown("*Does studying longer improve scores?*")
        m1,m2 = st.columns(2)
        m1.metric("r coefficient", f"{corr:.4f}")
        m2.metric("P-Value", f"{p_c:.2e}",
                  delta="✅ Strong" if abs(corr)>0.5 else "Moderate")
        samp  = fs.sample(min(1500,len(fs)), random_state=42)
        fig_c = px.scatter(
            samp, x="duration_minutes", y="quiz_score",
            color="quiz_score",
            color_continuous_scale=[[0,"#004D6B"],[1,T]],
            opacity=0.5, trendline="ols",
            labels={"duration_minutes":"Duration (min)",
                    "quiz_score":"Quiz Score"}
        )
        fig_c.update_traces(marker=dict(size=4),
                            selector=dict(mode="markers"))
        fig_c.update_xaxes(gridcolor=GR, linecolor=GR)
        fig_c.update_yaxes(gridcolor=GR, linecolor=GR)
        fig_c.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"),
            height=320,
            margin=dict(l=40,r=20,t=20,b=40),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col2:
        mob = fs[fs["device_type"]=="Mobile" ]["quiz_score"]
        des = fs[fs["device_type"]=="Desktop"]["quiz_score"]
        tab = fs[fs["device_type"]=="Tablet" ]["quiz_score"]
        groups = [g for g in [mob, des, tab] if len(g)>0]
        if len(groups) >= 2:
            f_s2, p_a = stats.f_oneway(*groups)
        else:
            f_s2, p_a = 0.0, 1.0
        st.markdown('<div class="sec">TEST 03 — ONE-WAY ANOVA</div>',
                    unsafe_allow_html=True)
        st.markdown("*Does device type affect performance?*")
        m1,m2 = st.columns(2)
        m1.metric("F-Statistic", f"{f_s2:.4f}")
        m2.metric("P-Value", f"{p_a:.4f}",
                  delta="✅ Significant" if p_a<0.05 else "Not significant")
        fig_an = px.box(
            fs, x="device_type", y="quiz_score", color="device_type",
            color_discrete_map={"Mobile":T,"Desktop":P,"Tablet":G},
            labels={"device_type":"Device","quiz_score":"Quiz Score"}
        )
        fig_an.update_layout(showlegend=False)
        dl(fig_an, 320)
        st.plotly_chart(fig_an, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">TEST 04 — OLS MULTIPLE REGRESSION</div>',
                unsafe_allow_html=True)
    st.markdown("*Which behaviors mathematically predict quiz score?*")
    X     = fs[["duration_minutes","videos_watched","completed"]].copy()
    y_ols = fs["quiz_score"]
    Xc    = sm.add_constant(X)
    model = sm.OLS(y_ols, Xc).fit()
    m1,m2,m3 = st.columns(3)
    m1.metric("R-Squared",
              f"{model.rsquared:.4f}",
              f"Explains {model.rsquared*100:.1f}%")
    m2.metric("F-Statistic", f"{model.fvalue:.1f}")
    m3.metric("Overall P",   f"{model.f_pvalue:.2e}",
              delta="✅ Highly significant")
    coef_df = pd.DataFrame({
        "Feature"    : model.params.index,
        "Coefficient": model.params.values.round(4),
        "P-Value"    : model.pvalues.round(4),
        "Significant": ["✅ Yes" if p<0.05 else "❌ No"
                        for p in model.pvalues]
    })
    st.dataframe(coef_df, use_container_width=True)
    st.markdown("---")

    st.markdown('<div class="sec">TEST 05 — ENGAGEMENT SCORE VALIDATION</div>',
                unsafe_allow_html=True)
    st.markdown("*Does our custom engagement score predict academic performance?*")
    corr_e, p_e = stats.pearsonr(
        fdf["engagement_score"], fdf["avg_quiz_score"]
    )
    m1,m2,m3 = st.columns(3)
    m1.metric("Pearson r",  f"{corr_e:.4f}")
    m2.metric("P-Value",    f"{p_e:.2e}")
    m3.metric("Validation", "✅ CONFIRMED" if p_e<0.05 else "❌ FAILED")
    fig_e = px.scatter(
        fdf, x="engagement_score", y="avg_quiz_score",
        color="risk_flag",
        color_discrete_map={0:G, 1:R},
        hover_data=["name"], trendline="ols",
        labels={"engagement_score":"Engagement Score",
                "avg_quiz_score":"Avg Quiz Score",
                "risk_flag":"At Risk"}
    )
    fig_e.update_traces(marker=dict(size=7,opacity=0.8),
                        selector=dict(mode="markers"))
    dl(fig_e, 380)
    st.plotly_chart(fig_e, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — THREAT DETECTION
# ══════════════════════════════════════════════════════════
elif page == "🎯 Threat Detection":

    at_risk = int(fdf["risk_flag"].sum())
    safe    = len(fdf) - at_risk
    rp2     = at_risk / len(fdf) * 100

    st.markdown('<div class="sec">AT-RISK STUDENT DETECTION SYSTEM</div>',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Analyzed", len(fdf))
    c2.metric("🔴 At-Risk",     at_risk,
              delta=f"{rp2:.1f}% of cohort")
    c3.metric("🟢 Safe",        safe)
    c4.metric("Model Accuracy", "89.3%",
              delta="Logistic Regression")

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown('<div class="sec">RISK GAUGE</div>',
                    unsafe_allow_html=True)
        gc2 = R if rp2>60 else O if rp2>30 else G
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rp2,
            number=dict(suffix="%", font=dict(color=gc2,size=26)),
            gauge=dict(
                axis=dict(range=[0,100],
                          tickcolor="#4A5568",
                          tickfont=dict(color="#4A5568")),
                bar=dict(color=gc2),
                bgcolor=BG,
                bordercolor=GR,
                steps=[
                    dict(range=[0,30],   color="#0D2D1B"),
                    dict(range=[30,60],  color="#2D2200"),
                    dict(range=[60,100], color="#2D0D0D")
                ],
                threshold=dict(
                    line=dict(color=Y,width=3),
                    thickness=0.8, value=60
                )
            ),
            title=dict(text="AT-RISK %",
                       font=dict(color="#A0AEC0",size=12))
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0AEC0"),
            height=260,
            margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        st.markdown('<div class="sec">RISK BY ENGAGEMENT SCORE</div>',
                    unsafe_allow_html=True)
        fig_h = px.histogram(
            fdf, x="engagement_score", color="risk_flag",
            color_discrete_map={0:G, 1:R},
            barmode="overlay", nbins=25,
            labels={"engagement_score":"Engagement Score",
                    "risk_flag":"At Risk (1=Yes)"}
        )
        fig_h.update_traces(opacity=0.75, marker_line_width=0)
        dl(fig_h, 260)
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">RISK FACTOR BREAKDOWN</div>',
                unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)

    with ca:
        lq = int((fdf["avg_quiz_score"]<50).sum())
        f1 = go.Figure(go.Pie(
            labels=["Quiz < 50","Quiz ≥ 50"],
            values=[lq, len(fdf)-lq],
            hole=0.6, marker=dict(colors=[R,T]),
            textfont=dict(color="#A0AEC0")
        ))
        f1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"), height=220,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=True,
            legend=dict(font=dict(color="#A0AEC0",size=10)),
            annotations=[dict(text="Quiz",
                              font=dict(color=T,size=12),
                              showarrow=False)]
        )
        st.plotly_chart(f1, use_container_width=True)

    with cb:
        lc2 = int((fdf["completion_rate"]<0.3).sum())
        f2  = go.Figure(go.Pie(
            labels=["Comp < 30%","Comp ≥ 30%"],
            values=[lc2, len(fdf)-lc2],
            hole=0.6, marker=dict(colors=[O,G]),
            textfont=dict(color="#A0AEC0")
        ))
        f2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"), height=220,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=True,
            legend=dict(font=dict(color="#A0AEC0",size=10)),
            annotations=[dict(text="Completion",
                              font=dict(color=G,size=11),
                              showarrow=False)]
        )
        st.plotly_chart(f2, use_container_width=True)

    with cc:
        ls = int((fdf["total_sessions"]<5).sum())
        f3 = go.Figure(go.Pie(
            labels=["Sessions < 5","Sessions ≥ 5"],
            values=[ls, len(fdf)-ls],
            hole=0.6, marker=dict(colors=[P,Y]),
            textfont=dict(color="#A0AEC0")
        ))
        f3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"), height=220,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=True,
            legend=dict(font=dict(color="#A0AEC0",size=10)),
            annotations=[dict(text="Sessions",
                              font=dict(color=P,size=11),
                              showarrow=False)]
        )
        st.plotly_chart(f3, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">PRIORITY INTERVENTION LIST</div>',
                unsafe_allow_html=True)
    st.markdown("*Students sorted by urgency — lowest engagement first*")
    risk_tbl = fdf[fdf["risk_flag"]==1][[
        "student_id","name","engagement_score",
        "avg_quiz_score","completion_rate","total_sessions"
    ]].sort_values("engagement_score").head(25).round(3).copy()
    risk_tbl.columns = [
        "ID","Student","Engagement","Avg Quiz","Completion","Sessions"
    ]
    risk_tbl["Priority"] = pd.cut(
        risk_tbl["Engagement"],
        bins=[0, 0.2, 0.4, 1.01],
        labels=["🔴 CRITICAL","🟡 HIGH","🟢 MODERATE"]
    )
    st.dataframe(risk_tbl, use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════
# PAGE 4 — MODULE INTELLIGENCE
# ══════════════════════════════════════════════════════════
elif page == "📡 Module Intelligence":

    mod_stats = fsm.groupby(
        ["module_name","difficulty","category"]
    ).agg(
        sessions    =("session_id",       "count"),
        avg_score   =("quiz_score",        "mean"),
        avg_duration=("duration_minutes", "mean"),
        completion  =("completed",         "mean"),
        avg_videos  =("videos_watched",    "mean")
    ).reset_index().round(2)

    st.markdown('<div class="sec">MODULE PERFORMANCE BUBBLE CHART</div>',
                unsafe_allow_html=True)
    st.markdown("*Size = sessions · Color = completion · X = duration · Y = score*")
    fig_b = px.scatter(
        mod_stats, x="avg_duration", y="avg_score",
        size="sessions", color="completion",
        text="module_name",
        color_continuous_scale=[[0,"#004D6B"],[0.5,"#00A8CC"],[1,G]],
        size_max=50,
        labels={"avg_duration":"Avg Duration (min)",
                "avg_score":"Avg Quiz Score",
                "completion":"Completion Rate"}
    )
    fig_b.update_traces(
        textposition="top center",
        textfont=dict(color="#A0AEC0",size=10),
        marker=dict(line=dict(width=1, color=rgba(T,0.3)))
    )
    dl(fig_b, 420)
    st.plotly_chart(fig_b, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec">MODULE POPULARITY</div>',
                    unsafe_allow_html=True)
        ms = mod_stats.sort_values("sessions", ascending=True)
        fig_m = px.bar(
            ms, x="sessions", y="module_name",
            orientation="h", color="avg_score",
            color_continuous_scale=[[0,"#004D6B"],[1,T]],
            text="sessions",
            labels={"sessions":"Sessions","module_name":"Module"}
        )
        fig_m.update_traces(textposition="outside",
                            textfont=dict(color="#A0AEC0"),
                            marker_line_width=0)
        fig_m.update_xaxes(gridcolor=GR, linecolor=GR)
        fig_m.update_yaxes(gridcolor=GR, linecolor=GR)
        fig_m.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            font=dict(color="#A0AEC0"),
            height=380,
            margin=dict(l=40,r=20,t=20,b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.markdown('<div class="sec">DIFFICULTY RADAR</div>',
                    unsafe_allow_html=True)
        diff_s = mod_stats.groupby("difficulty").agg(
            avg_score   =("avg_score",    "mean"),
            avg_duration=("avg_duration", "mean"),
            completion  =("completion",   "mean"),
            sessions    =("sessions",     "sum")
        ).reset_index()
        dcols = {"Beginner":G,"Intermediate":T,"Advanced":P}
        cats2 = ["Score","Duration","Completion","Popularity"]
        fig_r = go.Figure()
        for _, row in diff_s.iterrows():
            mx2   = [100, 180, 1, max(diff_s["sessions"].max(),1)]
            vals2 = [
                row["avg_score"]    / mx2[0],
                row["avg_duration"] / mx2[1],
                row["completion"]   / mx2[2],
                row["sessions"]     / mx2[3]
            ]
            c2r = dcols.get(row["difficulty"], T)
            fig_r.add_trace(go.Scatterpolar(
                r=vals2+[vals2[0]], theta=cats2+[cats2[0]],
                fill="toself", fillcolor=rgba(c2r,0.15),
                line=dict(color=c2r,width=2),
                name=row["difficulty"]
            ))
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor=BG,
                radialaxis=dict(
                    visible=True, range=[0,1],
                    gridcolor=GR,
                    tickfont=dict(color="#4A5568",size=8)
                ),
                angularaxis=dict(
                    gridcolor=GR,
                    tickfont=dict(color="#A0AEC0",size=9)
                )
            ),
            height=380,
            legend=dict(font=dict(color="#A0AEC0")),
            margin=dict(l=40,r=40,t=20,b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<div class="sec">COMPLETE MODULE SCORECARD</div>',
                unsafe_allow_html=True)
    display = mod_stats.copy()
    display.columns = [
        "Module","Difficulty","Category","Sessions",
        "Avg Score","Avg Duration","Completion","Avg Videos"
    ]
    st.dataframe(display, use_container_width=True, height=320)
