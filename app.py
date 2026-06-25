"""
app.py
-------
PLACIFY AI - Streamlit dashboard.
Editorial / portfolio-style redesign: huge type, overline labels,
single strong accent color, generous whitespace.
Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

st.set_page_config(page_title="Placify AI", layout="wide")

# ============================================================
# GLOBAL STYLING
# ============================================================
ACCENT = "#C9FF3D"   # acid-green accent, like the Image 1 / Image 2 single-accent approach
BG = "#0A0A0A"
CARD_BG = "#121212"
TEXT_MUTED = "#8A8A8A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;900&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Archivo', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.stApp {{
    background: {BG};
}}

/* Overline label, like "... / About project ..." */
.overline {{
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: {ACCENT};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}

/* Giant headline */
.mega-title {{
    font-family: 'Archivo', sans-serif;
    font-weight: 900;
    font-size: 5rem;
    line-height: 0.95;
    color: #F5F5F5;
    letter-spacing: -0.02em;
    margin: 0;
}}
.mega-title .accent {{ color: {ACCENT}; }}

.hero-sub {{
    color: {TEXT_MUTED};
    font-size: 1.05rem;
    max-width: 640px;
    margin-top: 1.2rem;
    line-height: 1.6;
}}

.divider {{
    border-top: 1px solid #232323;
    margin: 2.2rem 0 2.2rem 0;
}}

/* Section label row */
.section-label {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.4rem;
}}
.section-label .num {{
    font-family: 'Space Mono', monospace;
    color: {ACCENT};
    font-size: 0.9rem;
    border: 1px solid {ACCENT};
    border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
}}
.section-label .label {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #F5F5F5;
    text-transform: uppercase;
}}

/* Card */
.card {{
    background: {CARD_BG};
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}}

/* Result panel - huge number like editorial typography */
.result-box {{
    background: {ACCENT};
    border-radius: 4px;
    padding: 2.4rem 2.2rem;
    margin-top: 1rem;
}}
.result-label {{
    font-family: 'Space Mono', monospace;
    color: #0A0A0A;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.7;
}}
.result-value {{
    font-family: 'Archivo', sans-serif;
    font-weight: 900;
    font-size: 4rem;
    color: #0A0A0A;
    line-height: 1;
    margin-top: 0.3rem;
}}

/* Widget label polish */
div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label {{
    color: #C9C9C9 !important;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Button - sharp, not rounded-pill, matches the harder editorial look */
.stButton > button {{
    background: {ACCENT};
    color: #0A0A0A;
    border: none;
    border-radius: 2px;
    padding: 0.85rem 2rem;
    font-weight: 800;
    font-family: 'Archivo', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.95rem;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    background: #F5F5F5;
    transform: translateY(-1px);
}}

/* Tabs styled like top-nav pills (Starbucks-ref nav) */
div[data-baseweb="tab-list"] {{
    gap: 6px;
    background: {CARD_BG};
    padding: 6px;
    border-radius: 999px;
    width: fit-content;
}}
button[data-baseweb="tab"] {{
    font-family: 'Space Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    border-radius: 999px !important;
    color: {TEXT_MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: {ACCENT} !important;
    color: #0A0A0A !important;
}}
div[data-baseweb="tab-highlight"] {{ background: transparent !important; }}
div[data-baseweb="tab-border"] {{ display: none !important; }}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}}
div[data-testid="stMetricLabel"] {{
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
}}
</style>
""", unsafe_allow_html=True)

# Matplotlib theme to match
mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#2A2A2A",
    "axes.labelcolor": "#C9C9C9",
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "text.color": "#F5F5F5",
    "grid.color": "#1E1E1E",
    "font.family": "sans-serif",
})

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("placify_model.pkl")

bundle = load_model()
model = bundle["model"]
FEATURES = bundle["features"]
r2 = bundle["r2"]
mae = bundle["mae"]

@st.cache_data
def load_data():
    return pd.read_csv("placement_data.csv")

df = load_data()

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="overline">// AI &nbsp; PLACEMENT &nbsp; INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="mega-title">PLACIFY<span class="accent">.</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A predictive analytics platform that forecasts placement packages '
    'from academic performance and skill signals — built on Linear Regression, '
    'fully interpretable, no black box.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Predict", "Insights", "Model"])

# ===================== TAB 1: PREDICT =====================
with tab1:
    st.markdown("""
    <div class="section-label">
        <div class="num">01</div>
        <div class="label">Enter Student Profile</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.05)
        internships = st.number_input("Internships completed", 0, 10, 1)
        backlogs = st.number_input("Active Backlogs", 0, 5, 0)
    with col2:
        projects = st.number_input("Projects completed", 0, 15, 2)
        certifications = st.number_input("Certifications", 0, 10, 1)
    with col3:
        coding_score = st.slider("Coding Score (0-100)", 0, 100, 65)
        communication_score = st.slider("Communication Score (0-100)", 0, 100, 70)
    st.markdown('</div>', unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Certifications": certifications,
        "Coding_Score": coding_score,
        "Communication_Score": communication_score,
        "Backlogs": backlogs,
    }])[FEATURES]

    predict_clicked = st.button("Predict My Package →", type="primary")

    if predict_clicked:
        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 0)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Placement Package</div>
            <div class="result-value">₹ {prediction:.2f} LPA</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("""
        <div class="section-label">
            <div class="num">02</div>
            <div class="label">Why This Number</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        contributions = (input_df.iloc[0] * model.coef_)
        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions.values
        }).sort_values("Contribution", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#FF4D4D" if v < 0 else ACCENT for v in contrib_df["Contribution"]]
        ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors, height=0.5)
        ax.set_xlabel("Contribution to predicted package (LPA)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== TAB 2: INSIGHTS =====================
with tab2:
    st.markdown("""
    <div class="section-label">
        <div class="num">01</div>
        <div class="label">Dataset Overview</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-label">
            <div class="num">02</div>
            <div class="label">Correlations</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = sns.light_palette(ACCENT, as_cmap=True)
        sns.heatmap(df.corr(), annot=True, cmap="mako", fmt=".2f", ax=ax,
                    cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor=BG)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-label">
            <div class="num">03</div>
            <div class="label">CGPA vs Package</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x="CGPA", y="Package_LPA", hue="Internships",
                         palette="cool", ax=ax2, s=45, edgecolor="none")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        ax2.grid(alpha=0.12)
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== TAB 3: MODEL =====================
with tab3:
    st.markdown("""
    <div class="section-label">
        <div class="num">01</div>
        <div class="label">Model Performance</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("Mean Absolute Error", f"{mae:.2f} LPA")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
        <div class="num">02</div>
        <div class="label">Feature Weights</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    st.caption("Positive = increases predicted package. Negative = decreases it (e.g. Backlogs).")
    st.markdown('</div>', unsafe_allow_html=True)