"""
app.py
-------
PLACIFY AI - Streamlit dashboard.
Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Placify AI", layout="wide")

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

# ---------- Header ----------
st.title("Placify AI — Placement Package Predictor")
st.caption("Predict a student's expected placement package (LPA) using Linear Regression, "
           "based on academic performance and skill-based parameters.")

tab1, tab2, tab3 = st.tabs([" Predict", " Data Insights", " Model Info"])

# ===================== TAB 1: PREDICT =====================
with tab1:
    st.subheader("Enter Student Details")

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

    input_df = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Certifications": certifications,
        "Coding_Score": coding_score,
        "Communication_Score": communication_score,
        "Backlogs": backlogs,
    }])[FEATURES]

    if st.button("Predict My Package", type="primary"):
        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 0)
        st.success(f"### Predicted Placement Package: ₹ {prediction:.2f} LPA")

        # Show which factors contributed most for THIS student
        contributions = (input_df.iloc[0] * model.coef_)
        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions.values
        }).sort_values("Contribution", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#d9534f" if v < 0 else "#5cb85c" for v in contrib_df["Contribution"]]
        ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors)
        ax.set_xlabel("Contribution to predicted package (LPA)")
        ax.set_title("What's driving your prediction")
        st.pyplot(fig)

# ===================== TAB 2: DATA INSIGHTS =====================
with tab2:
    st.subheader("Dataset Overview")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("CGPA vs Package")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="CGPA", y="Package_LPA", hue="Internships", palette="viridis", ax=ax2)
    st.pyplot(fig2)

# ===================== TAB 3: MODEL INFO =====================
with tab3:
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("Mean Absolute Error", f"{mae:.2f} LPA")

    st.subheader("Feature Importance (Linear Regression Coefficients)")
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    st.caption("A positive coefficient means increasing that feature tends to increase the "
               "predicted package; negative means it tends to decrease it (e.g. Backlogs).")