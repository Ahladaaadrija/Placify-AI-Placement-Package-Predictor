Placify AI — Placement Package Predictor

An AI-powered Streamlit dashboard that predicts a student's expected
placement package (in LPA) using Linear Regression, based on CGPA,
internships, projects, certifications, and skill scores.

---

## 1. How this project works (read this first)

| File | What it does |
|---|---|
| `generate_data.py` | Creates `placement_data.csv` — a synthetic dataset of 1200 student records. (Swap this for your college's real placement data later.) |
| `train_model.py` | Loads the CSV, trains a `LinearRegression` model from scikit-learn, prints accuracy metrics (R², MAE, RMSE), and saves the trained model to `placify_model.pkl`. |
| `app.py` | The Streamlit dashboard. Loads `placify_model.pkl`, takes user input via sliders/number boxes, and shows the predicted package plus charts. |
| `requirements.txt` | The exact Python libraries needed, so anyone (including Streamlit Cloud) can recreate your environment. |

**The core ML concept — Linear Regression in one sentence:**
It finds the best straight-line relationship between input numbers (CGPA, internships, etc.)
and an output number (package), represented as:

```
package = intercept + (coef1 × CGPA) + (coef2 × Internships) + ... + (coefN × Backlogs)
```

Training = the algorithm finding the best values for each `coef` so its predictions
are as close as possible to the real packages in your data.

---

## 2. Run it locally — step by step

### Step 1: Install Python
You need Python 3.10+ installed (check with `python3 --version`).

### Step 2: Get the project files
Download the files attached in this conversation into a folder called `placify-ai`,
or follow the GitHub steps below first and clone your repo.

### Step 3: Create a virtual environment (recommended, keeps things clean)
```bash
cd placify-ai
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Generate the dataset
```bash
python3 generate_data.py
```
This creates `placement_data.csv`.

### Step 6: Train the model
```bash
python3 train_model.py
```
This prints your model's accuracy and creates `placify_model.pkl`.

### Step 7: Launch the dashboard
```bash
streamlit run app.py
```
Your browser will open `http://localhost:8501` with the live dashboard.

---

## 3. Push it to GitHub (repository building)

```bash
cd placify-ai
git init
git add .
git commit -m "Initial commit: Placify AI placement predictor"
```

Then on GitHub.com:
1. Click **New repository** → name it `placify-ai` → don't initialize with a README (you already have one) → **Create repository**.
2. Copy the commands GitHub shows you (something like):
```bash
git remote add origin https://github.com/<your-username>/placify-ai.git
git branch -M main
git push -u origin main
```

**Important:** Make sure `placement_data.csv` and `placify_model.pkl` are committed too
(don't gitignore them) — Streamlit Cloud needs them to run the app, since this version
loads a pre-trained model rather than training on the fly.

---

## 4. Deploy for free on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **New app**.
3. Select your `placify-ai` repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.
5. Wait 1–2 minutes — Streamlit Cloud reads `requirements.txt`, installs everything, and gives you a public URL like:
   `https://placify-ai-yourname.streamlit.app`

That link is what you put in your resume/portfolio as "Link".

---

## 5. Ways to extend this project (good for interviews)

- **Try other models** — compare Linear Regression against `RandomForestRegressor` or
  `XGBoost` and show R² side by side (great talking point: "I benchmarked 3 models").
- **Use real data** — survey seniors/placement cell for actual anonymized records.
- **Add SHAP values** for deeper interpretability instead of just raw coefficients.
- **Add authentication** so each student logs in and sees their own prediction history.
- **Branch-wise / company-wise prediction** — add categorical features (department,
  company tier) using one-hot encoding.

---

## 6. Talking points for your resume/interview

- "Engineered a predictive analytics platform... based on academic performance and
  skill-based parameters" → you can explain exactly which 7 features you used and why.
- "Architected an interactive Streamlit dashboard... interpretability of key factors" →
  point to the coefficient bar chart in Tab 1 — that's literally interpretability:
  showing which inputs push the prediction up or down for that specific student.
