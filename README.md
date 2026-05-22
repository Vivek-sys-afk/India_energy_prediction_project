# ⚡ India Energy Transition & Grid Decarbonization Sandbox

An interactive, machine-learning-powered data science platform designed to model and simulate India's electricity grid decarbonization, capacity expansion, and power sector emissions.

Built using **Python, Streamlit, Plotly, and Scikit-Learn**, this application translates raw long-format electricity statistics into actionable insights and a live transition sandbox.

---

## 🚀 Live Interactive Dashboard
The project is configured for instant deployment to **Streamlit Community Cloud** directly from GitHub.

### Key Core Sections:
1. **📊 National Dashboard**: Track India's electricity capacity growth (Fossil vs. Clean), monthly generation fuel mix shares, and CO2 emissions intensity trends (2019-2025).
2. **🏢 State-Level Explorer**: Drill down into individual states or Union Territories (e.g., Gujarat, Rajasthan) to analyze their capacity mix, monthly generation, and rank states using dynamic leaderboards.
3. **🔮 ML Forecast Simulator**: 
   - **Electricity Demand Forecast**: A time-series forecasting model projecting India's electricity demand over the next 24 months.
   - **Grid Carbon Intensity Sandbox**: An interactive simulator where you adjust renewable shares and use a machine learning model to estimate the resulting grid carbon intensity ($gCO_2/kWh$).

---

## 🧠 Machine Learning Engine

The project utilizes two machine learning models developed in the [Jupyter Notebook](india_energy_analysis_ml.ipynb):

### 1. Autoregressive Ridge demand Forecaster (MAPE: 2.85%)
* **Task**: Forecast national monthly generation requirement (GWh).
* **Features**: Trend component, monthly seasonality dummies, and autoregressive lag variables (Lag-1, Lag-2, and Lag-12).
* **Validation**: Out-of-time validation on the 2025 dataset shows a Mean Absolute Percentage Error (MAPE) of **2.85%**.

### 2. Random Forest emissions Regressor ($R^2 > 98\%$)
* **Task**: Predict grid CO2 intensity ($gCO_2/kWh$) based on electricity generation fuel shares.
* **Features**: Coal share, Gas share, Hydro share, Nuclear share, Solar share, and Wind share.
* **Insight**: Coal share accounts for **96.4%** of the Gini importance, indicating that grid decarbonization remains fundamentally constrained by coal load factor displacement.

---

## 📁 Repository Structure

```filepath
├── pivoted_india_total_monthly.csv    # Cleaned pivoted national data
├── pivoted_states_monthly.csv         # Cleaned pivoted state data
├── emissions_rf_model.pkl             # Pre-trained Random Forest model
├── app.py                             # Streamlit dashboard application
├── build_project.py                   # Automation script for pipeline & notebook
├── india_energy_analysis_ml.ipynb     # Fully executed Jupyter Notebook with plots
├── requirements.txt                   # Dependency list for environment setup
└── README.md                          # Project documentation (this file)
```

---

## 🛠️ Local Setup & Run

Follow these instructions to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd India_energy_prediction_project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
This will open the dashboard in your default browser at `http://localhost:8501`.

*(Optional)* If you wish to re-run the entire pipeline (processing raw files, training the models, and re-executing the Jupyter notebook), run:
```bash
python build_project.py
```

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

You can host this project live on the web for free using Streamlit Community Cloud. Follow these steps:

1. **Push to GitHub**: Create a repository on GitHub and push this codebase.
   ```bash
   git init
   git add .
   git commit -m "Initial commit: India Grid transition ML sandbox"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```
2. **Sign up/Log in to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
3. **Deploy App**:
   - Click **"New app"**.
   - Select your repository (`YOUR_USERNAME/YOUR_REPO_NAME`), branch (`main`), and main file path (`app.py`).
   - Click **"Deploy!"**.
4. **Interact**: Your dashboard will be live on a public URL in less than a minute!

---

## 📊 Data Source
This project uses processed historical monthly capacity, generation, and emissions datasets for India's power sector, covering 2019 to late 2025.
