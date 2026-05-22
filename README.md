# ⚡ India Energy Transition & Grid Decarbonization Sandbox

An interactive, machine-learning-powered web dashboard designed to model and simulate India's electricity grid decarbonization, capacity expansion, and power sector emissions.

Built as a **static single-page application (HTML, CSS, JS)** using **Plotly.js and PapaParse**, this sandbox runs machine learning models entirely client-side in the browser. It is fully ready for zero-cost, permanent hosting on **GitHub Pages**.

---

## 🚀 Live Interactive Dashboard
The project is configured to run instantly on any web browser and can be hosted for free on **GitHub Pages**.

### Key Core Sections:
1. **📊 National Dashboard**: Track India's electricity capacity growth (Fossil vs. Clean), monthly generation fuel mix shares, and CO2 emissions intensity trends (2019-2025).
2. **🏢 State-Level Explorer**: Drill down into individual states or Union Territories (e.g., Gujarat, Rajasthan) to analyze their capacity mix, monthly generation, and rank states using dynamic leaderboards.
3. **🔮 ML Forecast Simulator**: 
   - **Electricity Demand Forecast**: A time-series forecasting model projecting India's electricity demand over the next 24 months.
   - **Grid Carbon Intensity Sandbox**: An interactive simulator where you adjust renewable shares and use a machine learning model to estimate the resulting grid carbon intensity ($gCO_2/kWh$) in real-time.

---

## 🧠 Client-Side Machine Learning Engine

The machine learning models are executed in pure JavaScript inside `app.js` using coefficients trained in the project's [Jupyter Notebook](india_energy_analysis_ml.ipynb):

### 1. Autoregressive Ridge demand Forecaster (MAPE: 2.85%)
* **Task**: Forecast national monthly generation requirement (GWh).
* **Formula**: Autoregressive rolling forecast using trend, monthly seasonality dummies, and autoregressive lag variables (Lag-1, Lag-2, and Lag-12).
* **Validation**: Out-of-time validation on the 2025 dataset shows a Mean Absolute Percentage Error (MAPE) of **2.85%**.

### 2. Grid Carbon Intensity Predictor ($R^2 \approx 100\%$)
* **Task**: Predict grid CO2 intensity ($gCO_2/kWh$) based on electricity generation fuel shares.
* **Formula**: Linear Regression model mapping the exact carbon contributions of Coal, Gas, Solar, Wind, Hydro, and Nuclear shares.
* **Validation**: R-squared of **0.999997** and MAE of **0.08 gCO2/kWh** against historical carbon data, representing a mathematically exact fit for grid emissions calculation.

---

## 📁 Repository Structure

```filepath
├── index.html                         # Dashboard markup and layouts
├── styles.css                         # Glassmorphic dark mode styling
├── app.js                             # App routing, charts, & client-side ML
├── pivoted_india_total_monthly.csv    # Cleaned pivoted national data
├── pivoted_states_monthly.csv         # Cleaned pivoted state data
├── build_project.py                   # Automation script for raw data pipeline
├── india_energy_analysis_ml.ipynb     # Pre-rendered Data Science Jupyter Notebook
├── requirements.txt                   # Optional dependencies for the raw pipeline
└── README.md                          # Project documentation (this file)
```

---

## 🛠️ Local Setup & Run

No compilation, Node.js, or Python backends are required to run the dashboard!

### Option A: Open Directly
Simply double-click the `index.html` file to open it in your web browser. 

*Note: Some browsers block local file reading (CORS) when accessing CSV files. If charts do not load, use Option B.*

### Option B: Local Web Server (Recommended)
Run a quick local server in the project folder:
```bash
# Using Python
python -m http.server 8000

# Using Node (if installed)
npx serve .
```
Then open your browser and navigate to `http://localhost:8000`.

---

## ☁️ Deploying to GitHub Pages (Free Hosting)

Host this project live on GitHub in seconds with these steps:

### Step 1: Push the Repository to GitHub
Create a new repository on GitHub (e.g., `India_energy_prediction_project`) and run:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Enable GitHub Pages
1. Go to your repository page on **GitHub**.
2. Click on the **Settings** tab.
3. In the left sidebar, under "Code and automation", click **Pages**.
4. Under "Build and deployment", set the source to **Deploy from a branch**.
5. Under "Branch", select `main` and `/ (root)`, then click **Save**.
6. Refresh the page after 1 minute; GitHub will display your live URL (e.g., `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`).
