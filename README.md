<div align="center">

# 📦 AI Assisted EDA — Anomaly Intelligence Engine

**Detect anomalies in any dataset using Machine Learning, Statistics, and AI-powered diagnostics.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

</div>

---

## What Does This Project Do?

This is an **end-to-end anomaly detection system** built as a Streamlit web application. It takes tabular data (CSV files), runs multiple anomaly detection algorithms on it, and presents the results through an interactive dashboard with visualizations and AI-generated explanations.

**Two ways to use it:**

| Mode | What Happens |
|------|-------------|
| 📊 **Sample Analysis** | The app comes pre-loaded with a real e-commerce dataset (96K+ orders from Brazilian marketplace OLIST). Open the app and instantly see the full analysis — no setup needed. |
| 📄 **Upload Your Own** | Upload any CSV file. The engine auto-detects numeric columns and runs anomaly detection on your data. |

---

## How It Works

The system uses **three independent anomaly detection engines** that work together:

```
Your Data (CSV)
     │
     ├──▶ Engine 1: Isolation Forest (ML)
     │        Trains an unsupervised ML model to find records that
     │        "don't fit" across multiple numeric dimensions at once.
     │
     ├──▶ Engine 2: IQR Statistical Detection
     │        Uses interquartile range math to flag extreme outliers
     │        in any single numeric column (optionally grouped by category).
     │
     └──▶ Engine 3: Time-Series 3σ Control Chart (Sample mode only)
              Tracks daily order volume over time and flags days where
              volume dropped below 3 standard deviations from the rolling mean.
              
     All flagged anomalies get:
     ├──▶ Priority severity labels (P1 Critical → P3 Info)
     └──▶ AI root-cause narratives (LLM-powered or rule-based fallback)
```

### The Anomaly Detection Algorithms Explained

#### 1. Isolation Forest (Machine Learning)
- **What:** An unsupervised ensemble algorithm from scikit-learn
- **How:** Builds 100 random decision trees. Anomalies are points that get isolated (separated from the rest) in fewer splits than normal data
- **Why this one:** Works without labeled data, scales well to 96K+ records, handles multiple features simultaneously
- **Parameters:** `contamination` (expected % of anomalies, default 1%) — adjustable via sidebar slider

#### 2. IQR (Interquartile Range) Detection
- **What:** A classical statistical method for finding outliers
- **How:** Calculates Q1 (25th percentile) and Q3 (75th percentile). Any value outside `Q3 + 1.5×IQR` or below `Q1 - 1.5×IQR` is flagged
- **Why this one:** Simple, interpretable, works per-group (e.g., flag delays per state/region separately)
- **Parameters:** `IQR multiplier` (default 1.5) — adjustable via sidebar slider

#### 3. Time-Series Control Chart (3σ)
- **What:** Statistical Process Control (SPC) applied to daily order counts
- **How:** Computes a 7-day rolling mean and standard deviation. Any day where order count falls below `mean - 3σ` is flagged as an anomaly
- **Why this one:** Catches sudden platform outages or demand drops that other methods miss

### AI Root-Cause Narratives

Every flagged anomaly gets an automated 3-bullet explanation:

1. **The Anomaly** — What went wrong
2. **Potential Root Cause** — Operational hypothesis
3. **Actionable Fix** — What to investigate next

These are generated via the **OpenRouter LLM API** (GPT/Claude/etc). If no API key is configured, a **deterministic rule-based engine** kicks in as fallback — so the app always works, with or without an LLM.

---

## Tech Stack

| Technology | Role |
|-----------|------|
| **Python 3.9+** | Core language |
| **Streamlit** | Web dashboard framework |
| **Pandas / NumPy** | Data manipulation and feature engineering |
| **Scikit-Learn** | Isolation Forest ML model |
| **Plotly** | Interactive dark-themed charts and visualizations |
| **OpenRouter API** | LLM-powered anomaly narratives (optional) |

---

## Project Structure

```
AI Assisted EDA/
├── app.py                        # Streamlit dashboard (UI, charts, routing)
├── ai_assisted_eda_agent.py      # Core anomaly detection pipeline (all ML/stats logic)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules (protects API keys)
├── .streamlit/
│   ├── config.toml               # Dark theme + server settings
│   └── secrets.toml              # Your API key (gitignored — never pushed)
├── OLIST Dataset/                # Sample e-commerce data (4 CSVs, ~38MB)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   └── olist_customers_dataset.csv
├── README.md                     # This file
└── interview_guide.md            # Interview Q&A guide for this project
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-assisted-eda.git
cd ai-assisted-eda
pip install -r requirements.txt
```

### 2. (Optional) Add your OpenRouter API key

Create `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "your-key-here"
```
Get a free key at [openrouter.ai](https://openrouter.ai) → Keys → Create Key.

> **Without a key**, the app still works — it uses rule-based narratives instead of LLM-generated ones.

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The sample analysis loads immediately.

---

## Deployment (Streamlit Community Cloud — Free)

This is the easiest way to put your app online:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/ai-assisted-eda.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New App** → select your GitHub repo
   - Set main file: `app.py`
   - Under **Advanced Settings → Secrets**, paste:
     ```toml
     OPENROUTER_API_KEY = "your-key-here"
     ```
   - Click **Deploy** 🚀

Your app will be live at `https://your-app-name.streamlit.app` in ~2 minutes.

---

## Sample Dataset

The app includes the [Brazilian E-Commerce (OLIST) dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — a real-world dataset of 96,460 orders from a Brazilian online marketplace (2016–2018).

**What the pipeline extracts from it:**

| Metric | How It's Computed |
|--------|-------------------|
| Delivery Time | `delivered_date - purchase_date` (in hours) |
| Prep Time | `carrier_handoff_date - approved_date` (in hours) |
| Delivery Delay | `delivered_date - estimated_date` (negative = early) |
| Order Value | Sum of all payment values per order |
| Freight Cost | Sum of freight values per order |

---

## Configuration

Both parameters are adjustable in real-time via the sidebar:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| **Contamination Rate** | 1% | How many records Isolation Forest expects to be anomalous (lower = fewer flags) |
| **IQR Multiplier** | 1.5 | How far from the IQR bounds a value must be to count as an outlier (higher = only extreme outliers) |

---

## License

MIT License — free to use, modify, and distribute.

<div align="center">

**Built with Streamlit, Scikit-Learn, Plotly & OpenRouter**

⭐ Star this repo if you find it useful!

</div>
