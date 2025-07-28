# ⚡ Anomaly Detection in Electricity Market Prices (DK1)

This project focuses on detecting anomalous price events in the Danish electricity market (zone: DK1) using historical time-series data. The goal is to build a foundational framework for identifying irregular price behavior that could signal operational issues, forecasting errors, or market manipulation.

---

## 📌 Project Objective

To build a data-driven anomaly detection pipeline for electricity prices that:

- Ingests and processes historical market data
- Applies statistical and machine learning techniques to detect anomalies
- Visualizes results in an interpretable and accessible format

This framework is adaptable and can be extended to other bidding zones or energy datasets.

---

## 🧠 Methodology

### 1. **Data Preparation**
- Imported DK1 electricity price data from Excel files (2023–2025)
- Handled datetime formatting, missing values, and outliers
- Unified time-series granularity for consistent analysis

### 2. **Feature Engineering**
- Converted prices to `pence/kWh` for normalization
- Calculated rolling statistics (mean, std)
- Added timestamp features: hour, day of week, etc.

### 3. **Anomaly Detection Logic**
- **Z-Score Based Detection**:
  - Flagged price points that deviate significantly from local mean (based on rolling windows)
  - Threshold tuning to balance noise vs. signal

- **Visual Validation**:
  - Matplotlib-based time-series visualization
  - Color-coded anomalies for easy inspection

> ⚙️ Note: This pipeline is currently rule-based. Future upgrades may include isolation forests, LSTM models, or real-time detection logic.

---

## 📊 Tools & Technologies

- **Python 3**
- **Pandas**, **NumPy** for data wrangling
- **Matplotlib**, **Seaborn** for visualization
- **Jupyter Notebook** for analysis and documentation

---

## 🔄 Future Improvements

- Integrate machine learning-based anomaly models (Isolation Forest, Autoencoder, etc.)
- Deploy interactive dashboards using **Streamlit** or **Plotly Dash**
- Real-time anomaly alerts for live market data

---

## 👤 Author

**Ramkumar Kannan**  
MSc Business Analytics | Data Analyst | Energy Enthusiast  
📍 Edinburgh, UK  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile)
