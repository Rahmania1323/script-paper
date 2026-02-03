# ATL Weather Delay Analysis

This repository contains analysis code and data for the study:

**“Analysis of the Impact of Weather Conditions on Flight Delays at Hartsfield–Jackson Atlanta International Airport (ATL) During 2013–2023”**, intended for submission to the *International Journal of Climatology*.

---

## 📄 Description

This repository provides scripts to investigate the impact of meteorological conditions on flight delays at ATL airport during 2013–2023.  

- **Python (Jupyter Notebook)** scripts handle data preprocessing, regression analyses, and generation of Figures 2, 4, 5, and 6, as well as Tables 2 and 3.  
- **MATLAB** scripts are used specifically for seasonal statistics computation and visualization (Figure 3 and Table 1).

---

## 📁 Repository Structure

atl-weather-delay-analysis/
├── data/ ← Raw datasets
│ ├── Data Variabel Meteorologi.xlsx
│ ├── Data Variabel Meteorologi.csv
│ ├── data speed and direction.xlsx
│ └── data speed and direction.csv
├── outputs/ ← Generated figures and CSV results
│ ├── ANOVA_Summary.csv
│ └── Seasonal_Analysis_ATL.png
├── scripts/ ← Analysis scripts
│ ├── Seasonal_mean_weather_delay_analysis_ATL.m
│ └── atl_weather_delay_analysis.ipynb
├── .gitignore
├── LICENSE
└── README.md


- `data/` → source Excel and CSV datasets  
- `outputs/` → generated figures and CSV summaries  
- `scripts/` → Python notebooks and MATLAB scripts  

---

## 🔗 Data Sources

- **Meteorological data:** NASA POWER (Prediction Of Worldwide Energy Resources)  
  [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/)  

- **Flight operational data:** Bureau of Transportation Statistics (BTS)  
  [Weather’s Share of Delayed Flights database](https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?6B2r=G&20=E)  

---

## ▶️ How to Run

### Python Notebook

1. Install dependencies:

```bash
pip install -r requirements.txt

