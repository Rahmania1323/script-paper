# script-paper

This repository contains the analysis code originally written in a Jupyter Notebook for the study:

**"Analysis of the Impact of Weather Conditions on Flight Delays at Hartsfield–Jackson Atlanta International Airport (ATL) During 2013–2023"**, intended for submission to *International Journal of Climatology*.

## 📄 Description

This repository contains the analysis scripts used to investigate the impact of meteorological conditions on flight delays at Hartsfield–Jackson Atlanta International Airport (ATL) during 2013–2023.

Python (Jupyter Notebook–based) scripts were used for data preprocessing, regression analyses, and the generation of most figures and tables (Figures 2, 4, 5, and 6; Tables 2 and 3). MATLAB was used specifically for the computation and visualization of seasonal statistics presented in Figure 3 and Table 1.

## 📁 Contents

- `scripts/code_paper_converted.py`: Auto-generated script from the original Jupyter Notebook.
- `requirements.txt`: Python dependencies.
- `figures/`: (Optional) Output plots from the analysis.

## 🔗 Data Sources

- **Meteorological Data**: NASA POWER — https://power.larc.nasa.gov  
- **Flight Delay Data**: BTS Transtats — https://www.transtats.bts.gov

## ▶️ How to Run

```bash
pip install -r requirements.txt
python scripts/code_paper_converted.py
