# script-paper

This repository contains the analysis code originally written in a Jupyter Notebook for the study:

**"Analysis of the Impact of Weather Conditions on Flight Delays at Hartsfield–Jackson Atlanta International Airport (ATL) During 2013–2023"**, intended for submission to *International Journal of Climatology*.

## 📄 Description

This repository contains the analysis scripts used to investigate the impact of meteorological conditions on flight delays at Hartsfield–Jackson Atlanta International Airport (ATL) during 2013–2023.

Python (Jupyter Notebook–based) scripts were used for data preprocessing, regression analyses, and the generation of most figures and tables (Figures 2, 4, 5, and 6; Tables 2 and 3). MATLAB was used specifically for the computation and visualization of seasonal statistics presented in Figure 3 and Table 1.

## 📁 Contents

- `scripts/code_paper_converted.py`: Python script converted from the original Jupyter Notebook and used for data processing and analysis.
- `requirements.txt`: List of Python dependencies required to reproduce the analysis.
- `figures/`: Figures included in the manuscript and generated from the analysis.

## 🔗 Data Sources

- **Meteorological data**: NASA POWER (Prediction Of Worldwide Energy Resources), accessed via the NASA POWER Data Access Viewer  
  https://power.larc.nasa.gov/data-access-viewer/

- **Flight Operational Data**: Bureau of Transportation Statistics (BTS),  
  Weather’s Share of Delayed Flights database (Atlanta, GA – ATL)  
  https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?6B2r=G&20=E

## ▶️ How to Run

```bash
pip install -r requirements.txt
python scripts/code_paper_converted.py
