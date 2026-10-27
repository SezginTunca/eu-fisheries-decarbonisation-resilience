# Operational Decarbonisation Supports Enterprise Economic Resilience Across European Commercial Fishing Fleets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23688882.svg)](https://doi.org/10.5281/zenodo.23688882)

**Author:** Sezgin Tunca (ORCID: [0000-0002-7369-4636](https://orcid.org/0000-0002-7369-4636))  
**Affiliation:** University of Southern Denmark, Department of Business and Sustainability

---

## Overview

This repository provides the complete data-handling, cleaning, bio-economic fusion, econometric estimation, policy micro-simulation, and publication visualisation pipeline for the study:

> **Operational Decarbonisation Supports Enterprise Economic Resilience Across European Commercial Fishing Fleets**

In accordance with open-science reproducibility standards, this repository distributes the complete, self-contained Python analytical codebase. Underlying primary datasets are publicly accessible directly from the official European Union, international fisheries, and macroeconomic statistical repositories detailed below.

---

## Repository Structure

```text
├── scripts/
│   ├── 01_prepare_data.py              # STECF panel ingestion, deflation, winsorization & Scope 1 LCA calculation
│   ├── 01b_fuse_biology_economics.py   # Fusion of ICES Stock Assessment & FAO GFCM biological indicators (SAR, B/Bmsy)
│   ├── 01c_fetch_external.py           # Automated API retrieval of OECD FSE, Eurostat, World Bank & EU ETS series
│   ├── 02_decarbonisation_analysis.py  # TWFE, Tapio decoupling, Oster bounds, MM-QR, Ward clustering & tax/rebate simulations
│   └── 03_visualisations.py   # High-resolution publication figure suite (Figures 1–5 and Figures S1–S6)
├── requirements.txt                    # Python package dependencies
├── .zenodo.json                        # Zenodo archival metadata descriptor
└── LICENSE                             # MIT Open-Source Software License
```

---

## Public Data Sources & Acquisition

To execute the pipeline end-to-end, download the public primary releases from the following official agencies and place them in `data/raw/` (or run `scripts/01c_fetch_external.py` for automated API retrieval of macroeconomic and policy series):

1. **European Commission Joint Research Centre (JRC) & STECF**:
   - *Annual Economic Report (AER) on the EU Fishing Fleet (STECF 24-03 & 24-07)*: Fleet segment socio-economic, effort, fuel consumption, and landings records spanning 22 coastal EU Member States (2008–2023). Available at: <https://stecf.jrc.ec.europa.eu/reports/economic>
2. **International Council for the Exploration of the Sea (ICES) & FAO GFCM**:
   - *ICES Stock Assessment Graphs (SAG) Database* and *FAO General Fisheries Commission for the Mediterranean (GFCM) Stock Status Portal*: Stock-level spawning stock biomass ($B/B_{\text{MSY}}$), fishing mortality ($F/F_{\text{MSY}}$), and Stocks-at-Risk ($\text{SAR}$) indicators. Available at: <https://standardgraphs.ices.dk> and <https://www.fao.org/gfcm/data/en/>
3. **OECD Fisheries Support Estimate (FSE) Database**:
   - Member State direct operational support and marine fuel tax concession series (`OECD.TAD.ARP,DSD_FSE@DF_FSE`). Retrieved automatically via `scripts/01c_fetch_external.py`.
4. **Eurostat, World Bank WDI & EU ETS**:
   - Harmonised Index of Consumer Prices (HICP, `2015 = 100`), environmental tax revenues (`env_ac_tax`), World Bank governance and macroeconomic controls, and historical European Union Emissions Trading System (EU ETS) allowance prices. Retrieved automatically via `scripts/01c_fetch_external.py`.

---

## Replication Pipeline Execution

### 1. Environment Setup
Install the required Python 3.10+ dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation & Bio-Economic Fusion
```bash
python3 scripts/01_prepare_data.py
python3 scripts/01b_fuse_biology_economics.py
python3 scripts/01c_fetch_external.py
```
- Harmonises segment identifiers across Country $\times$ Fishing Technology $\times$ Vessel Length Class strata.
- Applies country-year HICP deflation (`2015 = 100`), computes physical carbon intensity ($\text{kg CO}_2\text{ per kg catch}$) and real monetary carbon intensity ($\text{kg CO}_2\text{ per 2015 EUR revenue}$), constructs the non-fuel operating margin to eliminate mechanical fuel-cost accounting overlap, and merges biological stock status and external policy indicators.

### 3. Econometric Estimation & Carbon Pricing Micro-Simulation
```bash
python3 scripts/02_decarbonisation_analysis.py
```
- Estimates Two-Way Fixed Effects (TWFE) models with Driscoll-Kraay and two-way clustered standard errors, Tapio aggregate decoupling elasticities, Oster (2019) coefficient stability bounds ($\delta$), Method of Moments Quantile Regression (MM-QR) across $\tau \in [0.10, 0.90]$, Ward's hierarchical cluster typology ($k=4$) with 1,000-iteration bootstrap Jaccard validation, non-linear Pareto frontier parameters, and revenue-neutral carbon tax and rebate micro-simulations (€0–€250/$\text{tCO}_2$).

### 4. Publication Visualisations
```bash
python3 scripts/03_visualisations.py
```
- Generates all main-text figures (`Fig1_Temporal_Trends` through `Fig5_LCA_Validation`) and supplementary figures (`FigS1_Age_Carbon_Intensity` through `FigS6_Diversification_Buffering`) at 600 DPI PNG and vector PDF formats in `figures/`.

---

## License

This software code is released under the **MIT License** (see [LICENSE](LICENSE)).
