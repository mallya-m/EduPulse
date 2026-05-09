# EduPulse — Statistical Intelligence Engine for E-Learning Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

EduPulse is a four-stage statistical analytics pipeline that transforms raw
e-learning interaction logs into actionable predictions. It detects at-risk
students early, quantifies engagement drivers using OLS Regression, and
delivers insights through a live Streamlit dashboard.

**Problem:** E-learning platforms collect enormous behavioral data but rarely
act on it. Students at risk of failing are identified too late. This engine
statistically proves what drives academic performance.

## Team

| Name | USN | Role |
|------|-----|------|
| Mallya Moni | 1DS24CS114 | System Architect, ML Lead |
| Nandana Raghunath | 1DS24CS127 | Data Engineer, EDA Lead |
| Pari Tirthwani | 1DS24CS137 | Visualization, Dashboard Lead |

## Analytical Pipeline  
[Data Ingestion] → [Feature Engineering] → [Statistical Inference] → [Dashboard]
Faker + SQLite     Pandas + NumPy          SciPy + Statsmodels      Streamlit

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Faker | Synthetic dataset generation |
| SQLite | Relational data storage |
| Pandas, NumPy | Data processing and feature engineering |
| SciPy, Statsmodels | T-tests, ANOVA, OLS Regression |
| Scikit-learn | At-risk classification model |
| Plotly, Seaborn | Interactive and static visualizations |
| Streamlit | Live web dashboard |

## Project Structure
EduPulse/
├── database/          # Schema and SQLite DB
├── data_gen/          # Synthetic data generation
├── processing/        # Feature engineering + normalization
├── inference/         # Hypothesis tests + regression models
├── visualizations/    # Static and interactive charts
├── dashboard/         # Streamlit web app
└── notebooks/         # EDA and analysis notebooks

## Institution
Dayananda Sagar College of Engineering (DSCE)
Department of Computer Science & Engineering
Affiliated to VTU, Belagavi | Approved by AICTE
Mini Project — 2024-25 Even Semester
