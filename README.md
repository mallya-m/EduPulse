# EduPulse — Statistical Intelligence Engine

> A professional educational analytics platform that transforms raw e-learning interaction logs into actionable insights using statistical inference, predictive modeling, and interactive visualization.

---

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square">

<img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square">

<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square">

<img src="https://img.shields.io/badge/SQLite-Database-lightgrey?style=flat-square">

<img src="https://img.shields.io/badge/Status-Active-success?style=flat-square">

</p>

---

## Overview

Modern e-learning platforms generate massive amounts of behavioral data such as:

* Login frequency
* Watch time
* Quiz attempts
* Assignment submissions
* Session duration
* Attendance activity

However, most institutions struggle to convert this raw data into meaningful academic insights.

EduPulse bridges this gap through:

* Statistical Analysis
* Feature Engineering
* Predictive Modeling
* Interactive Dashboards
* Educational Data Intelligence

The platform helps identify:

* Student engagement patterns
* Performance trends
* At-risk students
* Academic correlations
* Learning behavior insights

---

## Problem Statement

Traditional educational dashboards only display raw activity metrics without explaining:

* What actually influences academic performance
* Which students are at academic risk
* How engagement affects outcomes
* What intervention strategies may help

EduPulse applies statistical inference and machine learning techniques to transform educational interaction logs into actionable intelligence for educators and institutions.

---

## Core Features

<table>
<tr>
<td width="50%">

### Synthetic Educational Dataset

Generate realistic student datasets using the Faker library.

Includes:

* Student demographics
* Login activity
* Quiz scores
* Watch time
* Attendance
* Assignment completion
* Final grades

</td>

<td width="50%">

### Engagement Score Engine

Custom feature-engineered engagement score using:

* Login frequency
* Session duration
* Quiz participation
* Assignment completion

Used to quantify learning engagement levels.

</td>
</tr>

<tr>
<td width="50%">

### Statistical Inference

Implemented statistical methods:

* Pearson Correlation
* T-Tests
* ANOVA
* OLS Regression

Hypothesis:

> Higher engagement improves academic performance.

</td>

<td width="50%">

### Predictive Modeling

Machine learning models used to:

* Detect at-risk students
* Predict academic trends
* Analyze engagement patterns

Algorithms:

* Linear Regression
* Logistic Regression

</td>
</tr>
</table>

---

## Interactive Dashboard

The Streamlit dashboard provides:

| Feature             | Description                |
| ------------------- | -------------------------- |
| KPI Cards           | Quick academic insights    |
| Filters             | Student/course filtering   |
| Heatmaps            | Correlation analysis       |
| Distribution Graphs | Performance distribution   |
| Interactive Charts  | Plotly visualizations      |
| Risk Analysis       | At-risk student prediction |
| Trend Analysis      | Engagement tracking        |

---

## Tech Stack

| Category                  | Technologies                   |
| ------------------------- | ------------------------------ |
| Programming Language      | Python 3.11                    |
| Data Processing           | Pandas, NumPy                  |
| Statistical Analysis      | SciPy, Statsmodels             |
| Machine Learning          | Scikit-learn                   |
| Visualization             | Plotly, Seaborn, Matplotlib    |
| Dashboard Framework       | Streamlit                      |
| Database                  | SQLite                         |
| Synthetic Data Generation | Faker                          |
| Environment               | Jupyter Notebook, Google Colab |
| Version Control           | GitHub                         |

---

## System Architecture

```text
Synthetic Data Generation (Faker)
                ↓
        SQLite Database
                ↓
      Data Processing Layer
                ↓
       Feature Engineering
                ↓
      Statistical Inference
                ↓
      Predictive Modeling
                ↓
      Data Visualization
                ↓
     Streamlit Dashboard
```

---

## Project Structure

```bash
EduPulse/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── database/
│   ├── schema.py
│   └── eduPulse.db
│
├── data_gen/
│   ├── generate_data.py
│   └── load_data.py
│
├── processing/
│   ├── features.py
│   └── normalize.py
│
├── inference/
│   ├── hypothesis_tests.py
│   ├── ols_regression.py
│   └── risk_model.py
│
├── visualizations/
│   ├── static_plots.py
│   └── interactive.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Analysis.ipynb
│
└── docs/
    ├── report/
    ├── presentation/
    └── screenshots/
```

---

## Team Contributions

| Team Member       | Responsibilities                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| Mallya Moni       | System Architecture, OLS Regression, ML Integration, GitHub Management, Final Dashboard Integration |
| Nandana Raghunath | Faker Dataset Generation, SQLite Database, Data Cleaning, Feature Engineering                       |
| Pari Tirthwani    | Streamlit UI, Plotly Visualizations, Dashboard Styling, Documentation                               |

---

## Statistical Techniques Used

| Technique           | Purpose                                    |
| ------------------- | ------------------------------------------ |
| Pearson Correlation | Relationship between engagement and grades |
| T-Test              | Compare student performance groups         |
| ANOVA               | Multi-group comparison                     |
| OLS Regression      | Quantify engagement impact                 |
| Logistic Regression | Predict at-risk students                   |

---

## Example Insights Generated

The system can identify insights such as:

* Students with consistent login activity tend to score higher academically.
* Assignment completion strongly correlates with performance.
* Low watch time increases risk probability.
* Higher engagement score positively affects final grades.

---

## Quick Start

### Clone Repository

```bash
git clone https://github.com/mallya-m/eduinsight-learning-analytics.git
cd eduinsight-learning-analytics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Synthetic Dataset

```bash
python data_gen/generate_data.py
```

### Load Data into SQLite

```bash
python data_gen/load_data.py
```

### Run Feature Engineering

```bash
python processing/features.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Development Timeline

| Phase   | Tasks                                                | Progress |
| ------- | ---------------------------------------------------- | -------- |
| Phase 1 | Repository Setup, SQLite, Faker Dataset, Initial EDA | 45%      |
| Phase 2 | Statistical Analysis, Regression, ML Models          | 65%      |
| Phase 3 | Dashboard Development, Plotly Integration            | 90%      |
| Phase 4 | Documentation, PPT, Viva Preparation                 | 100%     |

---

## Academic Relevance

This project demonstrates practical applications of:

* Statistical Data Science
* Educational Data Mining
* Predictive Analytics
* Machine Learning
* Human Behavioral Analysis
* Data Visualization

---

## Future Scope

Potential future enhancements:

* Real LMS integration
* Real-time analytics
* Personalized learning recommendations
* AI-based academic intervention systems
* Student performance forecasting

---

## Institution

Dayananda Sagar College of Engineering (DSCE)
Department of Computer Science & Engineering
Affiliated to VTU, Belagavi

---

## License

This project is developed for academic and educational purposes.

---

## Team EduPulse

Mallya Moni
Nandana Raghunath
Pari Tirthwani
