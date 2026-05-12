import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(".")
from processing.features import load_and_engineer


def run_all_statistics():
    """
    The core of EduPulse — runs all statistical tests
    and saves results to inference/stats_results.txt
    """
    
    df, sessions, modules = load_and_engineer()
    results = []
    results.append("EduPulse — Statistical Analysis Report")
    results.append("=" * 50)
    
    # ── TEST 1: T-Test ───────────────────────────────────────
    # Question: Do students who complete modules score higher?
    # We split students into two groups and compare quiz scores
    
    completed   = sessions[sessions["completed"] == 1]["quiz_score"]
    incomplete  = sessions[sessions["completed"] == 0]["quiz_score"]
    
    t_stat, p_value = stats.ttest_ind(completed, incomplete)
    
    results.append("\nTEST 1 — Independent T-Test")
    results.append("Question: Do students who complete modules score higher on quizzes?")
    results.append(f"   Completed   group mean : {completed.mean():.2f}")
    results.append(f"   Incomplete  group mean : {incomplete.mean():.2f}")
    results.append(f"   T-statistic            : {t_stat:.4f}")
    results.append(f"   P-value                : {p_value:.6f}")
    if p_value < 0.05:
        results.append("   Result: SIGNIFICANT — completion strongly predicts quiz score (p < 0.05)")
    else:
        results.append("   Result: Not significant at 0.05 level")
    
    # ── TEST 2: Pearson Correlation ──────────────────────────
    # Question: Does study duration correlate with quiz score?
    
    corr, p_corr = stats.pearsonr(
        sessions["duration_minutes"], sessions["quiz_score"]
    )
    
    results.append("\nTEST 2 — Pearson Correlation")
    results.append("Question: Does studying longer lead to higher quiz scores?")
    results.append(f"   Correlation coefficient : {corr:.4f}")
    results.append(f"   P-value                 : {p_corr:.6f}")
    if abs(corr) > 0.3:
        strength = "strong" if abs(corr) > 0.5 else "moderate"
        results.append(f"   Result:  {strength.upper()} correlation found")
    else:
        results.append("   Result: Weak correlation")
    
    # ── TEST 3: ANOVA ────────────────────────────────────────
    # Question: Do quiz scores differ significantly across device types?
    # ANOVA tests if 3+ group means are different
    
    mobile  = sessions[sessions["device_type"]=="Mobile" ]["quiz_score"]
    desktop = sessions[sessions["device_type"]=="Desktop"]["quiz_score"]
    tablet  = sessions[sessions["device_type"]=="Tablet" ]["quiz_score"]
    
    f_stat, p_anova = stats.f_oneway(mobile, desktop, tablet)
    
    results.append("\nTEST 3 — One-Way ANOVA")
    results.append("Question: Does quiz score differ by device type?")
    results.append(f"   Mobile  mean : {mobile.mean():.2f}")
    results.append(f"   Desktop mean : {desktop.mean():.2f}")
    results.append(f"   Tablet  mean : {tablet.mean():.2f}")
    results.append(f"   F-statistic  : {f_stat:.4f}")
    results.append(f"   P-value      : {p_anova:.6f}")
    if p_anova < 0.05:
        results.append("   Result:  SIGNIFICANT — device type affects performance")
    else:
        results.append("   Result: No significant difference across devices")
    
    # ── TEST 4: OLS Regression ───────────────────────────────
    # Question: Which engagement factors best predict quiz score?
    # OLS gives us a mathematical equation:
    # quiz_score = a + b1*duration + b2*videos + b3*completed + error
    
    X = sessions[["duration_minutes","videos_watched","completed"]].copy()
    y = sessions["quiz_score"]
    
    # Add constant (intercept) to the model
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()
    
    results.append("\nTEST 4 — OLS Multiple Regression")
    results.append("Question: What predicts quiz score best?")
    results.append(f"   R-squared  : {model.rsquared:.4f}  (model explains {model.rsquared*100:.1f}% of variance)")
    results.append(f"   F-statistic: {model.fvalue:.4f}  (p = {model.f_pvalue:.6f})")
    results.append("   Coefficients:")
    for name, coef, pval in zip(
        model.params.index, model.params.values, model.pvalues.values
    ):
        sig = " significant" if pval < 0.05 else "not significant"
        results.append(f"      {name:20} coef={coef:8.4f}  p={pval:.4f}  {sig}")
    
    # ── TEST 5: Engagement Score Correlation ─────────────────
    # Question: Does our engagement score actually predict quiz performance?
    
    corr_eng, p_eng = stats.pearsonr(
        df["engagement_score"], df["avg_quiz_score"]
    )
    
    results.append("\nTEST 5 — Engagement Score Validation")
    results.append("Question: Does our engagement score predict academic performance?")
    results.append(f"   Correlation : {corr_eng:.4f}")
    results.append(f"   P-value     : {p_eng:.6f}")
    if p_eng < 0.05:
        results.append("   Result:  Engagement score is a valid predictor")
    
    # ── Save results ─────────────────────────────────────────
    report = "\n".join(results)
    with open("inference/stats_results.txt", "w") as f:
        f.write(report)
    
    print(report)
    print("\n Results saved to inference/stats_results.txt")
    
    return df, model


if __name__ == "__main__":
    df, model = run_all_statistics()
