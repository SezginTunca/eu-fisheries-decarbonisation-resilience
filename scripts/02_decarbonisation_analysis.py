"""
02_decarbonisation_analysis.py
==============================
Econometric, clustering, carbon pricing simulation, and Pareto frontier analysis script.

Modules:
  1. Two-way fixed effects panel regressions with segment-clustered standard errors.
  2. Physical and real monetary carbon intensity models and non-fuel operating margin specifications.
  3. Lagged explanatory specifications (t-1) for temporal precedence assessment.
  4. Method of Moments Quantile Regression (MM-QR) across deciles (tau = 0.10 to 0.90).
  5. Ward hierarchical clustering diagnostics (Silhouette, Calinski-Harabasz, Davies-Bouldin, bootstrap Jaccard).
  6. Carbon tax and revenue-neutral rebate micro-simulations across allocation rules.
  7. Non-linear Pareto frontier estimation with bootstrap confidence intervals.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import curve_fit
import scipy.stats as stats

# Set random seed for exact reproducibility
np.random.seed(42)

# Directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "trilemma_master_panel.csv")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")
os.makedirs(TABLE_DIR, exist_ok=True)


def load_and_prepare_panel():
    """Load and index panel for linearmodels and statsmodels."""
    print("Loading revised master panel...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter out invalid reporting outliers for primary regressions
    # (revenue < 1,000 EUR or non-positive energy)
    df_reg = df[df['is_valid_analysis_segment']].copy()
    
    # Format DatetimeIndex for linearmodels PanelOLS
    df_reg['year_dt'] = pd.to_datetime(df_reg['year'].astype(str) + '-01-01')
    
    # Create segment-level lag of carbon intensity and SAR
    df_reg = df_reg.sort_values(['segment_id', 'year'])
    df_reg['carbon_intensity_mon_real_w_lag1'] = df_reg.groupby('segment_id')['carbon_intensity_mon_real_w'].shift(1)
    df_reg['carbon_intensity_phys_w_lag1'] = df_reg.groupby('segment_id')['carbon_intensity_phys_w'].shift(1)
    df_reg['SAR_w_lag1'] = df_reg.groupby('segment_id')['SAR_w'].shift(1)
    
    # Compute interaction term for Model 5
    if 'shannon_h' in df_reg.columns:
        df_reg['shannon_h_clean'] = df_reg['shannon_h'].fillna(df_reg['shannon_h'].median())
        df_reg['ci_mon_x_shannon'] = df_reg['carbon_intensity_mon_real_w'] * df_reg['shannon_h_clean']
        df_reg['ci_phys_x_shannon'] = df_reg['carbon_intensity_phys_w'] * df_reg['shannon_h_clean']
    
    # Log engine power and activity ratio for Model 4
    df_reg['ln_kw'] = np.log1p(df_reg['total_kw'].fillna(df_reg['total_kw'].median()))
    df_reg['activity_ratio'] = df_reg['vessel_utilisation'].fillna(0.70)
    df_reg['mean_age_clean'] = df_reg['mean_age'].fillna(df_reg['mean_age'].median())
    
    # Subsidy intensity winsorized
    if 'subsidy_intensity' in df_reg.columns:
        sub_p99 = df_reg['subsidy_intensity'].quantile(0.99)
        df_reg['subsidy_intensity_w'] = df_reg['subsidy_intensity'].clip(0, sub_p99).fillna(0)
    else:
        df_reg['subsidy_intensity_w'] = 0.0

    print(f"Prepared panel: {len(df_reg)} valid observations across {df_reg['segment_id'].nunique()} segments.")
    return df, df_reg


# =====================================================================
# 1. Multi-Model Panel Econometrics (Models 1–6)
# =====================================================================
def run_panel_econometrics(df_reg):
    print("\n" + "="*80)
    print("PHASE 2.1: ESTIMATING MULTI-MODEL PANEL REGRESSIONS (WITHIN-SEGMENT FE)")
    print("="*80)
    
    # Set panel MultiIndex: [entity, time]
    panel_df = df_reg.set_index(['segment_id', 'year_dt'])
    
    results_list = []
    model_summaries = {}
    
    # ── Model 1: Determinants of Carbon Intensity (Physical CI) ──────
    # DV = Physical Carbon Intensity; Controls = SAR, EBIT Margin
    mod1_phys = PanelOLS(panel_df['carbon_intensity_phys_w'],
                         panel_df[['SAR_w', 'ebit_margin_w']],
                         entity_effects=True, time_effects=True)
    res1_phys = mod1_phys.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 1 (Physical CI)'] = res1_phys
    
    # ── Model 1b: Determinants of Carbon Intensity (Monetary Real CI) ─
    mod1_mon = PanelOLS(panel_df['carbon_intensity_mon_real_w'],
                        panel_df[['SAR_w', 'ebit_margin_w']],
                        entity_effects=True, time_effects=True)
    res1_mon = mod1_mon.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 1b (Monetary Real CI)'] = res1_mon

    # ── Model 2: Operating Margin Baseline (Physical CI) ─────────────
    # Breaking shared denominator: Physical CI has NO revenue in denominator!
    mod2_phys = PanelOLS(panel_df['ebit_margin_w'],
                         panel_df[['carbon_intensity_phys_w', 'SAR_w']],
                         entity_effects=True, time_effects=True)
    res2_phys = mod2_phys.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 2 (Physical CI on EBIT)'] = res2_phys

    # ── Model 2b: Operating Margin Baseline (Monetary Real CI) ────────
    mod2_mon = PanelOLS(panel_df['ebit_margin_w'],
                        panel_df[['carbon_intensity_mon_real_w', 'SAR_w']],
                        entity_effects=True, time_effects=True)
    res2_mon = mod2_mon.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 2b (Monetary CI on EBIT)'] = res2_mon

    # ── Model 2c: Non-Fuel Margin Robustness (Breaking fuel in EBIT) ──
    # DV = Non-Fuel Margin; Explanatory = Physical CI, SAR
    mod2_nonfuel = PanelOLS(panel_df['margin_nonfuel_w'],
                            panel_df[['carbon_intensity_phys_w', 'SAR_w']],
                            entity_effects=True, time_effects=True)
    res2_nonfuel = mod2_nonfuel.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 2c (Non-Fuel Margin on Phys CI)'] = res2_nonfuel

    # ── Model 3: Subsidy-Carbon Nexus ─────────────────────────────────
    mod3 = PanelOLS(panel_df['ebit_margin_w'],
                    panel_df[['carbon_intensity_mon_real_w', 'SAR_w', 'subsidy_intensity_w']],
                    entity_effects=True, time_effects=True)
    res3 = mod3.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 3 (Subsidy Nexus)'] = res3

    # ── Model 4: Fleet Technical & Modernity Characteristics ──────────
    # DV = Real Monetary CI; Regressors = Vessel Age, ln(kW), Activity Ratio
    mod4 = PanelOLS(panel_df['carbon_intensity_mon_real_w'],
                    panel_df[['mean_age_clean', 'ln_kw', 'activity_ratio']],
                    entity_effects=True, time_effects=True)
    res4 = mod4.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 4 (Fleet Technical)'] = res4

    # ── Model 5: Catch Diversification Moderation ─────────────────────
    mod5 = PanelOLS(panel_df['ebit_margin_w'],
                    panel_df[['carbon_intensity_mon_real_w', 'SAR_w', 'shannon_h_clean', 'ci_mon_x_shannon']],
                    entity_effects=True, time_effects=True)
    res5 = mod5.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 5 (Diversification Moderation)'] = res5

    # ── Model 6: Full Multivariate Joint Specification ───────────────
    mod6 = PanelOLS(panel_df['ebit_margin_w'],
                    panel_df[['carbon_intensity_mon_real_w', 'SAR_w', 'subsidy_intensity_w',
                              'shannon_h_clean', 'mean_age_clean', 'ln_kw']],
                    entity_effects=True, time_effects=True)
    res6 = mod6.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Model 6 (Full Multivariate)'] = res6

    # ── Robustness: Lagged Carbon Intensity & Lagged SAR (t-1) ────────
    panel_lag = panel_df.dropna(subset=['carbon_intensity_mon_real_w_lag1', 'SAR_w_lag1'])
    mod_lag = PanelOLS(panel_lag['ebit_margin_w'],
                       panel_lag[['carbon_intensity_mon_real_w_lag1', 'SAR_w_lag1']],
                       entity_effects=True, time_effects=True)
    res_lag = mod_lag.fit(cov_type='clustered', cluster_entity=True)
    model_summaries['Robustness (Lagged CI t-1)'] = res_lag

    # Compile structured results for Table 1 and Table S3
    for m_name, res in model_summaries.items():
        params = res.params
        std_err = res.std_errors
        pvals = res.pvalues
        ci = res.conf_int()
        
        for var in params.index:
            pval = pvals[var]
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            results_list.append({
                "Model": m_name,
                "Variable": var,
                "Coefficient": round(params[var], 6),
                "Std_Error": round(std_err[var], 6),
                "t_stat": round(res.tstats[var], 3),
                "p_value": round(pval, 4),
                "Significant": sig,
                "CI_Lower": round(ci.loc[var, 'lower'], 6),
                "CI_Upper": round(ci.loc[var, 'upper'], 6),
                "Within_R2": round(res.rsquared_within, 4),
                "Overall_R2": round(res.rsquared_overall, 4),
                "N_obs": int(res.nobs),
                "N_entities": int(res.entity_info['total'])
            })
            
    df_results = pd.DataFrame(results_list)
    out_table1 = os.path.join(TABLE_DIR, "Table1_Econometric_Results.csv")
    df_results.to_csv(out_table1, index=False)
    print(f"✅ Saved Table 1 Econometric Results to {out_table1}")

    # Write formatted text summary
    summary_txt = os.path.join(TABLE_DIR, "Econometric_Summary.txt")
    with open(summary_txt, 'w') as f:
        f.write("="*85 + "\n")
        f.write("REVISED ECONOMETRIC RESULTS: WITHIN-FLEET-SEGMENT FIXED EFFECTS PANEL ESTIMATES\n")
        f.write("="*85 + "\n\n")
        for m_name, res in model_summaries.items():
            f.write(f"\n--- {m_name} ---\n")
            f.write(f"N = {res.nobs}, Entities = {res.entity_info['total']}, Within-R² = {res.rsquared_within:.4f}\n")
            f.write(res.summary.as_text() + "\n\n")
    print(f"✅ Saved full econometric text summary to {summary_txt}")

    return model_summaries, df_results


# =====================================================================
# 2. Full 17-Quantile Panel Regression Path via Machado & Santos Silva (2019) MMQR
# =====================================================================
def run_quantile_regressions(df_reg):
    print("\n" + "="*80)
    print("PHASE 2.2: ESTIMATING 17-QUANTILE REGRESSION PATH VIA MACHADO & SANTOS SILVA (2019) MMQR")
    print("="*80)
    
    quantiles = np.round(np.linspace(0.10, 0.90, 17), 2)
    df_q = df_reg.dropna(subset=['ebit_margin_w', 'carbon_intensity_mon_real_w', 'SAR_w']).copy()
    panel_df = df_q.set_index(['segment_id', 'year_dt'])
    
    # 1. Location equation (within entity and time fixed effects)
    mod_loc = PanelOLS(panel_df['ebit_margin_w'],
                       panel_df[['carbon_intensity_mon_real_w', 'SAR_w']],
                       entity_effects=True, time_effects=True)
    res_loc = mod_loc.fit(cov_type='clustered', cluster_entity=True)
    beta_loc = res_loc.params
    resids = res_loc.resids
    
    # 2. Scale equation (within entity and time fixed effects on absolute residuals)
    panel_df['abs_resids'] = np.abs(resids)
    mod_scale = PanelOLS(panel_df['abs_resids'],
                         panel_df[['carbon_intensity_mon_real_w', 'SAR_w']],
                         entity_effects=True, time_effects=True)
    res_scale = mod_scale.fit(cov_type='clustered', cluster_entity=True)
    gamma_scale = res_scale.params
    
    # Standardized residual distribution
    fitted_scale = res_scale.fitted_values.squeeze().clip(lower=0.01)
    w = resids / fitted_scale
    q_w = np.quantile(w, quantiles)
    
    cov_loc = res_loc.cov
    cov_scale = res_scale.cov
    
    q_results = []
    for idx_q, q in enumerate(quantiles):
        qw = q_w[idx_q]
        ci_est = beta_loc['carbon_intensity_mon_real_w'] + gamma_scale['carbon_intensity_mon_real_w'] * qw
        sar_est = beta_loc['SAR_w'] + gamma_scale['SAR_w'] * qw
        
        ci_var = cov_loc.loc['carbon_intensity_mon_real_w', 'carbon_intensity_mon_real_w'] + (qw**2) * cov_scale.loc['carbon_intensity_mon_real_w', 'carbon_intensity_mon_real_w']
        sar_var = cov_loc.loc['SAR_w', 'SAR_w'] + (qw**2) * cov_scale.loc['SAR_w', 'SAR_w']
        
        ci_se = np.sqrt(max(ci_var, 1e-4))
        sar_se = np.sqrt(max(sar_var, 1e-4))
        
        ci_t = ci_est / ci_se
        sar_t = sar_est / sar_se
        
        ci_pval = 2 * (1 - stats.norm.cdf(abs(ci_t)))
        sar_pval = 2 * (1 - stats.norm.cdf(abs(sar_t)))
        
        q_results.append({
            'Quantile': q,
            'CI_Coefficient': round(ci_est, 4),
            'CI_Std_Error': round(ci_se, 4),
            'CI_t_stat': round(ci_t, 3),
            'CI_p_value': round(ci_pval, 4),
            'CI_Lower_95': round(ci_est - 1.96 * ci_se, 4),
            'CI_Upper_95': round(ci_est + 1.96 * ci_se, 4),
            'SAR_Coefficient': round(sar_est, 4),
            'SAR_Std_Error': round(sar_se, 4),
            'SAR_p_value': round(sar_pval, 4),
            'N': int(len(df_q))
        })
        
    df_q_out = pd.DataFrame(q_results)
    out_q = os.path.join(TABLE_DIR, "TableS5_Quantile_Estimates.csv")
    df_q_out.to_csv(out_q, index=False)
    print(f"✅ Saved Machado & Santos Silva (2019) MMQR estimates to {out_q}")
    print(f"  Quantile 0.10 CI coef: {df_q_out.loc[df_q_out['Quantile']==0.10, 'CI_Coefficient'].values[0]:.2f}")
    print(f"  Quantile 0.50 CI coef: {df_q_out.loc[df_q_out['Quantile']==0.50, 'CI_Coefficient'].values[0]:.2f}")
    print(f"  Quantile 0.90 CI coef: {df_q_out.loc[df_q_out['Quantile']==0.90, 'CI_Coefficient'].values[0]:.2f}")
    
    return df_q_out



# =====================================================================
# 3. Cluster Validation & Stability Diagnostics (Ward's Hierarchical)
# =====================================================================
def run_cluster_diagnostics(df):
    print("\n" + "="*80)
    print("PHASE 2.3: CLUSTER DIAGNOSTICS & BOOTSTRAP RESAMPLING STABILITY")
    print("="*80)
    
    # Focus on 2023 fleet cross-section
    df_2023 = df[df['year'] == 2023].dropna(subset=['ebit_margin', 'carbon_intensity_mon_real', 'SAR']).copy()
    df_2023 = df_2023[df_2023['revenue'] > 1000].copy()
    
    # Aggregate to segment level if multiple records exist
    seg_2023 = df_2023.groupby('segment_id')[['ebit_margin', 'carbon_intensity_mon_real', 'SAR', 'revenue', 'fte', 'co2_kg', 'n_vessels']].mean().reset_index()
    print(f"2023 cross-section: {len(seg_2023)} fleet segments.")
    
    features = ['ebit_margin', 'carbon_intensity_mon_real', 'SAR']
    X = seg_2023[features].values
    # Standardize (Z-score)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Ward's linkage
    Z = linkage(X_std, method='ward')
    
    # 1. Multi-metric evaluation for k in [2, 6]
    cluster_metrics = []
    for k in range(2, 7):
        labels = fcluster(Z, k, criterion='maxclust')
        sil = silhouette_score(X_std, labels)
        ch = calinski_harabasz_score(X_std, labels)
        db = davies_bouldin_score(X_std, labels)
        
        cluster_metrics.append({
            'k': k,
            'Silhouette_Score': round(sil, 4),
            'Calinski_Harabasz': round(ch, 2),
            'Davies_Bouldin': round(db, 4)
        })
        print(f"  k={k}: Silhouette={sil:.4f}, Calinski-Harabasz={ch:.1f}, Davies-Bouldin={db:.4f}")
        
    df_diag = pd.DataFrame(cluster_metrics)
    out_diag = os.path.join(TABLE_DIR, "TableS6_Cluster_Diagnostics.csv")
    df_diag.to_csv(out_diag, index=False)
    print(f"✅ Saved cluster diagnostics to {out_diag}")
    
    # 2. Assign baseline 4-cluster partition (k=4)
    seg_2023['cluster'] = fcluster(Z, 4, criterion='maxclust')
    print("  Cluster sizes for k=4:", seg_2023['cluster'].value_counts().to_dict())
    
    # 3. Bootstrap Resampling Stability (1,000 iterations computing Jaccard similarity)
    print("  Running 1,000 bootstrap iterations for cluster stability...")
    n_samples = len(X_std)
    jaccard_scores = []
    
    base_labels = seg_2023['cluster'].values
    
    for _ in range(1000):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_std[boot_idx]
        Z_boot = linkage(X_boot, method='ward')
        boot_labels = fcluster(Z_boot, 4, criterion='maxclust')
        
        # Calculate pair-wise co-membership agreement (Rand / Jaccard proxy)
        pair_orig = (base_labels[boot_idx, None] == base_labels[boot_idx])
        pair_boot = (boot_labels[:, None] == boot_labels)
        
        intersection = np.logical_and(pair_orig, pair_boot).sum() - n_samples
        union = np.logical_or(pair_orig, pair_boot).sum() - n_samples
        jaccard = intersection / max(union, 1)
        jaccard_scores.append(jaccard)
        
    mean_jaccard = np.mean(jaccard_scores)
    ci_jaccard = (np.percentile(jaccard_scores, 2.5), np.percentile(jaccard_scores, 97.5))
    print(f"  Bootstrap Jaccard stability (k=4): mean={mean_jaccard:.3f}, 95% CI=[{ci_jaccard[0]:.3f}, {ci_jaccard[1]:.3f}]")
    
    # Save bootstrap stability report
    stability_df = pd.DataFrame({
        'Metric': ['Mean Jaccard Similarity', 'Lower 95% CI', 'Upper 95% CI', 'Iterations'],
        'Value': [round(mean_jaccard, 4), round(ci_jaccard[0], 4), round(ci_jaccard[1], 4), 1000]
    })
    stability_df.to_csv(os.path.join(TABLE_DIR, "TableS6b_Cluster_Stability_Jaccard.csv"), index=False)
    
    # Save 2023 clustered cross-section
    seg_2023.to_csv(os.path.join(PROJECT_DIR, "data", "cluster_segments_2023.csv"), index=False)
    
    return seg_2023, df_diag


# =====================================================================
# 4. Dynamic Behavioral Carbon Pricing & Feebate Simulation
# =====================================================================
def run_carbon_pricing_simulation(seg_2023):
    print("\n" + "="*80)
    print("PHASE 2.4: DYNAMIC BEHAVIORAL CARBON PRICING & FEEBATE SIMULATION")
    print("="*80)
    
    carbon_prices = [0, 50, 100, 150, 200, 250]
    sim_results = []
    
    tot_vessels = seg_2023['n_vessels'].sum()
    tot_fte = seg_2023['fte'].fillna(1).sum()
    
    for p_co2 in carbon_prices:
        co2_tonnes = seg_2023['co2_kg'] / 1000.0
        rev = seg_2023['revenue']
        ebit = (seg_2023['ebit_margin'] / 100.0) * rev
        
        # 1. Static Accounting (Zero Adaptation)
        tax_static = co2_tonnes * p_co2
        ebit_static = ebit - tax_static
        margin_static = np.where(rev > 0, (ebit_static / rev) * 100.0, np.nan)
        deficit_static = (margin_static < 0).mean() * 100.0
        vessel_deficit_static = ((margin_static < 0) * seg_2023['n_vessels']).sum() / tot_vessels * 100.0
        
        # 2. Dynamic Behavioral Adaptation
        delta_p_ratio = (p_co2 * 0.00268) / 0.85
        co2_dynamic = co2_tonnes * (1.0 - 0.30 * delta_p_ratio) * (1.0 - 0.03)
        tax_dynamic = co2_dynamic * p_co2
        
        fuel_tonnes_saved = (co2_tonnes - co2_dynamic) / 0.00268
        diesel_savings = fuel_tonnes_saved * 0.85
        
        rev_enhanced = rev + 0.15 * tax_dynamic
        ebit_dynamic = ebit - tax_dynamic + diesel_savings + 0.15 * tax_dynamic
        margin_dynamic = np.where(rev_enhanced > 0, (ebit_dynamic / rev_enhanced) * 100.0, np.nan)
        deficit_dynamic = (margin_dynamic < 0).mean() * 100.0
        vessel_deficit_dynamic = ((margin_dynamic < 0) * seg_2023['n_vessels']).sum() / tot_vessels * 100.0
        
        # 3. Feebate Recycling Rules (100% Revenue Neutral)
        tot_tax_dyn = tax_dynamic.sum()
        
        # Rule A: Labor-based recycling (proportional to FTE crew employment)
        rebate_fte = tot_tax_dyn * (seg_2023['fte'].fillna(1) / tot_fte)
        ebit_feebate_fte = ebit_dynamic + rebate_fte
        margin_feebate_fte = np.where(rev_enhanced > 0, (ebit_feebate_fte / rev_enhanced) * 100.0, np.nan)
        deficit_feebate_fte = (margin_feebate_fte < 0).mean() * 100.0
        vessel_deficit_feebate_fte = ((margin_feebate_fte < 0) * seg_2023['n_vessels']).sum() / tot_vessels * 100.0
        
        # Rule B: Vessel-equal recycling (weighted by segment vessel count)
        rebate_vessel = tot_tax_dyn * (seg_2023['n_vessels'] / tot_vessels)
        ebit_feebate_vessel = ebit_dynamic + rebate_vessel
        margin_feebate_vessel = np.where(rev_enhanced > 0, (ebit_feebate_vessel / rev_enhanced) * 100.0, np.nan)
        deficit_feebate_vessel = (margin_feebate_vessel < 0).mean() * 100.0
        vessel_deficit_feebate_vessel = ((margin_feebate_vessel < 0) * seg_2023['n_vessels']).sum() / tot_vessels * 100.0
        
        # Rule C: Green Transition Bonus (restricted to below-median carbon intensity)
        med_ci = seg_2023['carbon_intensity_mon_real'].median()
        is_green = (seg_2023['carbon_intensity_mon_real'] < med_ci).astype(float)
        inv_ci = is_green / (seg_2023['carbon_intensity_mon_real'] + 0.1)
        denom_green = inv_ci.sum()
        rebate_green = tot_tax_dyn * (inv_ci / denom_green) if denom_green > 0 else 0
        ebit_feebate_green = ebit_dynamic + rebate_green
        margin_feebate_green = np.where(rev_enhanced > 0, (ebit_feebate_green / rev_enhanced) * 100.0, np.nan)
        deficit_feebate_green = (margin_feebate_green < 0).mean() * 100.0
        vessel_deficit_feebate_green = ((margin_feebate_green < 0) * seg_2023['n_vessels']).sum() / tot_vessels * 100.0
        
        sim_results.append({
            'Carbon_Price_EUR': p_co2,
            'Median_EBIT_Static': round(float(np.nanmedian(margin_static)), 2),
            'Deficit_Pct_Static': round(float(deficit_static), 2),
            'Vessel_Deficit_Static': round(float(vessel_deficit_static), 2),
            'Median_EBIT_Dynamic': round(float(np.nanmedian(margin_dynamic)), 2),
            'Deficit_Pct_Dynamic': round(float(deficit_dynamic), 2),
            'Vessel_Deficit_Dynamic': round(float(vessel_deficit_dynamic), 2),
            'Deficit_Feebate_FTE': round(float(deficit_feebate_fte), 2),
            'Vessel_Deficit_Feebate_FTE': round(float(vessel_deficit_feebate_fte), 2),
            'Deficit_Feebate_Vessel': round(float(deficit_feebate_vessel), 2),
            'Vessel_Deficit_Feebate_Vessel': round(float(vessel_deficit_feebate_vessel), 2),
            'Deficit_Feebate_Green': round(float(deficit_feebate_green), 2),
            'Vessel_Deficit_Feebate_Green': round(float(vessel_deficit_feebate_green), 2)
        })
        print(f"  Price €{p_co2}/tCO2: Static Deficit={deficit_static:.1f}%, Dynamic={deficit_dynamic:.1f}%, Feebate(Vessel)={deficit_feebate_vessel:.1f}%")
        
    df_sim = pd.DataFrame(sim_results)
    out_sim = os.path.join(TABLE_DIR, "Table2_Simulation_Results.csv")
    df_sim.to_csv(out_sim, index=False)
    print(f"✅ Saved Table 2 Simulation Results to {out_sim}")
    return df_sim


# =====================================================================
# 5. Non-Linear Pareto Envelope Estimation (Non-Dominated Boundary)
# =====================================================================
def run_pareto_envelope_estimation(df):
    print("\n" + "="*80)
    print("PHASE 2.5: ESTIMATING NON-LINEAR PARETO ENVELOPE (EQUATION 11)")
    print("="*80)
    
    df_valid = df[df['is_valid_analysis_segment']].copy()
    df_valid['phys_eff'] = 1.0 / df_valid['carbon_intensity_phys'].replace(0, np.nan)
    df_valid['econ_eff'] = df_valid['real_revenue'] / df_valid['co2_kg'].replace(0, np.nan)
    
    gear_summary = df_valid.groupby('fishing_tech')[['phys_eff', 'econ_eff']].median().dropna().reset_index()
    gear_summary = gear_summary[(gear_summary['phys_eff'] > 0) & (gear_summary['econ_eff'] > 0)]
    
    # Identify non-dominated Pareto frontier technologies: PGO, MGO, PG, PS, TM
    gear_sorted = gear_summary.sort_values(by=['phys_eff', 'econ_eff'], ascending=[False, False])
    frontier_rows = []
    curr_max_y = -1.0
    for idx, row in gear_sorted.iterrows():
        if row['econ_eff'] > curr_max_y:
            frontier_rows.append(row)
            curr_max_y = row['econ_eff']
    df_frontier = pd.DataFrame(frontier_rows)
    print(f"  Identified {len(df_frontier)} non-dominated frontier technologies: {df_frontier['fishing_tech'].tolist()}")
    
    x_front = df_frontier['phys_eff'].values
    y_front = df_frontier['econ_eff'].values
    
    def pareto_hyperbolic(x, theta, gamma):
        return theta / (x ** gamma)
        
    try:
        popt, pcov = curve_fit(pareto_hyperbolic, x_front, y_front, p0=[2.5, 0.40], bounds=([0.1, 0.01], [50.0, 2.0]))
        theta_hat, gamma_hat = popt
        theta_se, gamma_se = np.sqrt(np.diag(pcov))
        t_gamma = gamma_hat / gamma_se
        p_gamma = 2 * (1 - stats.t.cdf(abs(t_gamma), df=len(x_front) - 2))
        print(f"  Non-Dominated Pareto Envelope: theta={theta_hat:.4f} (SE={theta_se:.4f}), gamma={gamma_hat:.4f} (SE={gamma_se:.4f}, t={t_gamma:.3f}, p={p_gamma:.4f})")
        
        # 1,000 bootstrap iterations for parameter confidence bands
        boot_thetas, boot_gammas = [], []
        n_pts = len(x_front)
        for _ in range(1000):
            idx = np.random.choice(n_pts, size=n_pts, replace=True)
            try:
                b_popt, _ = curve_fit(pareto_hyperbolic, x_front[idx], y_front[idx], p0=popt, maxfev=2000)
                boot_thetas.append(b_popt[0])
                boot_gammas.append(b_popt[1])
            except Exception:
                pass
                
        theta_ci = (np.percentile(boot_thetas, 2.5), np.percentile(boot_thetas, 97.5)) if boot_thetas else (theta_hat - 1.96*theta_se, theta_hat + 1.96*theta_se)
        gamma_ci = (np.percentile(boot_gammas, 2.5), np.percentile(boot_gammas, 97.5)) if boot_gammas else (gamma_hat - 1.96*gamma_se, gamma_hat + 1.96*gamma_se)
        
        pareto_df = pd.DataFrame({
            'Parameter': ['theta (scale)', 'gamma (curvature)'],
            'Estimate': [round(theta_hat, 4), round(gamma_hat, 4)],
            'Std_Error': [round(theta_se, 4), round(gamma_se, 4)],
            't_stat': [round(theta_hat/theta_se, 3), round(t_gamma, 3)],
            'p_value': [round(2*(1-stats.t.cdf(theta_hat/theta_se, df=len(x_front)-2)), 4), round(p_gamma, 4)],
            'CI_Lower_95': [round(theta_ci[0], 4), round(gamma_ci[0], 4)],
            'CI_Upper_95': [round(theta_ci[1], 4), round(gamma_ci[1], 4)]
        })
        out_pareto = os.path.join(TABLE_DIR, "TableS8_Pareto_Parameters.csv")
        pareto_df.to_csv(out_pareto, index=False)
        print(f"✅ Saved Pareto frontier parameters and bootstrap confidence intervals to {out_pareto}")
        
    except Exception as e:
        print(f"  Warning on Pareto curve fitting: {e}")
        pareto_df = pd.DataFrame()
        
    return gear_summary, pareto_df


# =====================================================================
# 6. Exact Observation-Weighted Summary Statistics (Table S1)
# =====================================================================
def generate_table_s1_summary_statistics(df):
    print("\n" + "="*80)
    print("PHASE 2.6: GENERATING EXACT OBSERVATION-WEIGHTED TABLE S1 SUMMARY STATISTICS")
    print("="*80)
    
    df_val = df[df['is_valid_analysis_segment']].copy()
    
    def categorize_gear(gt):
        if gt in ['DTS', 'TT']:
            return 'Active Demersal Trawlers (DTS)'
        elif gt == 'TBB':
            return 'Beam Trawlers (TBB)'
        elif gt in ['TM', 'OTM', 'PTM']:
            return 'Pelagic Trawls (TM)'
        elif gt in ['PS', 'PTS']:
            return 'Purse Seiners (PS)'
        elif gt in ['PG', 'DFN', 'GNS', 'GND', 'GTN']:
            return 'Passive Coastal Gears (PG/DFN)'
        elif gt in ['HOK', 'LHP', 'LHM', 'LLD', 'LLS']:
            return 'Hooks and Lines (HOK)'
        elif gt in ['DRB', 'HMD']:
            return 'Dredgers (DRB)'
        else:
            return 'Other / Mixed Gears (MGO/PMP)'
            
    df_val['gear_group'] = df_val['fishing_tech'].apply(categorize_gear)
    
    gear_order = [
        'Active Demersal Trawlers (DTS)',
        'Beam Trawlers (TBB)',
        'Pelagic Trawls (TM)',
        'Purse Seiners (PS)',
        'Passive Coastal Gears (PG/DFN)',
        'Hooks and Lines (HOK)',
        'Dredgers (DRB)',
        'Other / Mixed Gears (MGO/PMP)'
    ]
    
    rows = []
    for g in gear_order:
        sub = df_val[df_val['gear_group'] == g]
        n_obs = len(sub)
        rows.append({
            'Fishing Technology': g,
            'Observations (N)': n_obs,
            'Physical CI (kg CO2/kg)': f"{sub['carbon_intensity_phys_w'].mean():.2f} ({sub['carbon_intensity_phys_w'].std():.2f})",
            'Real Monetary CI (kg CO2/EUR)': f"{sub['carbon_intensity_mon_real_w'].mean():.2f} ({sub['carbon_intensity_mon_real_w'].std():.2f})",
            'Operating Margin (EBIT %)': f"{sub['ebit_margin_w'].mean():.1f} ({sub['ebit_margin_w'].std():.1f})",
            'Non-Fuel Margin (%)': f"{sub['margin_nonfuel_w'].mean():.1f} ({sub['margin_nonfuel_w'].std():.1f})",
            'Stocks-at-Risk (SAR)': f"{sub['SAR_w'].mean():.2f} ({sub['SAR_w'].std():.2f})",
            'Fuel Cost Share (% OPEX)': f"{sub['energy_cost_share'].clip(0, 100).mean():.1f}%"
        })
        
    n_tot = len(df_val)
    rows.append({
        'Fishing Technology': 'Total Master Panel',
        'Observations (N)': n_tot,
        'Physical CI (kg CO2/kg)': f"{df_val['carbon_intensity_phys_w'].mean():.2f} ({df_val['carbon_intensity_phys_w'].std():.2f})",
        'Real Monetary CI (kg CO2/EUR)': f"{df_val['carbon_intensity_mon_real_w'].mean():.2f} ({df_val['carbon_intensity_mon_real_w'].std():.2f})",
        'Operating Margin (EBIT %)': f"{df_val['ebit_margin_w'].mean():.1f} ({df_val['ebit_margin_w'].std():.1f})",
        'Non-Fuel Margin (%)': f"{df_val['margin_nonfuel_w'].mean():.1f} ({df_val['margin_nonfuel_w'].std():.1f})",
        'Stocks-at-Risk (SAR)': f"{df_val['SAR_w'].mean():.2f} ({df_val['SAR_w'].std():.2f})",
        'Fuel Cost Share (% OPEX)': f"{df_val['energy_cost_share'].clip(0, 100).mean():.1f}%"
    })
    
    df_s1 = pd.DataFrame(rows)
    out_s1 = os.path.join(TABLE_DIR, "TableS1_Summary_Statistics.csv")
    df_s1.to_csv(out_s1, index=False)
    print(f"✅ Saved mathematically consistent Table S1 to {out_s1}")
    return df_s1


# =====================================================================
# Main Execution Pipeline
# =====================================================================
def main():
    print("="*85)
    print("STARTING ECONOMETRIC & SIMULATION PIPELINE")
    print("="*85)
    
    df, df_reg = load_and_prepare_panel()
    model_summaries, df_results = run_panel_econometrics(df_reg)
    df_q = run_quantile_regressions(df_reg)
    seg_2023, df_diag = run_cluster_diagnostics(df)
    df_sim = run_carbon_pricing_simulation(seg_2023)
    gear_summary, pareto_df = run_pareto_envelope_estimation(df)
    df_s1 = generate_table_s1_summary_statistics(df)
    
    print("\n" + "="*85)
    print("✅ PHASE 2 COMPLETE: ALL MODELS, DIAGNOSTICS, AND SIMULATIONS EXECUTED.")
    print("="*85)


if __name__ == '__main__':
    main()
