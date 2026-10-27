"""
01_prepare_data.py
==================
Prepares the master longitudinal panel for the European commercial fishing fleet study:
  1. Real price deflation to constant 2015 euros using national GDP deflators.
  2. Physical carbon intensity (kg CO2 per kg live weight landings) and real monetary carbon intensity.
  3. Non-fuel operating margin construction to separate energy expenditure from operating profitability.
  4. Fleet segment longitudinal identifier harmonisation (country x fishing technology x vessel length).
  5. Minimum revenue threshold filtering.
"""

import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
MASTER_CSV = os.path.join(DATA_DIR, "trilemma_master_panel.csv")
DEFLATOR_CSV = os.path.join(EXTERNAL_DIR, "gdp_deflators_2015.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "trilemma_master_panel.csv")


def prepare_master_panel():
    print("=== Phase 1: Preparing Master Decarbonisation Panel ===")

    # 1. Load existing fused master panel
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded master panel: {len(df)} rows, {len(df.columns)} columns.")

    # 2. Load and normalize GDP deflators to 2015 = 100
    df_def = pd.read_csv(DEFLATOR_CSV)
    # Rebase each country to 2015 = 100
    base_2015 = df_def[df_def['year'] == 2015].set_index('country_name')['gdp_deflator_2015'].to_dict()
    df_def['deflator_rebased_2015'] = df_def.apply(
        lambda r: (r['gdp_deflator_2015'] / base_2015.get(r['country_name'], 100.0)) * 100.0, axis=1
    )

    # Merge deflator onto master panel
    df = pd.merge(df, df_def[['country_name', 'year', 'deflator_rebased_2015']],
                  on=['country_name', 'year'], how='left')
    # Default to 100.0 if any missing
    df['deflator_rebased_2015'] = df['deflator_rebased_2015'].fillna(100.0)

    # 3. Compute Real Deflated Revenue (2015 constant euros)
    df['real_revenue'] = df['revenue'] / (df['deflator_rebased_2015'] / 100.0)
    df['real_ebit'] = df['ebit'] / (df['deflator_rebased_2015'] / 100.0)
    df['real_energy_costs'] = df['energy_costs'] / (df['deflator_rebased_2015'] / 100.0)

    # 4. Compute Monetary Carbon Intensity (kg CO2 per real EUR revenue)
    # Direct combustion emissions factor = 2.68 kg CO2 / L diesel
    df['co2_kg'] = df['energy_consumption'] * 2.68
    df['carbon_intensity_mon_real'] = np.where(
        df['real_revenue'] > 0,
        df['co2_kg'] / df['real_revenue'],
        np.nan
    )
    # Monetary CI based on nominal revenue (for comparison)
    df['carbon_intensity_mon_nom'] = np.where(
        df['revenue'] > 0,
        df['co2_kg'] / df['revenue'],
        np.nan
    )

    # 5. Compute Physical Carbon Intensity (kg CO2 per kg live weight landings)
    lw = df['live_weight_landings'].dropna()
    print(f"Live weight landings summary: median={lw.median():.1f}, min={lw.min():.1f}, max={lw.max():.1f}")
    
    # live_weight_landings in STECF Annex V is in kilograms (e.g. 19.4 million kg = 19,400 tonnes)
    df['carbon_intensity_phys'] = np.where(
        (df['live_weight_landings'] > 0) & (df['co2_kg'] > 0),
        df['co2_kg'] / df['live_weight_landings'],
        np.nan
    )

    # 6. Compute Non-Fuel Operating Margin (breaking mechanical fuel deduction in EBIT)
    df['ebit_nonfuel'] = df['ebit'] + df['energy_costs']
    df['margin_nonfuel'] = np.where(
        df['revenue'] > 0,
        (df['ebit_nonfuel'] / df['revenue']) * 100.0,
        np.nan
    )

    # 7. Create Segment Entity Identifier (country_name + fishing_tech + vessel_length)
    df['segment_id'] = df['country_name'] + '_' + df['fishing_tech'] + '_' + df['vessel_length']
    n_segments = df['segment_id'].nunique()
    print(f"Created segment_id: {n_segments} unique fleet segments across {df['country_name'].nunique()} countries.")

    # 8. Outlier and Reporting Anomaly Flags
    df['is_low_revenue_outlier'] = (df['revenue'] < 1000.0) | (df['revenue'].isna())
    df['is_valid_analysis_segment'] = (~df['is_low_revenue_outlier']) & (df['energy_consumption'] > 0)

    # 9. Winsorized continuous metrics for robust econometric estimation (1st and 99th percentiles)
    for col in ['carbon_intensity_mon_real', 'carbon_intensity_mon_nom', 'carbon_intensity_phys',
                'ebit_margin', 'margin_nonfuel', 'SAR']:
        if col in df.columns:
            valid_mask = df[col].notna() & np.isfinite(df[col])
            p01 = df.loc[valid_mask, col].quantile(0.01)
            p99 = df.loc[valid_mask, col].quantile(0.99)
            df[f'{col}_w'] = df[col].clip(p01, p99)
            print(f"  {col}_w clipped to [{p01:.3f}, {p99:.3f}]: mean={df[f'{col}_w'].mean():.3f}")

    # 10. Save revised master panel
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved revised panel to {OUTPUT_CSV}: {len(df)} rows, {len(df.columns)} columns.")

    return df


if __name__ == '__main__':
    prepare_master_panel()
