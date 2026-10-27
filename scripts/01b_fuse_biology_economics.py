"""
01b_fuse_biology_economics.py
=============================
Fuse the economic/LCA decarbonisation panel with:
  1. Biological indicators from STECF Balance Capacity (Annex V)
     - Original 4: SAR, SHI, EDI, overfished_stocks
     - New 7: Break-even revenue, Revenue/BER ratio, Vessel Utilisation,
              ROI, NVA/FTE, % inactive vessels, Live weight of landings
  2. Species diversity indices (Shannon H, CR3 concentration)

Output: trilemma_master_panel.csv
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# 1. Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'external')
UPSTREAM_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')

panel_file = os.path.join(PROJECT_DIR, 'data', 'decarbonisation_panel.csv')
balance_file = os.path.join(EXTERNAL_DATA_DIR, 'STECF 25-12 ANNEX V – Balance capacity – indicator table.xlsx')
diversity_file = os.path.join(UPSTREAM_DIR, 'species_diversity.csv')
output_file = os.path.join(PROJECT_DIR, 'data', 'trilemma_master_panel.csv')


def fuse_biology_and_economics():
    # ── 1. Load base economic panel ──────────────────────────────────
    print("Loading Decarbonisation Panel (Economics + LCA)...")
    if not os.path.exists(panel_file):
        raise FileNotFoundError(f"Missing base panel: {panel_file}")

    df_econ = pd.read_csv(panel_file)
    print(f"  Loaded {len(df_econ)} economic/LCA records ({len(df_econ.columns)} columns).")

    # ── 2. Load and pivot biological indicators ──────────────────────
    print("Loading STECF Balance Capacity (Biology)...")
    df_bio = pd.read_excel(balance_file, sheet_name='data')
    print(f"  Loaded {len(df_bio)} balance capacity records.")

    # Extended target variables (original 4 + new 7)
    target_vars = [
        # Original
        'Sustainable Harvest Indicator',
        'Stocks At Risks',
        'Economic dependency indicator',
        'Number of overfished stocks',
        # NEW — economic resilience
        'Break-even revenue',
        'Current revenue to Break-even revenue',
        # NEW — capacity
        'Vessel Utilisation Ratio',
        '% of inactive vessels',
        # NEW — productivity
        'Return on Investment',
        'Net Value Added per FTE',
        # NEW — physical output
        'Live weight of landings',
    ]

    df_bio_filtered = df_bio[df_bio['variable_name'].isin(target_vars)]

    merge_keys = ['country_name', 'year', 'fishing_tech', 'vessel_length']
    df_econ['year'] = df_econ['year'].astype(int)
    df_bio_filtered = df_bio_filtered.dropna(subset=merge_keys)
    df_bio_filtered['year'] = df_bio_filtered['year'].astype(int)

    # Pivot the biological data
    print("  Pivoting biological indicators...")
    df_bio_pivot = df_bio_filtered.pivot_table(
        index=merge_keys,
        columns='variable_name',
        values='value',
        aggfunc='mean'
    ).reset_index()

    print(f"  Pivoted bio data: {len(df_bio_pivot)} unique fleet segments.")

    # ── 3. Merge economics + biology ─────────────────────────────────
    print("Fusing datasets...")
    df_fused = pd.merge(df_econ, df_bio_pivot, on=merge_keys, how='left')

    # Clean up column names
    df_fused.rename(columns={
        'Sustainable Harvest Indicator': 'SHI',
        'Stocks At Risks': 'SAR',
        'Economic dependency indicator': 'EDI',
        'Number of overfished stocks': 'overfished_stocks',
        'Break-even revenue': 'break_even_revenue',
        'Current revenue to Break-even revenue': 'revenue_ber_ratio',
        'Vessel Utilisation Ratio': 'vessel_utilisation',
        '% of inactive vessels': 'pct_inactive_vessels',
        'Return on Investment': 'roi',
        'Net Value Added per FTE': 'nva_per_fte',
        'Live weight of landings': 'live_weight_landings',
    }, inplace=True)

    # ── 4. Impute missing biological data ────────────────────────────
    print("  Filling missing biological indicators using group medians...")
    bio_cols = ['SHI', 'SAR', 'EDI', 'overfished_stocks',
                'break_even_revenue', 'revenue_ber_ratio',
                'vessel_utilisation', 'pct_inactive_vessels',
                'roi', 'nva_per_fte', 'live_weight_landings']

    for col in bio_cols:
        if col in df_fused.columns:
            # Fill with median for same country × fishing tech
            df_fused[col] = df_fused.groupby(['country_name', 'fishing_tech'])[col].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing, fill with overall median
            df_fused[col] = df_fused[col].fillna(df_fused[col].median())

    # ── 5. Derive break-even carbon price ────────────────────────────
    # Maximum carbon price a segment can tolerate before EBIT < 0
    if 'ebit' in df_fused.columns and 'co2_emissions_tonnes' in df_fused.columns:
        df_fused['break_even_carbon_price'] = (
            df_fused['ebit'] / df_fused['co2_emissions_tonnes'].replace(0, np.nan)
        )

    # ── 6. Merge species diversity indices ───────────────────────────
    if os.path.exists(diversity_file):
        print("Loading species diversity indices...")
        df_div = pd.read_csv(diversity_file)
        # Select key diversity metrics
        div_cols = ['country_name', 'year', 'shannon_h', 'cr3_weight',
                    'n_species', 'simpson_lambda']
        available_div = [c for c in div_cols if c in df_div.columns]
        df_div_select = df_div[available_div].copy()
        df_div_select['year'] = df_div_select['year'].astype(int)

        # Country × Year merge
        df_fused = pd.merge(df_fused, df_div_select,
                            on=['country_name', 'year'], how='left')
        print(f"  Merged species diversity: {df_div_select.shape[0]} country-year obs.")

        # Fill missing diversity with country medians
        for col in ['shannon_h', 'cr3_weight', 'n_species', 'simpson_lambda']:
            if col in df_fused.columns:
                df_fused[col] = df_fused.groupby('country_name')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df_fused[col] = df_fused[col].fillna(df_fused[col].median())
    else:
        print("  Species diversity file not found — skipping.")

    # ── 7. Save output ───────────────────────────────────────────────
    df_fused.to_csv(output_file, index=False)

    # Write to consolidated Input workbook
    xlsx_path = os.path.join(PROJECT_DIR, 'Input_Data_Consolidated.xlsx')
    if os.path.exists(xlsx_path):
        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            df_fused.to_excel(writer, sheet_name='Master_Panel', index=False)
    else:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df_fused.to_excel(writer, sheet_name='Master_Panel', index=False)

    print(f"\nFusion complete! Master dataset saved to: {output_file}")
    print(f"  + Written to {xlsx_path} (sheet: Master_Panel)")
    print(f"  Final shape: {df_fused.shape}")
    print(f"  Columns ({len(df_fused.columns)}):")
    for col in sorted(df_fused.columns):
        n_valid = df_fused[col].notna().sum()
        pct = n_valid / len(df_fused) * 100
        print(f"    {col:35s} {n_valid:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    fuse_biology_and_economics()
