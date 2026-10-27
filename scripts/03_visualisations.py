"""
03_visualisations.py
====================
Generates main-text figures (Figures 1-5) and supplementary figures (Figures S1-S6) for:
"Operational Decarbonisation Supports Enterprise Economic Resilience Across European Commercial Fishing Fleets"
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "trilemma_master_panel.csv")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11.0,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.2,
    "legend.title_fontsize": 9.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.format": "png",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.size": 4,
    "ytick.major.width": 0.8,
    "ytick.major.size": 4,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

# Okabe-Ito Color Palette
PAL = {
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'teal':       '#009E73',
    'orange':     '#E69F00',
    'sky':        '#56B4E9',
    'pink':       '#CC79A7',
    'yellow':     '#F0E442',
    'black':      '#222222',
    'grey':       '#777777',
    'light_grey': '#E0E0E0',
}

CLUSTER_COLORS = {
    1: PAL['teal'],        # Cluster 1: High Resilience / Low Carbon
    2: PAL['sky'],         # Cluster 2: Moderate Resilience / Moderate Carbon
    3: PAL['orange'],      # Cluster 3: Low Resilience / High Carbon
    4: PAL['vermillion'],  # Cluster 4: Vulnerable / Severe Deficit
}

CLUSTER_LABELS = {
    1: 'Cluster 1: High Resilience / Low Carbon',
    2: 'Cluster 2: Moderate Resilience / Moderate Carbon',
    3: 'Cluster 3: Low Resilience / High Carbon',
    4: 'Cluster 4: Vulnerable / Severe Deficit',
}

def save_fig(fig, name):
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15)
    
    # Figures saved to figures/
    
    plt.close(fig)
    print(f"  ✓ Saved: {name}.png (600 DPI) & {name}.pdf (figures/)")


def load_data():
    print("Loading master panel...")
    df = pd.read_csv(DATA_PATH)
    df = df[df['revenue'] > 0].copy()

    # Ensure cleaned / winsorized versions of variables
    df['revenue_real'] = df['real_revenue']
    df['non_fuel_margin_w'] = df['margin_nonfuel_w']
    df['mean_age_clean'] = df['mean_age'].clip(df['mean_age'].quantile(0.01), df['mean_age'].quantile(0.99))
    df['shannon_h_clean'] = df['shannon_h'].clip(df['shannon_h'].quantile(0.01), df['shannon_h'].quantile(0.99))
    df['subsidy_intensity_w'] = df['subsidy_intensity'].clip(df['subsidy_intensity'].quantile(0.01), df['subsidy_intensity'].quantile(0.99))
    df['kg_fish_per_kg_co2'] = df['live_weight_landings'] / (df['co2_kg'] + 1e-6)
    df['eur_rev_per_kg_co2'] = df['real_revenue'] / (df['co2_kg'] + 1e-6)

    # Categorize gears
    passive_gears = ['DFN', 'FPO', 'HOK', 'PG', 'PGO', 'PGP']
    demersal_trawls = ['DTS', 'TBB']
    pelagic_gears = ['PS', 'TM']
    
    def categorize_gear(gt):
        if gt in passive_gears:
            return 'Passive Coastal Gears'
        elif gt in demersal_trawls:
            return 'Active Demersal Trawlers'
        elif gt in pelagic_gears:
            return 'Pelagic Trawls and Seiners'
        else:
            return 'Dredges and Other Gears'

    df['gear_family'] = df['fishing_tech'].apply(categorize_gear)
    print(f"  Panel loaded: {len(df):,} observations ({df['year'].min()}–{df['year'].max()}).")
    return df


# ──────────────────────────────────────────────────────────────────
# FIGURE 1: Longitudinal Decarbonisation Dynamics & Decoupling
# ──────────────────────────────────────────────────────────────────
def generate_figure_1(df):
    print("\nGenerating Figure 1: Longitudinal Decarbonisation Dynamics (2x2 Symmetric)...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.2, 10.2),
                                                 gridspec_kw={'hspace': 0.34, 'wspace': 0.22})

    years = np.sort(df['year'].unique())

    # Panel (a): Macroeconomic Decoupling Index
    yearly = df.groupby('year').agg({
        'real_revenue': 'sum',
        'energy_consumption': 'sum',
        'co2_emissions_tonnes': 'sum',
    }).loc[years]

    idx_rev = (yearly['real_revenue'] / yearly['real_revenue'].loc[2008]) * 100
    idx_fuel = (yearly['energy_consumption'] / yearly['energy_consumption'].loc[2008]) * 100
    idx_co2 = (yearly['co2_emissions_tonnes'] / yearly['co2_emissions_tonnes'].loc[2008]) * 100

    ax1.axhline(100, color='#888888', linestyle=':', linewidth=0.8, zorder=1)
    ax1.plot(years, idx_rev, color=PAL['blue'], marker='o', markersize=4.5, linewidth=1.8,
             label='Real Gross Fleet Revenue (2015 Constant EUR)', zorder=3)
    ax1.plot(years, idx_fuel, color=PAL['orange'], marker='s', markersize=4.5, linewidth=1.8,
             linestyle='--', label='Marine Diesel Fuel Consumption (Litres)', zorder=3)
    ax1.plot(years, idx_co2, color=PAL['vermillion'], marker='^', markersize=4.5, linewidth=1.6,
             linestyle='-.', label='Carbon Dioxide Emissions (Scope 1 Tonnes)', zorder=3)

    milestones = [
        (2008, '2008 Crisis'),
        (2013, '2013 Reform'),
        (2019, '2019 Green Deal'),
        (2022, '2022 Fuel Shock')
    ]
    for yr, text in milestones:
        ax1.axvline(yr, color='#BBBBBB', linestyle='--', linewidth=0.7, zorder=2)
        y_pos = 136 if yr in [2013, 2022] else 128
        x_pos = 2008.45 if yr == 2008 else yr
        ax1.text(x_pos, y_pos, text, fontsize=8.8, ha='center', va='bottom',
                 bbox=dict(boxstyle='square,pad=0.22', facecolor='white', edgecolor='#CCCCCC', linewidth=0.5, alpha=0.92))

    ax1.set_xlabel('Calendar Year')
    ax1.set_ylabel('Decoupling Index (2008 = 100)')
    ax1.set_ylim(70, 152)
    ax1.set_xlim(2007.0, 2023.6)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax1.legend(loc='lower left', framealpha=0.92, fontsize=8.8)
    ax1.set_title('(a) Macroeconomic Decoupling Dynamics', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Gear groups for Panels (b) and (c)
    gear_groups = [
        ('Passive Coastal Gears', PAL['teal']),
        ('Pelagic Trawls and Seiners', PAL['blue']),
        ('Active Demersal Trawlers', PAL['vermillion']),
    ]

    # Panel (b): Physical Carbon Intensity Trend
    for grp_name, color in gear_groups:
        sub = df[df['gear_family'] == grp_name]
        stats_yr = sub.groupby('year')['carbon_intensity_phys_w'].agg(
            q25=lambda x: x.quantile(0.25),
            median='median',
            q75=lambda x: x.quantile(0.75)
        ).loc[years]

        ax2.plot(years, stats_yr['median'], color=color, linewidth=2.0, marker='o', markersize=4, label=grp_name)
        ax2.fill_between(years, stats_yr['q25'], stats_yr['q75'], color=color, alpha=0.18)

    ax2.set_xlabel('Calendar Year')
    ax2.set_ylabel('Physical Carbon Intensity\n(kg CO$_2$ per kg live weight landings)')
    ax2.set_ylim(0, 10.5)
    ax2.set_xlim(2007.0, 2023.6)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.8)
    ax2.set_title('(b) Physical Carbon Intensity by Gear Family', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (c): Real Monetary Carbon Intensity Trend
    for grp_name, color in gear_groups:
        sub = df[df['gear_family'] == grp_name]
        stats_yr = sub.groupby('year')['carbon_intensity_mon_real_w'].agg(
            q25=lambda x: x.quantile(0.25),
            median='median',
            q75=lambda x: x.quantile(0.75)
        ).loc[years]

        ax3.plot(years, stats_yr['median'], color=color, linewidth=2.0, marker='s', markersize=4, label=grp_name)
        ax3.fill_between(years, stats_yr['q25'], stats_yr['q75'], color=color, alpha=0.18)

    ax3.set_xlabel('Calendar Year')
    ax3.set_ylabel('Real Monetary Carbon Intensity\n(kg CO$_2$ per 2015 EUR real revenue)')
    ax3.set_ylim(0, 3.8)
    ax3.set_xlim(2007.0, 2023.6)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax3.legend(loc='upper right', framealpha=0.92, fontsize=8.8)
    ax3.set_title('(c) Real Monetary Carbon Intensity by Gear Family', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (d): Profitability Margins & Energy Cost Share
    fin_stats = df.groupby('year').agg({
        'ebit_margin_w': 'median',
        'non_fuel_margin_w': 'median',
        'energy_cost_share': 'median'
    }).loc[years]

    ec_pct = fin_stats['energy_cost_share'] * 100 if fin_stats['energy_cost_share'].max() <= 1.0 else fin_stats['energy_cost_share']

    ax4.axhline(0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
    ax4.plot(years, fin_stats['non_fuel_margin_w'], color=PAL['blue'], marker='^', markersize=4.5, linewidth=1.6,
             linestyle=':', label='Non-Fuel Operating Margin (Median %)')
    ax4.plot(years, fin_stats['ebit_margin_w'], color=PAL['teal'], marker='s', markersize=5, linewidth=1.8,
             label='Operating Profit Margin (Median %)')
    ax4.plot(years, ec_pct, color=PAL['orange'], marker='d', markersize=5, linewidth=1.8,
             linestyle='--', label='Energy Expenditure Share (% of Operating Costs)')

    ax4.set_xlabel('Calendar Year')
    ax4.set_ylabel('Enterprise Margin / Expenditure Share (%)')
    ax4.set_xlim(2007.0, 2023.6)
    ax4.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax4.set_ylim(-8, 30)
    ax4.legend(loc='upper right', framealpha=0.92, fontsize=8.8)
    ax4.set_title('(d) Operating Margins and Energy Cost Burden', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    save_fig(fig, "Fig1_Temporal_Trends")



# ──────────────────────────────────────────────────────────────────
# FIGURE 2: Multi-Model Econometric Specifications & Quantile Curve
# ──────────────────────────────────────────────────────────────────
def generate_figure_2():
    print("\nGenerating Figure 2: Econometric Model Specifications & Quantile Plot (1x3 Symmetric)...")
    # Load Table 1 results and Table S5 quantile estimates
    t1_path = os.path.join(TABLE_DIR, "Table1_Econometric_Results.csv")
    ts5_path = os.path.join(TABLE_DIR, "TableS5_Quantile_Estimates.csv")

    t1 = pd.read_csv(t1_path)
    ts5 = pd.read_csv(ts5_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.2, 5.5),
                                        gridspec_kw={'width_ratios': [1.35, 1.0, 1.0], 'wspace': 0.34})

    # Select representative estimates for Forest Plot
    forest_items = [
        {'spec': 'Model 1', 'var': 'Biological Stock Status (Stocks-at-Risk Index)',
         'coef': t1[(t1['Model'] == 'Model 1 (Physical CI)') & (t1['Variable'] == 'SAR_w')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 1 (Physical CI)') & (t1['Variable'] == 'SAR_w')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 1 (Physical CI)') & (t1['Variable'] == 'SAR_w')]['CI_Upper'].values[0],
         'cat': 'Eco'},
        {'spec': 'Model 1b', 'var': 'Operating Profit Margin (Percentage Points)',
         'coef': t1[(t1['Model'] == 'Model 1b (Monetary Real CI)') & (t1['Variable'] == 'ebit_margin_w')]['Coefficient'].values[0] * 100,
         'ci_lo': t1[(t1['Model'] == 'Model 1b (Monetary Real CI)') & (t1['Variable'] == 'ebit_margin_w')]['CI_Lower'].values[0] * 100,
         'ci_hi': t1[(t1['Model'] == 'Model 1b (Monetary Real CI)') & (t1['Variable'] == 'ebit_margin_w')]['CI_Upper'].values[0] * 100,
         'cat': 'Fin'},
        {'spec': 'Model 2', 'var': 'Physical Carbon Intensity (kg CO2/kg catch)',
         'coef': t1[(t1['Model'] == 'Model 2 (Physical CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_phys_w')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 2 (Physical CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_phys_w')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 2 (Physical CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_phys_w')]['CI_Upper'].values[0],
         'cat': 'EnvPhys'},
        {'spec': 'Model 2b', 'var': 'Real Monetary Carbon Intensity (kg CO2/EUR)',
         'coef': t1[(t1['Model'] == 'Model 2b (Monetary CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_mon_real_w')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 2b (Monetary CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_mon_real_w')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 2b (Monetary CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_mon_real_w')]['CI_Upper'].values[0],
         'cat': 'EnvMon'},
        {'spec': 'Model 3', 'var': 'Operational Subsidy Intensity (per EUR Revenue)',
         'coef': t1[(t1['Model'] == 'Model 3 (Subsidy Nexus)') & (t1['Variable'] == 'subsidy_intensity_w')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 3 (Subsidy Nexus)') & (t1['Variable'] == 'subsidy_intensity_w')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 3 (Subsidy Nexus)') & (t1['Variable'] == 'subsidy_intensity_w')]['CI_Upper'].values[0],
         'cat': 'Pol'},
        {'spec': 'Model 5', 'var': 'Catch Species Diversity (Shannon-Wiener Index)',
         'coef': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'shannon_h_clean')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'shannon_h_clean')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'shannon_h_clean')]['CI_Upper'].values[0],
         'cat': 'Eco'},
        {'spec': 'Model 5', 'var': 'Monetary Intensity × Species Diversity Interaction',
         'coef': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'ci_mon_x_shannon')]['Coefficient'].values[0],
         'ci_lo': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'ci_mon_x_shannon')]['CI_Lower'].values[0],
         'ci_hi': t1[(t1['Model'] == 'Model 5 (Diversification Moderation)') & (t1['Variable'] == 'ci_mon_x_shannon')]['CI_Upper'].values[0],
         'cat': 'Mod'},
    ]

    y_pos = np.arange(len(forest_items))[::-1]
    color_map = {
        'Eco': PAL['teal'],
        'Fin': PAL['blue'],
        'EnvPhys': PAL['sky'],
        'EnvMon': PAL['vermillion'],
        'Pol': PAL['orange'],
        'Mod': PAL['pink']
    }

    ax1.axvline(0, color='#666666', linestyle='--', linewidth=0.8)

    labels = []
    for idx, (y, item) in enumerate(zip(y_pos, forest_items)):
        col = color_map[item['cat']]
        coef = item['coef']
        lo, hi = item['ci_lo'], item['ci_hi']
        ax1.plot([lo, hi], [y, y], color=col, linewidth=1.8, solid_capstyle='round')
        ax1.plot(coef, y, marker='D', color=col, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        labels.append(f"{item['var']}\n[{item['spec']}]")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.2)
    ax1.set_ylim(-0.8, len(forest_items) - 0.2)
    ax1.set_xlabel('Estimated Parameter Coefficient (95% Clustered CI)')
    ax1.set_title('(a) Econometric Parameter Estimates (Models 1–5)', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    cat_patches = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PAL['teal'], label='Biological Stock Health', markersize=6.0),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PAL['sky'], label='Physical Carbon Intensity', markersize=6.0),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PAL['vermillion'], label='Monetary Carbon Intensity', markersize=6.0),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PAL['orange'], label='Subsidy Intensity', markersize=6.0),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PAL['pink'], label='Biodiversity Moderation', markersize=6.0),
    ]
    ax1.legend(handles=cat_patches, loc='lower left', framealpha=0.92, fontsize=8.6)

    # Panel (b): Full 17-Quantile Regression Ribbon Path for Carbon Intensity
    quantiles = ts5['Quantile'].values
    ci_coefs = ts5['CI_Coefficient'].values
    ci_lo = ts5['CI_Lower_95'].values
    ci_hi = ts5['CI_Upper_95'].values

    ols_row = t1[(t1['Model'] == 'Model 2b (Monetary CI on EBIT)') & (t1['Variable'] == 'carbon_intensity_mon_real_w')]
    ols_coef = ols_row['Coefficient'].values[0]
    ols_lo = ols_row['CI_Lower'].values[0]
    ols_hi = ols_row['CI_Upper'].values[0]

    ax2.axhline(0, color='#888888', linestyle=':', linewidth=0.8)
    ax2.axhline(ols_coef, color=PAL['blue'], linestyle='--', linewidth=1.5, label=f'Panel Mean FE ({ols_coef:.1f})')
    ax2.fill_between([0.05, 0.95], ols_lo, ols_hi, color=PAL['blue'], alpha=0.12)

    ax2.plot(quantiles, ci_coefs, 'o-', color=PAL['vermillion'], linewidth=1.8, markersize=4.5,
             label='Quantile Effect: Carbon Intensity')
    ax2.fill_between(quantiles, ci_lo, ci_hi, color=PAL['vermillion'], alpha=0.20)

    ax2.annotate(f'Acute Sensitivity for Distressed Fleets\n({ci_coefs[0]:.1f} pp at τ=0.10)',
                 xy=(quantiles[0], ci_coefs[0]), xytext=(quantiles[0] + 0.04, ci_coefs[0] + 35),
                 arrowprops=dict(arrowstyle='->', color='#222222', lw=0.8),
                 fontsize=8.8, ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#CCCCCC', alpha=0.92))

    ax2.annotate(f'Muted Sensitivity for Robust Fleets\n({ci_coefs[-1]:.1f} pp at τ=0.90)',
                 xy=(quantiles[-1], ci_coefs[-1]), xytext=(quantiles[-1] - 0.34, ci_coefs[-1] - 26),
                 arrowprops=dict(arrowstyle='->', color='#222222', lw=0.8),
                 fontsize=8.8, ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#CCCCCC', alpha=0.92))

    ax2.set_xlabel('Quantile of Operating Profit Margin (τ)')
    ax2.set_ylabel('Marginal Impact on Operating Margin (percentage points)')
    ax2.set_xlim(0.08, 0.92)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.6)
    ax2.set_title('(b) Quantile Sensitivity: Monetary Carbon Intensity', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    # Panel (c): Quantile Regression Ribbon Path for Biological Stock Health
    sar_coefs = ts5['SAR_Coefficient'].values
    sar_se = ts5['SAR_Std_Error'].values
    sar_lo = sar_coefs - 1.96 * sar_se
    sar_hi = sar_coefs + 1.96 * sar_se

    sar_ols_row = t1[(t1['Model'] == 'Model 2b (Monetary CI on EBIT)') & (t1['Variable'] == 'SAR_w')]
    if len(sar_ols_row) > 0:
        sar_ols_coef = sar_ols_row['Coefficient'].values[0]
        sar_ols_lo = sar_ols_row['CI_Lower'].values[0]
        sar_ols_hi = sar_ols_row['CI_Upper'].values[0]
    else:
        sar_ols_coef = 0.38
        sar_ols_lo = -3.37
        sar_ols_hi = 4.13

    ax3.axhline(0, color='#888888', linestyle=':', linewidth=0.8)
    ax3.axhline(sar_ols_coef, color=PAL['blue'], linestyle='--', linewidth=1.5, label=f'Panel Mean FE (+{sar_ols_coef:.2f})')
    ax3.fill_between([0.05, 0.95], sar_ols_lo, sar_ols_hi, color=PAL['blue'], alpha=0.10)

    ax3.plot(quantiles, sar_coefs, 's-', color=PAL['teal'], linewidth=1.8, markersize=4.5,
             label='Quantile Effect: Biological Stock Status')
    ax3.fill_between(quantiles, sar_lo, sar_hi, color=PAL['teal'], alpha=0.20)

    ax3.annotate(f'Progressive Return Across Deciles\n(+{sar_coefs[0]:.2f} at τ=0.10 to +{sar_coefs[-1]:.2f} at τ=0.90)',
                 xy=(quantiles[8], sar_coefs[8]), xytext=(quantiles[8] - 0.30, sar_coefs[8] + 2.6),
                 arrowprops=dict(arrowstyle='->', color='#222222', lw=0.8),
                 fontsize=8.8, ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#CCCCCC', alpha=0.92))

    ax3.set_xlabel('Quantile of Operating Profit Margin (τ)')
    ax3.set_ylabel('Marginal Impact on Operating Margin (percentage points)')
    ax3.set_xlim(0.08, 0.92)
    ax3.set_ylim(-6.0, 8.5)
    ax3.legend(loc='upper left', framealpha=0.92, fontsize=8.6)
    ax3.set_title('(c) Quantile Sensitivity: Biological Stock Health', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    save_fig(fig, "Fig2_Panel_Coefficient_Plot")



# ──────────────────────────────────────────────────────────────────
# FIGURE 3: Multi-Dimensional Decarbonisation Matrix & Cluster Radar
# ──────────────────────────────────────────────────────────────────
def generate_figure_3(df):
    print("\nGenerating Figure 3: Decarbonisation Matrix & Radar Scorecard (1x2 Balanced)...")
    df_2023 = df[df['year'] == 2023].copy()

    agg = df_2023.groupby(['country_name', 'fishing_tech']).agg({
        'carbon_intensity_mon_real_w': 'mean',
        'carbon_intensity_phys_w': 'mean',
        'ebit_margin_w': 'mean',
        'SAR_w': 'mean',
        'revenue_real': 'sum',
        'mean_age_clean': 'mean',
        'jobs_per_tco2': 'mean',
        'subsidy_intensity_w': 'mean'
    }).dropna().reset_index()

    features = ['carbon_intensity_mon_real_w', 'ebit_margin_w', 'SAR_w']
    X = StandardScaler().fit_transform(agg[features])
    Z = linkage(X, method='ward')
    agg['cluster'] = fcluster(Z, t=4, criterion='maxclust')

    fig = plt.figure(figsize=(14.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], polar=True)

    # Panel (a): Matrix
    med_c = agg['carbon_intensity_mon_real_w'].median()
    med_e = agg['ebit_margin_w'].median()

    ax1.axhline(med_e, color='#888888', linestyle=':', linewidth=0.9, alpha=0.7)
    ax1.axvline(med_c, color='#888888', linestyle=':', linewidth=0.9, alpha=0.7)

    rev_norm = (agg['revenue_real'] - agg['revenue_real'].min()) / (agg['revenue_real'].max() - agg['revenue_real'].min())
    bubble_sizes = 35 + rev_norm * 220

    for cid in sorted(agg['cluster'].unique()):
        mask = agg['cluster'] == cid
        color = CLUSTER_COLORS.get(cid, PAL['grey'])
        ax1.scatter(agg.loc[mask, 'carbon_intensity_mon_real_w'], agg.loc[mask, 'ebit_margin_w'],
                    s=bubble_sizes[mask], color=color, label=CLUSTER_LABELS.get(cid, f'Cluster {cid}'),
                    alpha=0.75, edgecolors='black', linewidths=0.5, zorder=3)

    ax1.set_xlim(-0.42, 3.25)
    ax1.set_ylim(-740, 240)

    kw_badge = dict(fontsize=8.6, ha='center', va='center', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#CCCCCC', alpha=0.94, linewidth=0.7))
    ax1.text(0.45, 168, 'High Resilience / Low Carbon\n(Cluster 1: Artisanal Frontier)',
             color=PAL['teal'], **kw_badge)
    ax1.text(2.15, 168, 'High Resilience / High Carbon\n(Cluster 3: Mixed Heavy Trawlers)',
             color=PAL['orange'], **kw_badge)
    ax1.text(0.45, -340, 'Moderate Resilience / Moderate Carbon\n(Cluster 2: Pelagic Industrial)',
             color=PAL['sky'], **kw_badge)
    ax1.text(2.18, -340, 'Vulnerable Fleets / Severe Deficit\n(Cluster 4: Vulnerable Trawl Segments)',
             color=PAL['vermillion'], **kw_badge)

    ax1.set_xlabel('Real Monetary Carbon Intensity (kg CO$_2$ per 2015 EUR real revenue)')
    ax1.set_ylabel('Fleet Operating Profit Margin (%)')
    ax1.legend(loc='lower right', framealpha=0.94, fontsize=8.4, handlelength=1.2)
    ax1.set_title('(a) Empirical Bio-Economic Typology Matrix', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    # Panel (b): Radar Scorecard covering ALL 4 CLUSTERS
    radar_metrics = [
        'Carbon\nEfficiency', 'Operating Profitability', 'Biological Stock Health',
        'Employment\nIntensity', 'Vessel Capital Modernity', 'Subsidy Autonomy'
    ]
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    p5_ebit = agg['ebit_margin_w'].quantile(0.05)
    p95_ebit = agg['ebit_margin_w'].quantile(0.95)

    c_profiles = {}
    for cid in [1, 2, 3, 4]:
        sub_c = agg[agg['cluster'] == cid]
        if len(sub_c) == 0:
            continue
        ce = 1.0 / (1.0 + sub_c['carbon_intensity_mon_real_w'].mean())
        prof = (sub_c['ebit_margin_w'].mean() - p5_ebit) / (p95_ebit - p5_ebit + 1e-4)
        sh = np.clip(1.0 - (sub_c['SAR_w'].mean() / 2.0), 0.1, 0.95)
        sy = np.clip(sub_c['jobs_per_tco2'].mean() / (agg['jobs_per_tco2'].quantile(0.95) + 1e-4), 0.1, 0.95)
        mod = 1.0 - (sub_c['mean_age_clean'].mean() / 50.0)
        sub_aut = np.clip(1.0 - sub_c['subsidy_intensity_w'].mean(), 0.1, 0.95)

        vals = [ce, prof, sh, sy, mod, sub_aut]
        vals = [float(np.clip(v, 0.10, 0.95)) for v in vals]
        vals += vals[:1]
        c_profiles[cid] = vals

    radar_styles = {
        1: {'ls': '-', 'lw': 2.0, 'alpha': 0.15},
        2: {'ls': '--', 'lw': 1.8, 'alpha': 0.12},
        3: {'ls': '-.', 'lw': 1.8, 'alpha': 0.12},
        4: {'ls': ':', 'lw': 2.2, 'alpha': 0.15},
    }

    for cid in [1, 2, 3, 4]:
        if cid in c_profiles:
            vals = c_profiles[cid]
            col = CLUSTER_COLORS[cid]
            sty = radar_styles[cid]
            ax2.plot(angles, vals, color=col, linewidth=sty['lw'], linestyle=sty['ls'], label=CLUSTER_LABELS[cid].split(':')[0])
            ax2.fill(angles, vals, color=col, alpha=sty['alpha'])

    ax2.tick_params(pad=18)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_metrics, fontsize=9.0)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0.25, 0.50, 0.75])
    ax2.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8.2, color='#777777')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.32, 1.16), framealpha=0.94, fontsize=8.6)
    ax2.set_title('(b) Multi-Dimensional Performance Scorecard (All 4 Clusters)', loc='left', fontsize=11.5, fontweight='bold', pad=26)

    save_fig(fig, "Fig3_Decarbonisation_Matrix")


# ──────────────────────────────────────────────────────────────────
# FIGURE 4: Granular Carbon Pricing Simulation & Revenue Recycling
# ──────────────────────────────────────────────────────────────────
def generate_figure_4(df):
    print("\nGenerating Figure 4: Carbon Pricing Simulation & Revenue Recycling (2x2 Symmetric)...")
    t2_path = os.path.join(TABLE_DIR, "Table2_Simulation_Results.csv")
    t2 = pd.read_csv(t2_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.6, 10.6),
                                                 gridspec_kw={'hspace': 0.36, 'wspace': 0.25})

    prices = t2['Carbon_Price_EUR'].values

    # Panel (a): Solvency Survival Curves (Segment Unweighted vs Vessel-Weighted)
    ax1.plot(prices, 100 - t2['Deficit_Pct_Static'], 'o-', color=PAL['vermillion'], linewidth=1.8,
             markersize=5, label='Static Regulatory Cost (Segment-Unweighted)')
    ax1.plot(prices, 100 - t2['Deficit_Pct_Dynamic'], 's--', color=PAL['orange'], linewidth=1.8,
             markersize=5, label='Dynamic Fleet Adaptation (Segment-Unweighted)')
    ax1.plot(prices, 100 - t2['Deficit_Feebate_Vessel'], '^-', color=PAL['blue'], linewidth=1.9,
             markersize=5.5, label='Uniform Vessel Rebate Rule B (Segment-Unweighted)')
    ax1.plot(prices, 100 - t2['Vessel_Deficit_Feebate_Vessel'], 'D-.', color=PAL['teal'], linewidth=2.2,
             markersize=5.5, label='Uniform Vessel Rebate Rule B (Vessel-Weighted Fleet)')
    ax1.axhline(100 - 43.41, color='#666666', linestyle=':', linewidth=0.9, label='Baseline Fleet Solvency (56.6%)')

    ax1.set_xlabel('Simulated Carbon Price (EUR per tonne CO$_2$)')
    ax1.set_ylabel('Solvent Fleet Proportion (Operating Margin > 0, %)')
    ax1.set_ylim(21, 95)
    ax1.set_xlim(-5, 255)
    ax1.legend(loc='lower left', framealpha=0.92, fontsize=8.4)
    ax1.set_title('(a) Fleet Solvency Survival Curves', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (b): Median EBIT Margin Path Across Scenarios
    ax2.plot(prices, t2['Median_EBIT_Static'], 'o-', color=PAL['vermillion'], linewidth=1.8,
             markersize=5, label='Static Policy Cost Scenario')
    ax2.plot(prices, t2['Median_EBIT_Dynamic'], 's--', color=PAL['orange'], linewidth=1.8,
             markersize=5, label='Dynamic Fleet Adaptation')
    ax2.axhline(0, color='#888888', linestyle=':', linewidth=0.9)

    ax2.set_xlabel('Simulated Carbon Price (EUR per tonne CO$_2$)')
    ax2.set_ylabel('Median Fleet Operating Profit Margin (%)')
    ax2.set_ylim(-12, 5)
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.8)
    ax2.set_title('(b) Operating Profit Margin Path', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (c): Paired Comparison at €100/tCO2: Segment-Unweighted vs Vessel-Weighted Fleet
    rules_labels = [
        'Static Regulatory\nCost (€100/t)',
        'Dynamic Fleet\nAdaptation (€100/t)',
        'Crew-Based\nRebate (Rule A)',
        'Uniform Vessel\nRebate (Rule B)',
        'Eco-Efficiency\nBonus (Rule C)'
    ]
    x_idx = np.arange(len(rules_labels))
    w = 0.36

    seg_deficits = [
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Deficit_Pct_Static'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Deficit_Pct_Dynamic'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Deficit_Feebate_FTE'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Deficit_Feebate_Vessel'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Deficit_Feebate_Green'].values[0],
    ]
    vessel_deficits = [
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Vessel_Deficit_Static'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Vessel_Deficit_Dynamic'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Vessel_Deficit_Feebate_FTE'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Vessel_Deficit_Feebate_Vessel'].values[0],
        t2.loc[t2['Carbon_Price_EUR'] == 100, 'Vessel_Deficit_Feebate_Green'].values[0],
    ]

    rects1 = ax3.bar(x_idx - w/2, seg_deficits, width=w, color=PAL['vermillion'], alpha=0.85,
                     edgecolor='black', linewidth=0.5, label='Segment Deficit Share (%)')
    rects2 = ax3.bar(x_idx + w/2, vessel_deficits, width=w, color=PAL['teal'], alpha=0.85,
                     edgecolor='black', linewidth=0.5, label='Vessel-Weighted Deficit (%)')

    ax3.axhline(43.41, color='#666666', linestyle='--', linewidth=0.8, label='Baseline Segment Deficit (43.4%)')
    ax3.axhline(50.61, color='#888888', linestyle=':', linewidth=0.8, label='Baseline Vessel Deficit (50.6%)')

    # Add data labels on Rule B bars
    ax3.annotate(f'{seg_deficits[3]:.1f}%', (x_idx[3] - w/2, seg_deficits[3]),
                 textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8.6, fontweight='bold')
    ax3.annotate(f'{vessel_deficits[3]:.1f}%', (x_idx[3] + w/2, vessel_deficits[3]),
                 textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8.6, fontweight='bold', color=PAL['teal'])

    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(rules_labels, fontsize=8.8)
    ax3.set_ylabel('Fleet Proportion in Operating Deficit (%)')
    ax3.set_ylim(0, 74)
    ax3.legend(loc='upper right', framealpha=0.92, fontsize=8.4)
    ax3.set_title(r'(c) Operating Deficits under Revenue Recycling at €100/tCO$_2$', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (d): Net Fiscal Rebate Transfer per Vessel under Rule B
    df_2023 = df[df['year'] == 2023].copy()
    p_sim = 100
    tot_co2 = df_2023['co2_emissions_tonnes'].sum()
    tot_rev = p_sim * tot_co2
    tot_vessels = df_2023['n_vessels'].sum()
    div_per_vessel = tot_rev / tot_vessels

    gear_order = ['Passive Coastal Gears', 'Dredges and Other Gears', 'Active Demersal Trawlers', 'Pelagic Trawls and Seiners']
    net_transfers_per_v = []
    tot_transfers_m = []
    vessel_counts = []
    bar_colors = []

    for gf in gear_order:
        sub = df_2023[df_2023['gear_family'] == gf]
        v_cnt = sub['n_vessels'].sum()
        co2_cnt = sub['co2_emissions_tonnes'].sum()
        tax_cnt = p_sim * co2_cnt
        div_cnt = v_cnt * div_per_vessel
        net_cnt = div_cnt - tax_cnt
        per_v = net_cnt / v_cnt if v_cnt > 0 else 0
        net_transfers_per_v.append(per_v)
        tot_transfers_m.append(net_cnt / 1e6)
        vessel_counts.append(v_cnt)
        bar_colors.append(PAL['teal'] if per_v > 0 else PAL['vermillion'])

    x_d = np.arange(len(gear_order))
    ax4.axhline(0, color='#333333', linestyle='-', linewidth=0.9)
    bars_d = ax4.bar(x_d, net_transfers_per_v, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)

    gear_display_labels = [
        f'Passive Coastal\nGears\n({int(vessel_counts[0]):,} vessels)',
        f'Dredges and\nOther Gears\n({int(vessel_counts[1]):,} vessels)',
        f'Active Demersal\nTrawlers\n({int(vessel_counts[2]):,} vessels)',
        f'Pelagic Trawls\nand Seiners\n({int(vessel_counts[3]):,} vessels)'
    ]

    for idx, (b, per_v, tot_m) in enumerate(zip(bars_d, net_transfers_per_v, tot_transfers_m)):
        if per_v > 0:
            ax4.text(b.get_x() + b.get_width() / 2, per_v + 1800,
                     f'+€{per_v:,.0f} / vessel\n(+€{tot_m:.1f}M Net Rebate)',
                     ha='center', va='bottom', fontsize=8.4, fontweight='bold', color=PAL['teal'])
        else:
            ax4.text(b.get_x() + b.get_width() / 2, per_v - 2200,
                     f'-€{abs(per_v):,.0f} / vessel\n(-€{abs(tot_m):.1f}M Net Tax)',
                     ha='center', va='top', fontsize=8.4, fontweight='bold', color=PAL['vermillion'])

    ax4.set_xticks(x_d)
    ax4.set_xticklabels(gear_display_labels, fontsize=8.8)
    ax4.set_ylabel('Net Annual Fiscal Transfer (EUR per vessel per year)')
    ax4.set_ylim(-74000, 26000)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:+,.0f} €'))
    ax4.set_title('(d) Net Fiscal Rebate Transfer per Vessel under Rule B', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    save_fig(fig, "Fig4_Simulation_Scenarios")


# ──────────────────────────────────────────────────────────────────
# FIGURE 5: Technological Distribution & Eco-Efficiency Pareto Frontier
# ──────────────────────────────────────────────────────────────────
def generate_figure_5(df):
    print("\nGenerating Figure 5: Technological Distribution & Eco-Efficiency Frontier (1x2 Balanced)...")
    fig = plt.figure(figsize=(15.4, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.05], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Technology name mapping for non-abbreviated presentation
    tech_name_map = {
        'DFN': 'Drift and Fixed Nets',
        'FPO': 'Pots and Traps',
        'HOK': 'Longlines and Hooks',
        'PG':  'Polyvalent Passive',
        'PGO': 'Passive Gears Other',
        'PGP': 'Polyvalent Passive Other',
        'DTS': 'Demersal Trawl and Seine',
        'TBB': 'Beam Trawl',
        'PS':  'Purse Seine',
        'TM':  'Pelagic Trawl',
        'DRB': 'Boat Dredge',
        'MGO': 'Other Mobile Gears',
        'MGP': 'Polyvalent Mobile Gears',
        'PMP': 'Polyvalent Mobile Other'
    }

    # Panel (a): Technological distribution of Physical Carbon Intensity
    df_clean = df.dropna(subset=['carbon_intensity_phys_w', 'fishing_tech']).copy()
    tech_order = df_clean.groupby('fishing_tech')['carbon_intensity_phys_w'].median().sort_values().index

    palette_tech = []
    passive_set = {'DFN', 'FPO', 'HOK', 'PG', 'PGO', 'PGP'}
    pelagic_set = {'PS', 'TM'}
    for t in tech_order:
        if t in passive_set:
            palette_tech.append(PAL['teal'])
        elif t in pelagic_set:
            palette_tech.append(PAL['blue'])
        else:
            palette_tech.append(PAL['vermillion'])

    sns.boxplot(data=df_clean, x='fishing_tech', y='carbon_intensity_phys_w', order=tech_order,
                palette=palette_tech, ax=ax1, width=0.45, linewidth=0.7, showfliers=False)

    for idx, t in enumerate(tech_order):
        vals = df_clean[df_clean['fishing_tech'] == t]['carbon_intensity_phys_w'].values
        if len(vals) > 80:
            vals = np.random.choice(vals, size=80, replace=False)
        jitter = np.random.normal(idx, 0.08, size=len(vals))
        ax1.scatter(jitter, vals, color=palette_tech[idx], alpha=0.25, s=7, edgecolor='none')

    ax1.set_xlabel('Harvesting Technology')
    ax1.set_ylabel('Physical Carbon Intensity (kg CO$_2$ per kg landed catch)')
    tech_display_labels = [tech_name_map.get(t, t) for t in tech_order]
    ax1.set_xticklabels(tech_display_labels, rotation=45, ha='right', fontsize=9.0)
    ax1.set_title('(a) Technological Distribution of Physical Carbon Intensity', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    legend_elements = [
        mpatches.Patch(facecolor=PAL['teal'], edgecolor='black', label='Passive Coastal Gears (Low Carbon Intensity)'),
        mpatches.Patch(facecolor=PAL['blue'], edgecolor='black', label='Pelagic Trawls and Seiners (Moderate Carbon Intensity)'),
        mpatches.Patch(facecolor=PAL['vermillion'], edgecolor='black', label='Active Demersal Trawlers and Dredges (High Carbon Intensity)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.92, fontsize=8.4)

    # Panel (b): Dual Functional Unit Pareto Frontier with Bootstrap Envelope
    p8_path = os.path.join(TABLE_DIR, "TableS8_Pareto_Parameters.csv")
    p8 = pd.read_csv(p8_path)
    theta = p8.loc[p8['Parameter'] == 'theta (scale)', 'Estimate'].values[0]
    gamma = p8.loc[p8['Parameter'] == 'gamma (curvature)', 'Estimate'].values[0]

    df_eff = df.dropna(subset=['kg_fish_per_kg_co2', 'eur_rev_per_kg_co2']).copy()
    df_eff = df_eff[(df_eff['kg_fish_per_kg_co2'] > 0) & (df_eff['eur_rev_per_kg_co2'] > 0)]

    tech_eff = df_eff.groupby('fishing_tech').agg({
        'kg_fish_per_kg_co2': 'median',
        'eur_rev_per_kg_co2': 'median',
        'revenue_real': 'sum',
        'gear_family': 'first'
    }).reset_index()

    gear_col_map = {
        'Passive Coastal Gears': PAL['teal'],
        'Pelagic Trawls and Seiners': PAL['blue'],
        'Active Demersal Trawlers': PAL['vermillion'],
        'Dredges and Other Gears': PAL['orange']
    }

    sample_eff = df_eff.sample(min(len(df_eff), 1500), random_state=42)
    ax2.scatter(sample_eff['kg_fish_per_kg_co2'], sample_eff['eur_rev_per_kg_co2'],
                c='#DDDDDD', alpha=0.35, s=6, edgecolor='none', zorder=1)

    annot_config = {
        'PGO': ('Passive Other', (-4, 16), 'center', 'bottom'),
        'TBB': ('Beam Trawl', (-16, -14), 'right', 'top'),
        'DTS': ('Demersal Trawl/Seine', (10, -24), 'left', 'top'),
        'HOK': ('Longlines/Hooks', (-22, -6), 'right', 'top'),
        'PGP': ('Polyv. Pass. Other', (-26, 2), 'right', 'center'),
        'DFN': ('Drift/Fixed Nets', (-24, 16), 'right', 'bottom'),
        'MGO': ('Other Mobile Gears', (8, 16), 'left', 'bottom'),
        'DRB': ('Boat Dredge', (-18, -18), 'right', 'top'),
        'FPO': ('Pots & Traps', (4, 12), 'left', 'bottom'),
        'PMP': ('Polyv. Mobile Other', (-2, -28), 'center', 'top'),
        'PG':  ('Polyvalent Passive', (16, 16), 'left', 'bottom'),
        'MGP': ('Polyv. Mobile Gears', (20, -10), 'left', 'top'),
        'PS':  ('Purse Seine', (20, 4), 'left', 'center'),
        'TM':  ('Pelagic Trawl', (16, -8), 'left', 'top'),
    }

    for _, row in tech_eff.iterrows():
        col = gear_col_map.get(row['gear_family'], PAL['grey'])
        ax2.scatter(row['kg_fish_per_kg_co2'], row['eur_rev_per_kg_co2'],
                    s=120, color=col, edgecolors='black', linewidth=0.6, zorder=4)
        lbl, off, ha_align, va_align = annot_config.get(row['fishing_tech'], (row['fishing_tech'], (4, 4), 'left', 'center'))
        ax2.annotate(lbl, (row['kg_fish_per_kg_co2'], row['eur_rev_per_kg_co2']),
                     fontsize=8.2, ha=ha_align, va=va_align, xytext=off, textcoords='offset points',
                     fontweight='bold', zorder=5,
                     arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6, shrinkA=0, shrinkB=4),
                     bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.85))

    p_x = np.logspace(-1.0, 1.2, 100)
    p_y = theta / (p_x ** gamma)
    ax2.plot(p_x, p_y, color='#222222', linestyle='-', linewidth=1.5, zorder=2,
             label=f'Pareto Hyperbolic Frontier: θ={theta:.2f}, γ={gamma:.2f}')

    # Add 95% bootstrap envelope with correct parameter corner envelope
    theta_lo = p8.loc[p8['Parameter'] == 'theta (scale)', 'CI_Lower_95'].values[0]
    theta_hi = p8.loc[p8['Parameter'] == 'theta (scale)', 'CI_Upper_95'].values[0]
    gamma_lo = p8.loc[p8['Parameter'] == 'gamma (curvature)', 'CI_Lower_95'].values[0]
    gamma_hi = p8.loc[p8['Parameter'] == 'gamma (curvature)', 'CI_Upper_95'].values[0]
    c1 = theta_lo / (p_x ** gamma_lo)
    c2 = theta_lo / (p_x ** gamma_hi)
    c3 = theta_hi / (p_x ** gamma_lo)
    c4 = theta_hi / (p_x ** gamma_hi)
    p_y_lo = np.minimum.reduce([c1, c2, c3, c4])
    p_y_hi = np.maximum.reduce([c1, c2, c3, c4])
    ax2.fill_between(p_x, p_y_lo, p_y_hi, color='#888888', alpha=0.18, label='95% Bootstrap Confidence Envelope')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Physical Catch Efficiency\n(kg landed catch per kg CO$_2$ emitted, Log Scale)')
    ax2.set_ylabel('Economic Revenue Efficiency\n(Real 2015 EUR revenue per kg CO$_2$ emitted, Log Scale)')
    ax2.set_xlim(0.06, 28.0)
    ax2.set_ylim(0.16, 52.0)

    ax2.text(0.08, 17.5, 'High Value / Low Catch Mass\n(Artisanal Crustacean and Demersal Lines)\n[Favored by Revenue Metric]',
             fontsize=8.3, color=PAL['teal'], fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.82))
    ax2.text(2.4, 0.23, 'High Catch Mass / Low Value\n(Industrial Pelagic Trawls)\n[Favored by Catch Weight Metric]',
             fontsize=8.3, color=PAL['blue'], fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.82))

    ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.4)
    ax2.set_title(r'(b) Empirical Dual Functional Unit Pareto Frontier', loc='left', fontsize=11.5, fontweight='bold', pad=12)

    save_fig(fig, "Fig5_LCA_Validation")


# ──────────────────────────────────────────────────────────────────
# SUPPLEMENTARY FIGURES S1 TO S6
# ──────────────────────────────────────────────────────────────────
def generate_supplementary_figures(df):
    print("\nGenerating Supplementary Figures S1–S6 (Fully Harmonised & Non-Abbreviated)...")

    # Fig S1: Fleet capital modernity (1x3 Symmetric)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.8), sharey=True)
    df_clean = df.dropna(subset=['carbon_intensity_phys_w', 'mean_age_clean', 'total_kw', 'total_gt']).copy()
    sample = df_clean.sample(min(len(df_clean), 2000), random_state=42)

    # Panel (a): Mean Vessel Age (linear fit + 2D KDE contours)
    ax1.scatter(sample['mean_age_clean'], sample['carbon_intensity_phys_w'], color=PAL['blue'], alpha=0.18, s=12, edgecolor='none')
    try:
        sns.kdeplot(data=sample, x='mean_age_clean', y='carbon_intensity_phys_w', levels=5, color=PAL['blue'], alpha=0.45, ax=ax1, linewidths=1.0)
    except Exception:
        pass
    res_age = stats.linregress(sample['mean_age_clean'], sample['carbon_intensity_phys_w'])
    x_age = np.linspace(sample['mean_age_clean'].min(), sample['mean_age_clean'].max(), 100)
    y_age = res_age.intercept + res_age.slope * x_age
    ax1.plot(x_age, y_age, color=PAL['vermillion'], linewidth=2.0,
             label=f'Linear Fit (Slope = {res_age.slope:+.2f})')
    ax1.set_xlabel('Mean Vessel Age (Years)')
    ax1.set_ylabel('Physical Carbon Intensity (kg CO$_2$ per kg catch)')
    ax1.set_ylim(0, 12.5)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8.8)
    ax1.set_title('(a) Mean Vessel Age', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (b): Total Installed Engine Power (log scale fit + 2D KDE contours)
    kw_sub = sample[sample['total_kw'] > 0]
    p_kw = np.polyfit(np.log(kw_sub['total_kw']), kw_sub['carbon_intensity_phys_w'], 1)
    kw_grid = np.logspace(np.log10(kw_sub['total_kw'].min()), np.log10(kw_sub['total_kw'].max()), 100)
    y_kw_grid = p_kw[0] * np.log(kw_grid) + p_kw[1]

    ax2.scatter(kw_sub['total_kw'], kw_sub['carbon_intensity_phys_w'], color=PAL['teal'], alpha=0.18, s=12, edgecolor='none')
    try:
        sns.kdeplot(data=kw_sub, x='total_kw', y='carbon_intensity_phys_w', log_scale=(True, False), levels=5, color=PAL['teal'], alpha=0.45, ax=ax2, linewidths=1.0)
    except Exception:
        pass
    ax2.plot(kw_grid, y_kw_grid, color=PAL['vermillion'], linewidth=2.0,
             label=f'Logarithmic Fit (Slope = {p_kw[0]:.2f})')
    ax2.set_xscale('log')
    ax2.set_xlabel('Total Installed Engine Power (Kilowatts, Log Scale)')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8.8)
    ax2.set_title('(b) Installed Engine Power', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    # Panel (c): Total Gross Tonnage (log scale fit + 2D KDE contours)
    gt_sub = sample[sample['total_gt'] > 0]
    p_gt = np.polyfit(np.log(gt_sub['total_gt']), gt_sub['carbon_intensity_phys_w'], 1)
    gt_grid = np.logspace(np.log10(gt_sub['total_gt'].min()), np.log10(gt_sub['total_gt'].max()), 100)
    y_gt_grid = p_gt[0] * np.log(gt_grid) + p_gt[1]

    ax3.scatter(gt_sub['total_gt'], gt_sub['carbon_intensity_phys_w'], color=PAL['orange'], alpha=0.18, s=12, edgecolor='none')
    try:
        sns.kdeplot(data=gt_sub, x='total_gt', y='carbon_intensity_phys_w', log_scale=(True, False), levels=5, color=PAL['orange'], alpha=0.45, ax=ax3, linewidths=1.0)
    except Exception:
        pass
    ax3.plot(gt_grid, y_gt_grid, color=PAL['vermillion'], linewidth=2.0,
             label=f'Logarithmic Fit (Slope = {p_gt[0]:.2f})')
    ax3.set_xscale('log')
    ax3.set_xlabel('Total Gross Tonnage (Gross Tonnes, Log Scale)')
    ax3.legend(loc='upper left', framealpha=0.9, fontsize=8.8)
    ax3.set_title('(c) Total Gross Tonnage', loc='left', fontsize=11.5, fontweight='bold', pad=10)

    save_fig(fig, "FigS1_Age_Carbon_Intensity")

    # Fig S2: Spatial & Econometric Fossil fuel tax exemptions vs Carbon Intensity
    geojson_path = os.path.join(PROJECT_DIR, "data", "external", "europe.geojson")
    if 'fuel_tax_exemption_eur_m' in df.columns and os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        
        # Ensure only Southern Cyprus (Republic of Cyprus) is mapped with fleet data
        # while Northern Cyprus is rendered as background non-reporting territory
        cyp_rows = gdf[gdf['ISO3'] == 'CYP']
        if not cyp_rows.empty and cyp_rows.geometry.iloc[0].bounds[3] > 35.25:
            # Full island geometry detected; split using Natural Earth 50m boundaries
            med_geojson = os.path.join(PROJECT_DIR, "data", "external", "ne_50m_euro_med_countries.geojson")
            if os.path.exists(med_geojson):
                gdf_med = gpd.read_file(med_geojson)
                cyp_s = gdf_med[gdf_med['ISO3'] == 'CYP'].geometry.iloc[0]
                cyp_n = gdf_med[gdf_med['ISO3'] == 'CYN'].geometry.iloc[0]
                cyp_idx = cyp_rows.index[0]
                gdf.loc[cyp_idx, 'geometry'] = cyp_s
                if 'CYN' not in gdf['ISO3'].values:
                    row_n = gdf.loc[cyp_idx].copy()
                    row_n['ISO3'] = 'CYN'
                    row_n['NAME'] = 'Northern Cyprus'
                    row_n['geometry'] = cyp_n
                    gdf = pd.concat([gdf, gpd.GeoDataFrame([row_n], crs=gdf.crs)], ignore_index=True)

        iso_map = {
            'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Croatia': 'HRV', 'Cyprus': 'CYP',
            'Denmark': 'DNK', 'Estonia': 'EST', 'Finland': 'FIN', 'France': 'FRA',
            'Germany': 'DEU', 'Greece': 'GRC', 'Ireland': 'IRL', 'Italy': 'ITA',
            'Latvia': 'LVA', 'Lithuania': 'LTU', 'Malta': 'MLT', 'Netherlands': 'NLD',
            'Poland': 'POL', 'Portugal': 'PRT', 'Romania': 'ROU', 'Slovenia': 'SVN',
            'Spain': 'ESP', 'Sweden': 'SWE'
        }
        df_geo = df.copy()
        df_geo['ISO3'] = df_geo['country_name'].map(iso_map)
        country_agg = df_geo.groupby('ISO3').agg({
            'fuel_tax_exemption_eur_m': 'mean',
            'carbon_intensity_phys_w': 'mean',
            'live_weight_landings': 'sum'
        }).dropna().reset_index()

        merged = gdf.merge(country_agg, on='ISO3', how='left')

        fig_w, fig_h = 12.2, 5.1
        fig = plt.figure(figsize=(fig_w, fig_h))

        map_w, map_h = 2.95, 3.60
        map_y0 = 1.12

        x_a = 0.35
        x_b = x_a + map_w + 0.35
        x_c = x_b + map_w + 0.90
        plot_w = 3.45

        ax1 = fig.add_axes([x_a / fig_w, map_y0 / fig_h, map_w / fig_w, map_h / fig_h])
        ax2 = fig.add_axes([x_b / fig_w, map_y0 / fig_h, map_w / fig_w, map_h / fig_h])

        ax3_y0 = 0.62
        ax3_h = (map_y0 + map_h) - ax3_y0
        ax3 = fig.add_axes([x_c / fig_w, ax3_y0 / fig_h, plot_w / fig_w, ax3_h / fig_h])

        xlim = (-11.5, 35.2)
        ylim = (34.0, 69.0)

        # Panel (a): Map Carbon Intensity
        gdf.plot(ax=ax1, color='#F2F2F2', edgecolor='#CCCCCC', linewidth=0.5)
        merged[merged['ISO3'].isin(iso_map.values())].plot(
            column='carbon_intensity_phys_w', ax=ax1, cmap='YlOrRd', edgecolor='#333333',
            linewidth=0.6
        )
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_axis_off()
        ax1.set_title('(a) Physical Carbon Intensity', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        # Colorbar for Panel (a)
        cax1 = fig.add_axes([x_a / fig_w + 0.02, 0.50 / fig_h, (map_w - 0.25) / fig_w, 0.12 / fig_h])
        sm1 = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=country_agg['carbon_intensity_phys_w'].min(), vmax=country_agg['carbon_intensity_phys_w'].max()))
        sm1._A = []
        cbar1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal')
        cbar1.set_label(r'Carbon Intensity (kg $\mathrm{CO}_2$ per kg catch)', fontsize=9.0)
        cbar1.ax.tick_params(labelsize=8.8)

        # Panel (b): Map Fuel Subsidies
        gdf.plot(ax=ax2, color='#F2F2F2', edgecolor='#CCCCCC', linewidth=0.5)
        merged[merged['ISO3'].isin(iso_map.values())].plot(
            column='fuel_tax_exemption_eur_m', ax=ax2, cmap='Blues', edgecolor='#333333',
            linewidth=0.6
        )
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_axis_off()
        ax2.set_title('(b) Fuel Tax Concessions (OECD)', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        # Colorbar for Panel (b)
        cax2 = fig.add_axes([x_b / fig_w + 0.02, 0.50 / fig_h, (map_w - 0.25) / fig_w, 0.12 / fig_h])
        sm2 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=country_agg['fuel_tax_exemption_eur_m'].min(), vmax=country_agg['fuel_tax_exemption_eur_m'].max()))
        sm2._A = []
        cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
        cbar2.set_label('Fuel Concessions (Million EUR)', fontsize=9.0)
        cbar2.ax.tick_params(labelsize=8.8)

        # Panel (c): Bivariate Econometric Regression
        res_s2 = stats.linregress(country_agg['fuel_tax_exemption_eur_m'], country_agg['carbon_intensity_phys_w'])
        landings_norm = country_agg['live_weight_landings'] / country_agg['live_weight_landings'].max() * 420 + 50
        ax3.scatter(country_agg['fuel_tax_exemption_eur_m'], country_agg['carbon_intensity_phys_w'],
            s=landings_norm, color=PAL['orange'], edgecolor='black', linewidth=0.8, alpha=0.75, zorder=4)

        iso_config = {
            'ESP': ((0, 24), 'center', 'bottom'),
            'ITA': ((0, 18), 'center', 'bottom'),
            'FRA': ((0, -24), 'center', 'top'),
            'SWE': ((14, 2), 'left', 'center'),
            'CYP': ((14, 4), 'left', 'center'),
            'HRV': ((12, 18), 'left', 'bottom'),
            'GRC': ((6, 16), 'center', 'bottom'),
            'PRT': ((18, 12), 'left', 'bottom'),
            'DNK': ((34, -6), 'left', 'top'),
            'DEU': ((16, 12), 'left', 'bottom'),
            'IRL': ((4, 22), 'center', 'bottom'),
            'NLD': ((36, 2), 'left', 'center'),
            'BEL': ((-2, 28), 'center', 'bottom'),
            'LTU': ((-14, 24), 'right', 'bottom'),
            'MLT': ((-20, 12), 'right', 'bottom'),
            'BGR': ((-20, 0), 'right', 'center'),
            'SVN': ((-18, -12), 'right', 'top'),
            'ROU': ((-2, -28), 'center', 'top'),
            'LVA': ((-22, -10), 'right', 'top'),
            'EST': ((-20, -26), 'right', 'top'),
            'FIN': ((14, -24), 'left', 'top'),
            'POL': ((26, -14), 'left', 'top')
        }
        for _, r in country_agg.iterrows():
            cfg = iso_config.get(r['ISO3'], ((0, 5), 'center', 'bottom'))
            off, ha_pos, va_pos = cfg
            ax3.annotate(r['ISO3'], (r['fuel_tax_exemption_eur_m'], r['carbon_intensity_phys_w']),
                         fontsize=8.2, ha=ha_pos, va=va_pos, xytext=off, textcoords='offset points', fontweight='bold',
                         arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6, shrinkA=0, shrinkB=3),
                         bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88))

        x_fit = np.linspace(country_agg['fuel_tax_exemption_eur_m'].min(), country_agg['fuel_tax_exemption_eur_m'].max(), 100)
        y_fit = res_s2.intercept + res_s2.slope * x_fit
        ax3.plot(x_fit, y_fit, color=PAL['blue'], linestyle='--', linewidth=1.8,
                 label=f'Linear Regression (Slope β = +{res_s2.slope:.4f}, p = {res_s2.pvalue:.3f})')

        ax3.set_xlabel('Mean Annual Fossil Fuel Tax Concessions (Million EUR)', fontsize=9.8)
        ax3.set_ylabel(r'Mean Fleet Physical Carbon Intensity (kg $\mathrm{CO}_2$ per kg catch)', fontsize=9.8)
        ax3.set_xlim(-85, 510)
        ax3.set_ylim(-5.2, 19.8)
        ax3.legend(loc='upper left', framealpha=0.92, fontsize=8.4)
        ax3.set_title('(c) Econometric Alignment: Subsidies vs Intensity', loc='left', fontsize=11.0, fontweight='bold', pad=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(True, linestyle='--', color='#EAEAEA', alpha=0.8)

        save_fig(fig, "FigS2_Subsidy_Carbon_Intensity")

    # Fig S3: Historical ETS vulnerability timeline & Cost Exposure (1x2 Symmetric)
    if 'ets_vulnerable' in df.columns and 'ets_price_eur' in df.columns:
        yearly = df.groupby('year').agg({
            'ets_vulnerable': 'mean',
            'ets_price_eur': 'first'
        }).loc[2013:2023].reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.2), gridspec_kw={'wspace': 0.32})

        # Panel (a): Historical ETS Price and Deficit Rate
        ax1.bar(yearly['year'], yearly['ets_vulnerable'] * 100, color=PAL['vermillion'],
                alpha=0.75, edgecolor='black', linewidth=0.5, label='Fleet Segments in Deficit (%)')
        ax1.set_ylabel('Fleet Segments in Operating Deficit under ETS (%)', color=PAL['vermillion'])
        ax1.set_xlabel('Calendar Year')
        ax1.set_ylim(25, 72)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(yearly['year'], yearly['ets_price_eur'], 'D-', color=PAL['blue'],
                     markersize=6, linewidth=2.0, label='Historical ETS Carbon Price')
        ax1_twin.set_ylabel('Emissions Trading System Carbon Price (EUR per tonne CO$_2$)', color=PAL['blue'])
        ax1_twin.spines['top'].set_visible(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.92, fontsize=8.8)
        ax1.set_title('(a) Historical Emissions Trading System Price and Deficit Rate', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        # Panel (b): Carbon Cost Exposure (% of Gross Revenue) across Simulated Carbon Prices
        df_2023 = df[df['year'] == 2023].copy()
        carbon_prices = [25, 50, 100, 150]
        for p in carbon_prices:
            df_2023[f'cost_share_{p}'] = (df_2023['co2_emissions_tonnes'] * p) / (df_2023['real_revenue'] + 1e-6) * 100

        gear_families = ['Passive Coastal Gears', 'Dredges and Other Gears', 'Pelagic Trawls and Seiners', 'Active Demersal Trawlers']
        summary_exposure = df_2023.groupby('gear_family')[[f'cost_share_{p}' for p in carbon_prices]].median().loc[gear_families]

        x_g = np.arange(len(gear_families))
        width_bar = 0.18
        palette_bars = [PAL['sky'], PAL['teal'], PAL['orange'], PAL['vermillion']]

        for i, p in enumerate(carbon_prices):
            vals = summary_exposure[f'cost_share_{p}'].values
            ax2.bar(x_g + (i - 1.5) * width_bar, vals, width=width_bar, color=palette_bars[i],
                    label=f'€{p} per tonne CO$_2$', edgecolor='black', linewidth=0.5, alpha=0.88)

        # Annotate highest impact on Demersal Trawlers
        demersal_150 = summary_exposure.loc['Active Demersal Trawlers', 'cost_share_150']
        passive_150 = summary_exposure.loc['Passive Coastal Gears', 'cost_share_150']
        ax2.annotate(f'{demersal_150:.1f}%\nRevenue Exposure',
                     xy=(3 + 1.5 * width_bar, demersal_150), xytext=(3 + 0.08, demersal_150 + 2.8),
                     arrowprops=dict(arrowstyle='->', color='#222222', lw=0.8),
                     fontsize=8.6, ha='center', fontweight='bold', color=PAL['vermillion'])
        ax2.annotate(f'{passive_150:.1f}%\nExposure',
                     xy=(0 + 1.5 * width_bar, passive_150), xytext=(0 + 0.08, passive_150 + 3.2),
                     arrowprops=dict(arrowstyle='->', color='#222222', lw=0.8),
                     fontsize=8.6, ha='center', fontweight='bold', color=PAL['teal'])

        gear_short_labels = ['Passive Coastal\nGears', 'Dredges and\nOther Gears', 'Pelagic Trawls\nand Seiners', 'Active Demersal\nTrawlers']
        ax2.set_xticks(x_g)
        ax2.set_xticklabels(gear_short_labels, fontsize=9.0)
        ax2.set_ylabel('Median Carbon Cost Share of Gross Revenue (%)')
        ax2.set_ylim(0, 25)
        ax2.legend(loc='upper left', framealpha=0.92, fontsize=8.6)
        ax2.set_title('(b) Carbon Cost Exposure across Harvesting Fleets', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        save_fig(fig, "FigS3_Dynamic_ETS_Vulnerability")

    # Fig S4: Cluster diagnostic curves (1x3 Symmetric)
    s6_path = os.path.join(TABLE_DIR, "TableS6_Cluster_Diagnostics.csv")
    if os.path.exists(s6_path):
        s6 = pd.read_csv(s6_path)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.5))

        # (a) Silhouette
        ax1.plot(s6['k'], s6['Silhouette_Score'], 'o-', color=PAL['teal'], linewidth=2.0, markersize=6)
        ax1.axvspan(3.8, 4.2, color='#EAEAEA', alpha=0.6)
        ax1.axvline(4, color=PAL['vermillion'], linestyle='--', linewidth=1.2)
        ax1.set_xlabel('Number of Candidate Clusters')
        ax1.set_ylabel('Mean Silhouette Score')
        ax1.set_title('(a) Silhouette Score Criterion', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        # (b) Calinski-Harabasz
        ax2.plot(s6['k'], s6['Calinski_Harabasz'], 's-', color=PAL['blue'], linewidth=2.0, markersize=6)
        ax2.axvspan(3.8, 4.2, color='#EAEAEA', alpha=0.6)
        ax2.axvline(4, color=PAL['vermillion'], linestyle='--', linewidth=1.2)
        ax2.set_xlabel('Number of Candidate Clusters')
        ax2.set_ylabel('Calinski-Harabasz Index')
        ax2.set_title('(b) Calinski-Harabasz Criterion', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        # (c) Davies-Bouldin
        ax3.plot(s6['k'], s6['Davies_Bouldin'], '^-', color=PAL['vermillion'], linewidth=2.0, markersize=6)
        ax3.axvspan(3.8, 4.2, color='#EAEAEA', alpha=0.6)
        ax3.axvline(4, color=PAL['vermillion'], linestyle='--', linewidth=1.2)
        ax3.annotate('Optimal 4-Cluster Solution\n(Minimizes Index)', xy=(4, s6.loc[s6['k']==4, 'Davies_Bouldin'].values[0]),
                     xytext=(4.3, 1.25), arrowprops=dict(facecolor=PAL['vermillion'], arrowstyle='->', lw=1.2),
                     fontsize=8.6, fontweight='bold', color=PAL['vermillion'],
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=PAL['vermillion'], alpha=0.92))
        ax3.set_xlabel('Number of Candidate Clusters')
        ax3.set_ylabel('Davies-Bouldin Index (Lower is Superior)')
        ax3.set_ylim(0.40, 1.55)
        ax3.set_title('(c) Davies-Bouldin Criterion', loc='left', fontsize=11.0, fontweight='bold', pad=10)

        save_fig(fig, "FigS4_Cluster_Diagnostics")

    # Fig S5: Comprehensive 3-Panel Cluster Stability and Validation (1x3 Symmetric)
    df_2023 = df[(df['year'] == 2023) & (df['is_valid_analysis_segment'] == 1)].copy()
    features = ['carbon_intensity_mon_real_w', 'ebit_margin_w', 'SAR_w', 'subsidy_intensity']
    df_cluster = df_2023.dropna(subset=features).copy()
    X = df_cluster[features].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n_samples = len(X_std)

    Z = linkage(X_std, method='ward')
    base_labels = fcluster(Z, 4, criterion='maxclust')

    np.random.seed(42)
    B = 1000
    overall_jaccards = []
    cluster_jaccards = {c: [] for c in range(1, 5)}

    for b in range(B):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_b = X_std[idx]
        Z_b = linkage(X_b, method='ward')
        labels_b = fcluster(Z_b, 4, criterion='maxclust')
        orig_b = base_labels[idx]
        pair_orig = (orig_b[:, None] == orig_b)
        pair_b = (labels_b[:, None] == labels_b)
        inter = np.logical_and(pair_orig, pair_b).sum() - n_samples
        union = np.logical_or(pair_orig, pair_b).sum() - n_samples
        overall_jaccards.append(inter / max(union, 1))

        for c in range(1, 5):
            in_c = (orig_b == c)
            if in_c.sum() == 0:
                continue
            max_j_c = 0
            for k_b in np.unique(labels_b):
                in_kb = (labels_b == k_b)
                int_c = np.logical_and(in_c, in_kb).sum()
                un_c = np.logical_or(in_c, in_kb).sum()
                j_val = int_c / un_c if un_c > 0 else 0
                if j_val > max_j_c:
                    max_j_c = j_val
            cluster_jaccards[c].append(max_j_c)

    k_values = [2, 3, 4, 5, 6]
    k_means = []
    k_ci_lo = []
    k_ci_hi = []
    for k in k_values:
        Z_k = linkage(X_std, method='ward')
        base_k = fcluster(Z_k, k, criterion='maxclust')
        k_jaccards = []
        for b in range(500):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b = X_std[idx]
            Z_b = linkage(X_b, method='ward')
            labels_b = fcluster(Z_b, k, criterion='maxclust')
            orig_b = base_k[idx]
            pair_orig = (orig_b[:, None] == orig_b)
            pair_b = (labels_b[:, None] == labels_b)
            inter = np.logical_and(pair_orig, pair_b).sum() - n_samples
            union = np.logical_or(pair_orig, pair_b).sum() - n_samples
            k_jaccards.append(inter / max(union, 1))
        k_means.append(np.mean(k_jaccards))
        k_ci_lo.append(np.percentile(k_jaccards, 2.5))
        k_ci_hi.append(np.percentile(k_jaccards, 97.5))

    # Plot 3-panel figure
    fig = plt.figure(figsize=(16.2, 5.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.30)

    # Panel (a): Density & Histogram
    ax1 = fig.add_subplot(gs[0])
    mean_j = np.mean(overall_jaccards)
    ci_lo = np.percentile(overall_jaccards, 2.5)
    ci_hi = np.percentile(overall_jaccards, 97.5)

    sns.histplot(overall_jaccards, bins=25, kde=True, color=PAL['teal'], stat='density',
                 edgecolor=PAL['teal'], linewidth=0.8, alpha=0.35, ax=ax1,
                 line_kws={'linewidth': 2.0, 'color': PAL['teal']})

    ax1.axvline(mean_j, color=PAL['teal'], linestyle='-', linewidth=2.0,
                label=f'Mean Jaccard = {mean_j:.3f}')
    ax1.axvspan(ci_lo, ci_hi, color=PAL['teal'], alpha=0.12,
                label=f'95% Bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax1.axvline(0.75, color='#444444', linestyle='--', linewidth=1.2,
                label='Stability Threshold (0.75)')
    ax1.axvline(0.60, color=PAL['vermillion'], linestyle=':', linewidth=1.2,
                label='Dissolution Threshold (0.60)')

    ax1.set_xlabel('Bootstrap Jaccard Similarity Score (4-Cluster Solution)')
    ax1.set_ylabel('Probability Density')
    ax1.set_xlim(0.3, 1.05)
    ax1.legend(loc='upper left', framealpha=0.92, fontsize=8.4)
    ax1.set_title('(a) Bootstrap Stability Distribution (4 Clusters)', loc='left', fontsize=11.0, fontweight='bold', pad=10)

    # Panel (b): Cluster-Specific Boxplots
    ax2 = fig.add_subplot(gs[1])
    data_b = [cluster_jaccards[c] for c in range(1, 5)]
    bplot = ax2.boxplot(data_b, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='#555555', linewidth=1.0),
                        capprops=dict(color='#555555', linewidth=1.0),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3, color='#888888'))

    for patch, color in zip(bplot['boxes'], [CLUSTER_COLORS[c] for c in range(1, 5)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    cluster_short_names = {
        1: 'Cluster 1\nArtisanal',
        2: 'Cluster 2\nPelagic',
        3: 'Cluster 3\nMixed Heavy',
        4: 'Cluster 4\nVulnerable'
    }

    for i, c in enumerate(range(1, 5)):
        m_val = np.mean(cluster_jaccards[c])
        ax2.plot(i + 1, m_val, 'D', color='white', markeredgecolor='black', markersize=6.5, zorder=5)
        if c == 4:
            ax2.text(i + 1 - 0.15, m_val - 0.04, f'{m_val:.2f}', va='top', ha='right', fontsize=8.4, fontweight='bold', color='#222222',
                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88))
        elif c == 3:
            ax2.text(i + 1 + 0.15, m_val - 0.035, f'{m_val:.2f}', va='top', ha='left', fontsize=8.4, fontweight='bold', color='#222222',
                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88))
        else:
            ax2.text(i + 1 + 0.15, m_val, f'{m_val:.2f}', va='center', ha='left', fontsize=8.4, fontweight='bold', color='#222222',
                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88))

    ax2.axhline(0.75, color='#444444', linestyle='--', linewidth=1.2)
    ax2.axhline(0.60, color=PAL['vermillion'], linestyle=':', linewidth=1.2)
    ax2.text(4.45, 0.77, 'Stability (0.75)', va='bottom', ha='right', fontsize=8.0, color='#444444', fontweight='bold')
    ax2.text(4.45, 0.62, 'Dissolution (0.60)', va='bottom', ha='right', fontsize=8.0, color=PAL['vermillion'], fontweight='bold')
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels([cluster_short_names[c] for c in range(1, 5)], fontsize=9.0)
    ax2.set_ylabel('Cluster-Specific Jaccard Similarity Score')
    ax2.set_ylim(0.12, 1.08)
    ax2.set_title('(b) Typology-Specific Jaccard Stability across Clusters', loc='left', fontsize=11.0, fontweight='bold', pad=10)

    # Panel (c): Multi-K Spectrum
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(k_values, k_means, 'o-', color=PAL['blue'], linewidth=2.0, markersize=7,
             label='Mean Bootstrap Jaccard')
    ax3.fill_between(k_values, k_ci_lo, k_ci_hi, color=PAL['blue'], alpha=0.18,
                     label='95% Bootstrap CI Envelope')
    ax3.axhline(0.75, color='#444444', linestyle='--', linewidth=1.2, label='Stability Threshold (0.75)')
    ax3.axhline(0.60, color=PAL['vermillion'], linestyle=':', linewidth=1.2, label='Dissolution (0.60)')

    ax3.plot(4, k_means[2], '*', color=PAL['vermillion'], markersize=14, zorder=6)
    ax3.annotate('Optimal 4-Cluster Boundary\n(Highest Stable Partition)', xy=(4, k_means[2]), xytext=(3.3, 0.45),
                 arrowprops=dict(facecolor=PAL['vermillion'], arrowstyle='->', lw=1.2),
                 fontsize=8.6, fontweight='bold', color=PAL['vermillion'],
                 bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor=PAL['vermillion'], alpha=0.92))

    ax3.set_xlabel('Candidate Number of Clusters')
    ax3.set_ylabel('Mean Bootstrap Jaccard Stability')
    ax3.set_xticks(k_values)
    ax3.set_ylim(0.35, 1.05)
    ax3.legend(loc='upper right', framealpha=0.92, fontsize=8.4)
    ax3.set_title('(c) Partition Stability Spectrum (2 to 6 Clusters)', loc='left', fontsize=11.0, fontweight='bold', pad=10)

    save_fig(fig, "FigS5_Cluster_Stability_Jaccard")

    # Fig S6: Species diversification moderation interaction & marginal effect curve (1x2 Symmetric)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.4), gridspec_kw={'wspace': 0.30})
    
    # Panel (a): Econometric Marginal Effect Curve d(Margin)/dH = 45.43 - 23.76 * CI
    ci_grid = np.linspace(0.0, 3.2, 200)
    beta1 = 45.4324
    beta2 = -23.7562
    var_beta1 = 110.3684
    var_beta2 = 49.9361
    cov_b1b2 = -47.3014

    me_h = beta1 + beta2 * ci_grid
    se_me = np.sqrt(var_beta1 + (ci_grid**2) * var_beta2 + 2 * ci_grid * cov_b1b2)
    ci_lo = me_h - 1.96 * se_me
    ci_hi = me_h + 1.96 * se_me
    ci_star = -beta1 / beta2  # 1.9124

    ax1.axhline(0, color='#666666', linestyle='-', linewidth=0.9)
    ax1.plot(ci_grid, me_h, color=PAL['blue'], linewidth=2.0, label='Marginal Effect: ∂Operating Margin / ∂Species Diversity')
    ax1.fill_between(ci_grid, ci_lo, ci_hi, color=PAL['blue'], alpha=0.18, label='95% Clustered Confidence Interval')
    ax1.axvline(ci_star, color=PAL['vermillion'], linestyle='--', linewidth=1.5,
                label=f'Zero-Crossing Threshold (CI* = {ci_star:.2f} kg CO$_2$/EUR)')

    # Annotate positive vs negative domains in open white space
    ax1.text(0.62, -22, 'Positive Diversification Return\n(Low-Carbon Fleets)',
             fontsize=8.8, color=PAL['teal'], ha='center', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.92, edgecolor=PAL['teal']))
    ax1.text(2.55, 30, 'Diversification Penalty\n(High-Carbon Fleets)',
             fontsize=8.8, color=PAL['vermillion'], ha='center', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.92, edgecolor=PAL['vermillion']))

    ax1.set_xlabel('Real Monetary Carbon Intensity (kg CO$_2$ per 2015 EUR real revenue)')
    ax1.set_ylabel('Marginal Effect of Catch Diversity\n(Percentage Points Margin per Unit Diversity)', labelpad=8)
    ax1.set_xlim(0, 3.2)
    ax1.set_ylim(-38, 68)
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.4)
    ax1.set_title('(a) Marginal Effect of Catch Diversity across Carbon Intensity', loc='left', fontsize=11.0, fontweight='bold', pad=10)

    # Panel (b): Empirical Bivariate Relationship by CI Domain (CI <= 1.91 vs CI > 1.91)
    df_div = df.dropna(subset=['shannon_h_clean', 'ebit_margin_w', 'carbon_intensity_mon_real_w']).copy()
    df_div['ci_regime'] = np.where(df_div['carbon_intensity_mon_real_w'] <= ci_star,
                                   f'Low Carbon Intensity Fleets (≤ {ci_star:.2f} kg CO2/EUR)',
                                   f'High Carbon Intensity Fleets (> {ci_star:.2f} kg CO2/EUR)')

    sns.regplot(data=df_div[df_div['carbon_intensity_mon_real_w'] <= ci_star],
                x='shannon_h_clean', y='ebit_margin_w',
                ax=ax2, color=PAL['teal'], scatter_kws={'alpha': 0.12, 's': 8}, line_kws={'linewidth': 2.0},
                label=f'Low Carbon Intensity Fleets (≤ {ci_star:.2f}): Positive Return')
    sns.regplot(data=df_div[df_div['carbon_intensity_mon_real_w'] > ci_star],
                x='shannon_h_clean', y='ebit_margin_w',
                ax=ax2, color=PAL['vermillion'], scatter_kws={'alpha': 0.12, 's': 8}, line_kws={'linewidth': 2.0},
                label=f'High Carbon Intensity Fleets (> {ci_star:.2f}): Negative Return')

    ax2.set_xlabel('Catch Species Diversity (Shannon-Wiener Index)')
    ax2.set_ylabel('Fleet Operating Profit Margin (%)')
    ax2.set_ylim(-38, 38)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.4)
    ax2.set_title('(b) Empirical Operating Margin Response across Regimes', loc='left', fontsize=11.0, fontweight='bold', pad=10)

    save_fig(fig, "FigS6_Diversification_Buffering")


def main():
    print("==================================================")
    print("GENERATING FIGURES 1-5 AND SUPPLEMENTARY FIGURES S1-S6")
    print("==================================================")
    df = load_data()

    # Generate Figures 1 to 5
    generate_figure_1(df)
    generate_figure_2()
    generate_figure_3(df)
    generate_figure_4(df)
    generate_figure_5(df)

    # Generate Supplementary Figures
    generate_supplementary_figures(df)

    print("\n==================================================")
    print("ALL FIGURES GENERATED.")
    print("==================================================")

if __name__ == '__main__':
    main()
