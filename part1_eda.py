

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    print("="*50)
    print(" PEMFC Data Exploratory Data Analysis (EDA) ")
    print("="*50)

    # 0. Setup directories
    data_path = r"d:\DOWNLOAD\b_tech_project\pemfc_efficiency_clean.xlsx"
    out_dir = r"d:\DOWNLOAD\b_tech_project\outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    df = pd.read_excel(data_path, sheet_name='Clean_Data')
    
    print(f"Data shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nNull Value Check:")
    print(df.isnull().sum())
    
    print("\nDuplicate Check:")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\nOutlier Detection (IQR Method):")
    outlier_counts = {}
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    for col, count in outlier_counts.items():
        print(f"  {col}: {count} outliers")
        
    # 2. Summary Statistics
    print("\n[2] Generating Summary Statistics...")
    summary_stats = df.describe().T
    summary_stats_path = os.path.join(out_dir, 'summary_stats.csv')
    summary_stats.to_csv(summary_stats_path)
    print(f"Saved summary statistics to: {summary_stats_path}")
    
    # 3. Distribution Plots
    print("\n[3] Generating Distribution Plots...")
    cols_to_plot = [
        'current_density_mA_cm2', 'cell_voltage_V', 'cathode_pressure_diff_kPa', 
        'anode_pressure_diff_kPa', 'anode_temp_beginning_C', 'anode_temp_middle_C', 
        'anode_temp_end_C', 'cathode_temp_beginning_C', 'cathode_temp_middle_C', 
        'cathode_temp_end_C', 'voltage_efficiency_pct'
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequency')
    
    # Remove the empty 12th subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    dist_path = os.path.join(out_dir, 'distributions.png')
    plt.savefig(dist_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved distribution plots to: {dist_path}")

    # 4. Correlation Heatmap
    print("\n[4] Generating Correlation Heatmap...")
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title('Correlation Heatmap across all features', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")

    # 5. Scatter Plots
    print("\n[5] Generating Scatter Plots (Features vs Target)...")
    scatter_features = [
        'current_density_mA_cm2', 'cathode_pressure_diff_kPa', 
        'anode_pressure_diff_kPa', 'anode_avg_temp_C', 'cathode_avg_temp_C'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(scatter_features):
        if col in df.columns:
            axes[i].scatter(df[col], df['voltage_efficiency_pct'], alpha=0.3, s=5, color='royalblue')
            axes[i].set_title(f'Efficiency vs {col}', fontsize=12)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Voltage Efficiency (%)')
        
    axes[-1].axis('off')
    axes[-1].text(0.5, 0.5, 'Scatter Plots vs Target\n(Alpha=0.3, s=5)', 
                  ha='center', va='center', fontsize=14, fontweight='bold')
                  
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, 'scatter_vs_target.png')
    plt.savefig(scatter_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved scatter plots to: {scatter_path}")

    # 6. Temperature Boxplots
    print("\n[6] Generating Temperature Boxplots...")
    temp_cols = [
        'anode_temp_beginning_C', 'anode_temp_middle_C', 'anode_temp_end_C',
        'cathode_temp_beginning_C', 'cathode_temp_middle_C', 'cathode_temp_end_C'
    ]
    plt.figure(figsize=(12, 6))
    existing_temp_cols = [col for col in temp_cols if col in df.columns]
    if existing_temp_cols:
        sns.boxplot(data=df[existing_temp_cols], palette='Set3')
        plt.title('Temperature Distributions (Note narrow range ~79.7-81.5°C)', fontsize=14)
        plt.ylabel('Temperature (°C)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        box_path = os.path.join(out_dir, 'temperature_boxplots.png')
        plt.savefig(box_path, dpi=300)
        plt.show()


    # 7. Pairplot
    print("\n[7] Generating Subsampled Pairplot...")
    pairplot_features = [
        'current_density_mA_cm2', 'cathode_pressure_diff_kPa', 
        'anode_pressure_diff_kPa', 'anode_avg_temp_C', 
        'cathode_avg_temp_C', 'voltage_efficiency_pct'
    ]
    existing_pair_features = [col for col in pairplot_features if col in df.columns]
    
    if existing_pair_features:
        # Subsample 3000 rows
        df_sub = df[existing_pair_features].sample(n=min(3000, len(df)), random_state=42)
        
        g = sns.pairplot(df_sub, kind='scatter', diag_kind='kde', 
                         plot_kws={'alpha': 0.2, 's': 3})
        g.figure.suptitle('Pairplot of Recommended Features (Subsampled to 3000 rows)', y=1.02, fontsize=16)
        
        pair_path = os.path.join(out_dir, 'pairplot_features.png')
        plt.savefig(pair_path, dpi=300)
        plt.show()

    # 8. Observations
    print("\n[8] Documenting Observations...")
    observations = """EDA Observations:
- Nonlinear vs Linear Relationships: The relationship between current_density_mA_cm2 and voltage_efficiency_pct shows a distinct nonlinear, declining polarization-curve shape.
- Multicollinearity: There is severe multicollinearity between current_density and both pressure differences (cathode and anode), with correlations around 0.97-0.99.
- Temperature Predictive Power: The temperature variations across all sensors fall within a very narrow range (~2°C), meaning they likely offer very low predictive power for the target.
- Data Quality / Implausibilities: There appear to be no impossible points like negative pressures or efficiencies outside 0-100% based on the clean dataset profile, but specific bounds should be validated.
- Feature Selection Recommendation: Due to the extreme multicollinearity (>0.95), it is highly recommended to drop the pressure difference features or use dimensionality reduction, rather than keeping all 3 highly correlated features.
"""
    print(observations)
    obs_path = os.path.join(out_dir, 'eda_observations.txt')
    with open(obs_path, 'w') as f:
        f.write(observations)
    print(f"Saved observations to: {obs_path}")

    # 9. VIF Calculation
    print("\n[9] Calculating Variance Inflation Factor (VIF)...")
    vif_features = ['current_density_mA_cm2', 'cathode_pressure_diff_kPa', 
                    'anode_pressure_diff_kPa', 'anode_avg_temp_C', 'cathode_avg_temp_C']
    existing_vif_features = [col for col in vif_features if col in df.columns]
    
    if existing_vif_features:
        X_vif = df[existing_vif_features].dropna().values
        vif_values = []
        for i in range(X_vif.shape[1]):
            X_other = np.delete(X_vif, i, axis=1)
            y_col = X_vif[:, i]
            r2 = LinearRegression().fit(X_other, y_col).score(X_other, y_col)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            vif_values.append(vif)
        
        vif_data = pd.DataFrame({
            'Feature': existing_vif_features,
            'VIF': vif_values
        })
        
        print("\nVIF Results:")
        print(vif_data.to_string(index=False))
    
    print("\n" + "="*50)
    print(" EDA Script Execution Complete ")
    print("="*50)

if __name__ == "__main__":
    main()
