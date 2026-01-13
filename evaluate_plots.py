import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# GLOBAL PLOT CONFIGURATION
plt.rcParams.update({
    'font.size': 20,             # Base font size
    'axes.titlesize': 30,         # Title font size
    'axes.labelsize': 20,         # Axis label font size
    'xtick.labelsize': 20,        # X-axis tick label size
    'ytick.labelsize': 15,        # Y-axis tick label size
    'legend.fontsize': 12,        # Legend font size
})

# Load the data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Metrics", "colorization_evaluation_report.csv")
df = pd.read_csv(csv_path)

# Define metric categories
mae_cols = [col for col in df.columns if 'MAE' in col]  # Lower is better
lpips_cols = [col for col in df.columns if 'LPIPS' in col]  # Lower is better
emd_cols = [col for col in df.columns if 'EMD' in col]  # Lower is better

# === 1. SUMMARY STATISTICS ===
summary = df.describe().loc[['mean', 'std', 'min', 'max']].T
summary = summary.rename(columns={'mean': 'Average', 'std': 'Std Dev', 'min': 'Minimum', 'max': 'Maximum'})

# === 2. SEPARATE BEST/WORST RANKINGS ===
def get_extremes_by_category(df, columns, lower_is_better=True):
    """Returns best/worst images separately for a given metric category."""
    results = {}
    for col in columns:
        best_idx = df[col].idxmin() if lower_is_better else df[col].idxmax()
        worst_idx = df[col].idxmax() if lower_is_better else df[col].idxmin()
        results[col] = pd.Series({
            'Best Image': df.loc[best_idx, 'filename'],
            'Best Score': df.loc[best_idx, col],
            'Worst Image': df.loc[worst_idx, 'filename'],
            'Worst Score': df.loc[worst_idx, col]
        })
    return pd.DataFrame(results).T

# Compute best/worst separately
extremes_mae = get_extremes_by_category(df, mae_cols, lower_is_better=True)
extremes_lpips = get_extremes_by_category(df, lpips_cols, lower_is_better=True)
extremes_emd = get_extremes_by_category(df, emd_cols, lower_is_better=True)

# === 3. BEST MODEL SELECTION (FIXED VERSION) ===
def normalize_scores(scores):
    """Normalize scores to 0-1 range where lower is better"""
    min_val = min(scores.values())
    max_val = max(scores.values())
    return {model: (score - min_val)/(max_val - min_val) for model, score in scores.items()}

# Calculate average scores for each model
model_scores = {}
for model in ["LA", "LB", "LAB", "LA+LB"]:
    model_scores[model] = {
        'lpips': df[f"{model}_LPIPS"].mean(),
        'mae': df[f"{model}_MAE"].mean(),
        'emd': df[f"{model}_EMD"].mean()
    }

# Normalize each metric across models
lpips_scores = {model: vals['lpips'] for model, vals in model_scores.items()}
mae_scores = {model: vals['mae'] for model, vals in model_scores.items()}
emd_scores = {model: vals['emd'] for model, vals in model_scores.items()}

norm_lpips = normalize_scores(lpips_scores)
norm_mae = normalize_scores(mae_scores)
norm_emd = normalize_scores(emd_scores)

# Calculate weighted scores (weights can be adjusted)
weighted_scores = {
    model: 0.6 * norm_lpips[model] + 0.3 * norm_mae[model] + 0.1 * norm_emd[model]
    for model in model_scores.keys()
}

best_model = min(weighted_scores, key=weighted_scores.get)


def get_closest_images(df, model):
    lpips_scores = df[f"{model}_LPIPS"]
    avg_score = lpips_scores.mean()

    # Get indices of images sorted by closeness to the average score
    sorted_indices = (lpips_scores - avg_score).abs().argsort()

    # Get the first and second closest images (fallback to one if only one exists)
    first_closest_image = df.iloc[sorted_indices[0]]["filename"]
    second_closest_image = df.iloc[sorted_indices[1]]["filename"] if len(sorted_indices) > 1 else first_closest_image

    return first_closest_image, second_closest_image


# Print example image filenames for each model
for model in ["LA", "LB", "LAB", "LA+LB"]:
    first_avg_img, second_avg_img = get_closest_images(df, model)
    print(f"{model} Closest Average LPIPS Score Image: {first_avg_img}")
    print(f"{model} Second Closest Average LPIPS Score Image: {second_avg_img}")
    plt.show()

# === 4. PRINT ALL RESULTS ===
print("\n=== KEY METRICS SUMMARY ===")
print(summary.round(3))

print("\n=== BEST/WORST FOR MAE ===")
print(extremes_mae.round(3))

print("\n=== BEST/WORST FOR LPIPS ===")
print(extremes_lpips.round(3))

print("\n=== BEST/WORST FOR EMD ===")
print(extremes_emd.round(3))

print("\n=== BEST MODEL BASED ON WEIGHTED SCORE ===")
print("Metric Averages:")
for model, scores in model_scores.items():
    print(f"{model}: LPIPS={scores['lpips']:.4f}, MAE={scores['mae']:.4f}, EMD={scores['emd']:.4f}")

print("\nNormalized Scores (0-1, lower is better):")
for model in model_scores.keys():
    print(f"{model}: LPIPS={norm_lpips[model]:.4f}, MAE={norm_mae[model]:.4f}, EMD={norm_emd[model]:.4f}")

print("\nWeighted Scores:")
for model, score in weighted_scores.items():
    print(f"{model}: {score:.4f}")
print(f"\n🏆 **The Best Model is: {best_model}** 🏆")

# === 5. VISUALIZATIONS ===
# Helper function to clean labels
def clean_labels(ax):
    labels = [label.get_text().split('_')[0] for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

# MAE Bar Plot
plt.figure(figsize=(10, 8))
mae_plot = sns.barplot(x=summary.loc[mae_cols].index, y=summary.loc[mae_cols, "Average"], palette="Blues")
mae_plot.set_title("Average MAE Scores by Model (Lower is better)", fontsize=25)
mae_plot.set_ylabel("MAE Score", fontsize=20)
mae_plot.set_xlabel("Model", fontsize=20)
mae_plot.set_xticklabels(mae_plot.get_xticklabels(), rotation=45)
clean_labels(mae_plot)
plt.tight_layout()
plt.show()

# LPIPS Box-and-Whisker Plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[lpips_cols], palette="Set3", linewidth=2.5)  # Increased line thickness
plt.title("LPIPS Score Distribution Across Models (Lower is better)", fontsize=25)
plt.ylabel("LPIPS Score", fontsize=20)
plt.xlabel("Model", fontsize=20)
plt.xticks(ticks=range(len(lpips_cols)), labels=[col.split('_')[0] for col in lpips_cols], rotation=45)
plt.tight_layout()
plt.show()

# LPIPS Bar Plot
plt.figure(figsize=(10, 8))
lpips_plot = sns.barplot(x=summary.loc[lpips_cols].index, y=summary.loc[lpips_cols, "Average"], palette="Greens")
lpips_plot.set_title("Average LPIPS Scores by Model (Lower is better)", fontsize=25)
lpips_plot.set_ylabel("LPIPS Score", fontsize=20)
lpips_plot.set_xlabel("Model", fontsize=20)
lpips_plot.set_xticklabels(lpips_plot.get_xticklabels(), rotation=45)
clean_labels(lpips_plot)
plt.tight_layout()
plt.show()

# EMD Bar Plot
plt.figure(figsize=(10, 8))
emd_plot = sns.barplot(x=summary.loc[emd_cols].index, y=summary.loc[emd_cols, "Average"], palette="Reds")
emd_plot.set_title("Average EMD Scores by Model (Lower is better)", fontsize=25)
emd_plot.set_ylabel("EMD Score", fontsize=20)
emd_plot.set_xlabel("Model", fontsize=20)
emd_plot.set_xticklabels(emd_plot.get_xticklabels(), rotation=45)
clean_labels(emd_plot)
plt.tight_layout()
plt.show()

# LPIPS KDE
plt.figure(figsize=(10, 8))
for model in lpips_cols:
    sns.kdeplot(df[model], label=model.split('_')[0], common_norm=False)
plt.title("LPIPS Score Distribution (Lower is better)", fontsize=25)
plt.xlabel("LPIPS Score", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(title="Models", fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()

# MAE KDE
plt.figure(figsize=(10, 8))
for model in mae_cols:
    sns.kdeplot(df[model], label=model.split('_')[0], common_norm=False)
plt.title("MAE Score Distribution (Lower is better)", fontsize=25)
plt.xlabel("MAE Score", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(title="Models", fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()

# EMD KDE
plt.figure(figsize=(10, 8))
for model in emd_cols:
    sns.kdeplot(df[model], label=model.split('_')[0], common_norm=False)
plt.title("EMD Score Distribution (Lower is better)", fontsize=25)
plt.xlabel("EMD Score", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(title="Models", fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.filter(items=mae_cols + lpips_cols + emd_cols).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
            annot_kws={"fontsize":10}, cbar_kws={'label': 'Correlation'})
plt.title("Metric Correlations", fontsize=16)
plt.tight_layout()
plt.show()

# MAE vs LPIPS Scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df["LAB_MAE"], y=df["LAB_LPIPS"])
plt.title("LAB: MAE vs LPIPS Correlation", fontsize=16)
plt.xlabel("LAB_MAE", fontsize=14)
plt.ylabel("LAB_LPIPS", fontsize=14)
plt.tight_layout()
plt.show()

# Weighted Scores Bar Plot
plt.figure(figsize=(10, 8))
sns.barplot(x=list(weighted_scores.keys()), y=list(weighted_scores.values()), palette="coolwarm")
plt.title("Overall Model Score (Lower is Better)", fontsize=25)
plt.ylabel("Weighted Score", fontsize=20)
plt.xlabel("Model", fontsize=20)
plt.tight_layout()
plt.show()

output_excel_path = os.path.join(BASE_DIR, "Metrics", "enhanced_report.xlsx")
with pd.ExcelWriter(output_excel_path) as writer:
    summary.round(3).to_excel(writer, sheet_name='Summary')
    extremes_mae.round(3).to_excel(writer, sheet_name='Best_Worst_MAE')
    extremes_lpips.round(3).to_excel(writer, sheet_name='Best_Worst_LPIPS')
    extremes_emd.round(3).to_excel(writer, sheet_name='Best_Worst_EMD')
    df.describe().round(3).to_excel(writer, sheet_name='Full Stats')
    pd.DataFrame.from_dict(weighted_scores, orient='index', columns=['Weighted Score']).to_excel(writer, sheet_name='Model_Scores')

print("\nEnhanced report saved as 'enhanced_report.xlsx'.")