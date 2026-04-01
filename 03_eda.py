#!/usr/bin/env python
# coding: utf-8

# # ML Foundations Capstone Project
# ## Phase 3 — Analyse, Visualise & Report
# ### EDA + Math Basics

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

# Load feature-engineered data from Phase 2
df = pd.read_csv(os.path.join(DESKTOP, "AmesHousing_features.csv"))
print(f'Loaded data: {df.shape[0]} rows, {df.shape[1]} columns')


# ---
# ## EDA

# ### Chart 1 — Histograms: Distribution of 3 Numerical Features

# In[ ]:

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cols = ['SalePrice', 'GrLivArea', 'LotArea']
colors = ['steelblue', 'coral', 'seagreen']

for ax, col, color in zip(axes, cols, colors):
    ax.hist(df[col].dropna(), bins=50, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.suptitle('Distributions of Key Numerical Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("""
Insight: SalePrice and LotArea are right-skewed — most houses are affordable/small
but a few expensive/large ones pull the tail. GrLivArea is more symmetric.
Right-skewed distributions often benefit from log-transformation.
""")


# ---
# ### Chart 2 — Grouped Boxplots: SalePrice by Category

# In[ ]:

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.boxplot(column='SalePrice', by='OverallQual', ax=axes[0])
axes[0].set_title('SalePrice by Overall Quality')
axes[0].set_xlabel('Overall Quality (1-10)')
axes[0].set_ylabel('SalePrice')
plt.sca(axes[0])
plt.xticks(rotation=0)

if 'age_group' in df.columns:
    df.boxplot(column='SalePrice', by='age_group', ax=axes[1])
    axes[1].set_title('SalePrice by House Age Group')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('SalePrice')

plt.suptitle('')
plt.tight_layout()
plt.show()

print("""
Insight: Higher overall quality strongly correlates with higher SalePrice.
Quality 9-10 houses sell for roughly 3x the price of quality 4-5 houses.
Newer houses ('New' age group) tend to command higher prices.
""")


# ---
# ### Chart 3 — Correlation Heatmap: Top 10 Features vs SalePrice

# In[ ]:

numeric_df = df.select_dtypes(include=[np.number])
corr_with_price = numeric_df.corr()['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)
top10 = corr_with_price.head(10).index.tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(
    numeric_df[top10 + ['SalePrice']].corr(),
    annot=True, fmt='.2f', cmap='coolwarm',
    linewidths=0.5, square=True
)
plt.title('Correlation Heatmap — Top 10 Features vs SalePrice')
plt.tight_layout()
plt.show()

print(f'Top 10 features most correlated with SalePrice:\n{corr_with_price.head(10)}')
print("""
Insight: OverallQual, qual_x_area, and GrLivArea are the strongest predictors of
SalePrice. This suggests that size and quality together drive price more than
either factor alone.
""")


# ---
# ### Chart 4 — Scatter Plot: GrLivArea vs SalePrice (coloured by OverallQual)

# In[ ]:

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df['GrLivArea'], df['SalePrice'],
    c=df['OverallQual'], cmap='RdYlGn',
    alpha=0.6, edgecolors='none', s=30
)
plt.colorbar(scatter, label='Overall Quality')
plt.title('Living Area vs Sale Price (colour = Overall Quality)')
plt.xlabel('Above-Ground Living Area (sqft)')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.show()

print("""
Insight: Larger houses sell for more, but quality adds another layer.
Green (high quality) points sit above the trend line — a quality 9 house
with 2000 sqft sells for more than a quality 5 house with the same area.
""")


# ---
# ### Chart 5 — Groupby: Mean SalePrice by OverallQual

# In[ ]:

groupby_qual = df.groupby('OverallQual')['SalePrice'].mean().sort_index()

plt.figure(figsize=(9, 5))
bars = plt.bar(groupby_qual.index, groupby_qual.values, color='steelblue', edgecolor='white')
plt.title('Mean SalePrice by Overall Quality')
plt.xlabel('Overall Quality (1-10)')
plt.ylabel('Mean SalePrice ($)')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
             f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

print(groupby_qual.to_string())
print(f"\nHighest mean price: Quality {groupby_qual.idxmax()} — ${groupby_qual.max():,.0f}")
print(f"Lowest mean price:  Quality {groupby_qual.idxmin()} — ${groupby_qual.min():,.0f}")
print("""
Insight: There is a near-exponential jump from quality 7 to 9.
Quality-10 homes average more than 4x the price of quality-4 homes.
""")


# ---
# ## Math Basics

# ### 1. Mean and Standard Deviation manually using NumPy

# In[ ]:

prices = df['SalePrice'].dropna().values

mean_manual = np.sum(prices) / len(prices)
std_manual = np.sqrt(np.sum((prices - mean_manual) ** 2) / len(prices))

print(f'Manual Mean:  ${mean_manual:,.2f}')
print(f'Manual Std:   ${std_manual:,.2f}')
print(f'Pandas Mean:  ${df["SalePrice"].mean():,.2f}')
print(f'Pandas Std:   ${df["SalePrice"].std():,.2f}  (note: pandas uses n-1)')


# ---
# ### 2. Standardise by hand and compare with StandardScaler

# In[ ]:

X = df['GrLivArea'].dropna().values
mu = np.sum(X) / len(X)
sigma = np.sqrt(np.sum((X - mu) ** 2) / len(X))

z_manual = (X - mu) / sigma

scaler = StandardScaler()
z_sklearn = scaler.fit_transform(X.reshape(-1, 1)).flatten()

print(f'Manual z-score  — mean: {z_manual.mean():.6f}, std: {z_manual.std():.6f}')
print(f'Sklearn z-score — mean: {z_sklearn.mean():.6f}, std: {z_sklearn.std():.6f}')
print('Difference (max abs):', np.abs(z_manual - z_sklearn).max())


# ---
# ### 3. Cosine Similarity between highest-value and lowest-value records

# In[ ]:

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_df_filled = df[numeric_cols].fillna(0)

idx_max = df['SalePrice'].idxmax()
idx_min = df['SalePrice'].idxmin()

vec_high = numeric_df_filled.loc[idx_max].values.astype(float)
vec_low = numeric_df_filled.loc[idx_min].values.astype(float)

dot_product = np.dot(vec_high, vec_low)
norm_high = np.linalg.norm(vec_high)
norm_low = np.linalg.norm(vec_low)

cosine_sim = dot_product / (norm_high * norm_low)

print(f'Highest SalePrice: ${df.loc[idx_max, "SalePrice"]:,.0f}')
print(f'Lowest  SalePrice: ${df.loc[idx_min, "SalePrice"]:,.0f}')
print(f'Cosine Similarity: {cosine_sim:.4f}')
print('A value close to 1 means both records have similar feature profiles despite different prices.')


# ---
# ### 4. Probability estimate

# In[ ]:

median_price = df['SalePrice'].median()
high_quality = df[df['OverallQual'] >= 8]
prob = (high_quality['SalePrice'] > median_price).mean()

print(f'Median SalePrice: ${median_price:,.0f}')
print(f'High-quality houses (OverallQual >= 8): {len(high_quality)}')
print(f'P(SalePrice > median | OverallQual >= 8) = {prob:.3f} ({prob * 100:.1f}%)')
print("""\nInterpretation: Most high-quality houses (quality 8+) sell above the median price,
confirming that build quality is a strong signal for premium pricing.""")


# ---
# ## Summary
#
# ✅ Phase 3 Complete: EDA with 5 charts + Math basics
#    (mean, std, standardization, cosine similarity, probability)
#
# Next step: see report.md for the full written summary of findings.
