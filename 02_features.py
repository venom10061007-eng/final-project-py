#!/usr/bin/env python
# coding: utf-8

# # ML Foundations Capstone Project
# ## Phase 2 — Engineer & Transform Features

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

# Load cleaned data from Phase 1
df = pd.read_csv(os.path.join(DESKTOP, "AmesHousing_cleaned.csv"))
print(f'Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns')


# ---
# ### 1. One-Hot Encoding
# **Why:** `Neighborhood` and `SaleType` are nominal categories with no natural order.
# One-hot encoding lets the model treat each value independently without implying rank.

# In[ ]:

df = pd.get_dummies(df, columns=['Neighborhood', 'SaleType'], drop_first=True)
print(f'Shape after one-hot encoding: {df.shape}')


# ---
# ### 2. Ordinal Encoding
# **Why:** Quality columns have a clear order (Poor < Fair < Typical < Good < Excellent).
# Mapping them to integers 0–5 captures this ordering and allows numeric comparisons.
# 'None' = 0 means the feature does not exist (e.g. no basement).

# In[ ]:

# ✅ 'None' key added — BsmtQual was filled with 'None' in Phase 1
#    so the map must include it, otherwise those rows become NaN silently.
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'KitchenQual', 'HeatingQC']
for col in ordinal_cols:
    if col in df.columns:
        df[col] = df[col].map(qual_map).fillna(0).astype(int)
print('Ordinal encoding applied to quality columns.')


# ---
# ### 3. StandardScaler
# **Why:** `GrLivArea` and `LotArea` are on very different scales (hundreds vs tens of thousands).
# Scaling puts them on the same footing so distance-based models are not biased toward larger numbers.

# In[ ]:

scaler = StandardScaler()
scale_cols = ['GrLivArea', 'LotArea']
df[['GrLivArea_scaled', 'LotArea_scaled']] = scaler.fit_transform(df[scale_cols])
print('Scaled GrLivArea and LotArea.')


# ---
# ### 4. Domain Features
# **Why price_per_sqft:** Normalises price by size so houses of different sizes can be
# compared fairly — a $300k house with 1000 sqft is very different from one with 3000 sqft.
#
# **Why total_bathrooms:** Buyers value total bathroom count. Combining all bathroom types
# into one weighted score (full=1, half=0.5) captures that signal in a single column.

# In[ ]:

df['price_per_sqft'] = df['SalePrice'] / df['GrLivArea'].replace(0, np.nan)

# ✅ FIX: df.get() does not work on a DataFrame (it is a dict method).
#    Use .fillna(0) on the actual columns instead.
df['total_bathrooms'] = (
    df['FullBath'] +
    0.5 * df['HalfBath'] +
    df['BsmtFullBath'].fillna(0) +
    0.5 * df['BsmtHalfBath'].fillna(0)
)

print('Created: price_per_sqft, total_bathrooms')


# ---
# ### 5. Interaction Feature
# **Why:** Overall quality and living area together explain more variance than either alone.
# A large low-quality house and a small high-quality house may have similar prices —
# the product captures this combined signal in one feature.

# In[ ]:

df['qual_x_area'] = df['OverallQual'] * df['GrLivArea']
print('Created interaction feature: qual_x_area')


# ---
# ### 6. Log-transform a skewed column
# **Why:** `LotArea` is heavily right-skewed (a few enormous lots distort the distribution).
# log1p compresses the tail and produces a more symmetric distribution that works better
# with linear models and distance metrics.

# In[ ]:

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['LotArea'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('LotArea — Before Log Transform')
axes[0].set_xlabel('LotArea')
axes[0].set_ylabel('Count')

df['LotArea_log'] = np.log1p(df['LotArea'])

axes[1].hist(df['LotArea_log'], bins=50, color='coral', edgecolor='white')
axes[1].set_title('LotArea — After Log Transform')
axes[1].set_xlabel('log(LotArea + 1)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
print('Log-transformed LotArea -> LotArea_log')


# ---
# ### 7. Bin a column into groups
# **Why:** Raw `YearBuilt` is noisy. Converting to age groups (New / Recent / Mid-Age / Old)
# reduces noise and makes the age effect easier to visualise and interpret in charts.

# In[ ]:

current_year = 2010
df['house_age'] = current_year - df['YearBuilt']

df['age_group'] = pd.cut(
    df['house_age'],
    bins=[-1, 10, 30, 60, 200],
    labels=['New', 'Recent', 'Mid-Age', 'Old']
)
print('Age groups created:')
print(df['age_group'].value_counts())


# ---
# ### 8. Remove highly correlated features (r > 0.95)
# **Why:** Highly correlated features carry redundant information and can cause
# multicollinearity issues in linear models. Dropping one of each pair keeps the
# dataset lean without losing predictive power.

# In[ ]:

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

print(f'Columns to drop (r > 0.95): {to_drop}')
df = df.drop(columns=to_drop, errors='ignore')
print(f'✅ Phase 2 complete. Shape: {df.shape}')

# Save engineered data for Phase 3
features_path = os.path.join(DESKTOP, "AmesHousing_features.csv")
df.to_csv(features_path, index=False)
print(f'✅ Feature-engineered CSV saved to: {features_path}')
