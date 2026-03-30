#!/usr/bin/env python
# coding: utf-8

# # ML Foundations Capstone Project
# ## Phase 1 — Load, Explore & Clean

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

print('All libraries imported successfully!')


# In[ ]:

#Relative path — works on any machine
raw_path = os.path.join("data", "raw", "AmesHousing.csv")
df = pd.read_csv(raw_path)

# Fix column names - remove spaces
df.columns = df.columns.str.replace(' ', '')

print('First 5 rows:')
df.head()


# In[ ]:

print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')


# In[ ]:

print(df.info())


# ### Fix Data Types
# `MSSubClass` and `MoSold` are stored as integers but are actually categorical —
# they represent category codes and should never be used in arithmetic.

# In[ ]:

df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
print('Types fixed: MSSubClass -> str, MoSold -> str')


# In[ ]:

missing = df.isnull().sum().sort_values(ascending=False)
print('Missing values:')
print(missing[missing > 0])


# ### Missing Value Strategy
# - `PoolQC`, `MiscFeature`, `Alley`, `Fence`, etc.: NaN means feature does not exist → fill with 'None'
# - `LotFrontage`: fill with median grouped by Neighborhood (similar streets → similar frontage)
# - Garage / Basement numeric columns: fill with 0 (no garage = 0 cars, 0 area)
# - `MasVnrArea`: fill with 0
# - `Electrical`: fill with mode (SBrkr = ~91% of rows)


# In[ ]:

def clean_data(df):
    """
    Cleans the raw Ames Housing DataFrame:
    - Fixes column name spaces
    - Fixes data types for MSSubClass and MoSold
    - Fills missing values with domain-appropriate values
    - Removes duplicate rows
    - Caps SalePrice outliers at the 99th percentile

    Parameters
    ----------
    df : pd.DataFrame  Raw Ames Housing data

    Returns
    -------
    pd.DataFrame  Cleaned data ready for feature engineering
    """
    df = df.copy()

    # Fix column names
    df.columns = df.columns.str.replace(' ', '')

    # Fix data types
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    # Fill categorical NaNs that mean "feature does not exist"
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Fill numeric NaNs with 0 (no garage / no basement = 0 size)
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: fill with neighborhood median
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))

    # Electrical: fill with mode
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # Remove duplicates
    df = df.drop_duplicates()

    # Cap SalePrice outliers at 99th percentile
    cap_99 = df['SalePrice'].quantile(0.99)
    df['SalePrice'] = df['SalePrice'].clip(upper=cap_99)
    print(f'SalePrice capped at 99th percentile: {cap_99:,.0f}')

    return df


# In[ ]:

df = clean_data(df)
print(f'Remaining nulls: {df.isnull().sum().sum()}')
print(f'Rows after cleaning: {len(df)}')


# In[ ]:

# Outlier visualisation
plt.figure(figsize=(8, 4))
plt.boxplot(df['SalePrice'], vert=False)
plt.title('SalePrice Boxplot (after capping at 99th percentile)')
plt.xlabel('SalePrice')
plt.tight_layout()
plt.show()


# In[ ]:

# Data quality checks
assert df['SalePrice'].isnull().sum() == 0, 'SalePrice has nulls!'
assert (df['SalePrice'] > 0).all(), 'SalePrice contains non-positive values!'
assert df.shape[1] >= 70, 'Unexpected number of columns!'
print('✅ All checks passed! Phase 1 complete.')

# Save cleaned CSV for Phase 2
os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
df.to_csv(os.path.join("data", "cleaned", "AmesHousing_cleaned.csv"), index=False)
print('✅ Cleaned CSV saved to data/cleaned/AmesHousing_cleaned.csv')
