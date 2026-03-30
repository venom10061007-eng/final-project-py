# ML Foundations Capstone — Report
## Ames Housing Dataset Analysis

---

## 1. Introduction

**Dataset used:** Ames Housing Dataset — 2,930 house sales in Ames, Iowa, with 80+ features
covering lot size, overall quality, neighbourhood, garage details, basement specs, and sale price.

**Questions I aimed to answer:**
- What features most strongly predict house sale price?
- Does build quality outweigh size when determining price?
- Are newer houses significantly more expensive than older ones?

---

## 2. Cleaning Summary

| Problem Found | How It Was Fixed |
|---|---|
| Hard-coded absolute file path | Replaced with `os.path.join("data", "raw", ...)` |
| `MSSubClass` and `MoSold` stored as integers | Cast to `str` — they are category codes, not numbers |
| 14 columns had NaN meaning "no feature" (e.g. no pool) | Filled with `'None'` string |
| Garage and basement numeric columns had NaN | Filled with `0` (no garage = 0 cars, 0 area) |
| `LotFrontage` had ~17% missing | Filled with **median grouped by Neighborhood** — similar streets have similar frontage |
| `Electrical` had 1 missing row | Filled with mode (`SBrkr` = 91% of rows) |
| `BsmtQual` filled with `'None'` but ordinal map had no `'None'` key | Added `'None': 0` to `qual_map` |
| `df.get()` used on DataFrame (only works on dict) | Replaced with `.fillna(0)` on the actual columns |
| Outliers in `SalePrice` | Capped at 99th percentile ($475,000) |
| All steps were inline with no reusable function | Wrapped everything in `clean_data()` |

All cleaning steps were encapsulated in a `clean_data()` function that can be called on the
raw CSV to reproduce the cleaned dataset at any time.

---

## 3. Feature Engineering Summary

| Feature | Type | Why It's Useful |
|---|---|---|
| `Neighborhood_*`, `SaleType_*` | One-hot encoding | Nominal categories with no natural order — each level needs its own indicator |
| `ExterQual`, `BsmtQual`, etc. | Ordinal encoding (0–5) | Ordered quality scale → captures rank without implying equal spacing |
| `GrLivArea_scaled`, `LotArea_scaled` | StandardScaler | Removes scale bias so models treat both features fairly |
| `price_per_sqft` | Domain ratio | Normalises price by size for fair cross-house comparison |
| `total_bathrooms` | Domain composite | Combines all bathroom types into one weighted count (full=1, half=0.5) |
| `qual_x_area` | Interaction | Quality × area captures the compounding effect of size + build quality |
| `LotArea_log` | Log transform | Compresses heavy right skew in `LotArea` for better model compatibility |
| `age_group` | Binning | Groups `YearBuilt` into New / Recent / Mid-Age / Old for clearer visualisation |
| Highly correlated pairs (r > 0.95) | Dropped | Removes redundant columns that add noise without new information |

---

## 4. Key Findings

### Finding 1 — Overall Quality is the Strongest Single Predictor
The correlation heatmap shows `OverallQual` has the highest correlation with `SalePrice` (~0.80).
Quality-10 homes average **more than 4× the price** of quality-4 homes. Even when controlling for
size, high-quality houses consistently sell above the trend line.

### Finding 2 — Size and Quality Multiply, Not Add
The interaction feature `qual_x_area` (Quality × Living Area) is more correlated with price than
either variable alone. This means a large *and* high-quality house is worth disproportionately more
— the relationship is multiplicative, not just additive.

*(Best chart: Scatter plot — Living Area vs Sale Price, coloured by Overall Quality.
Green high-quality points sit clearly above the regression trend.)*

![Chart 4: Scatter plot of GrLivArea vs SalePrice coloured by OverallQual]

### Finding 3 — Newer Houses Command a Significant Premium
The grouped boxplot of `SalePrice` by `age_group` shows that **New** houses (built within 10 years
of 2010) have a noticeably higher median price and a tighter distribution than **Old** houses (60+
years). However, high-quality old houses can still compete with average newer ones, confirming that
quality matters more than age alone.

---

## 5. What I Would Do Next

1. **Train a baseline regression model** (e.g. Ridge or Random Forest) using the engineered features
   to move from exploration to prediction and quantify feature importances.
2. **Log-transform `SalePrice`** before modelling — its right-skewed distribution means a model
   trained on raw prices will over-predict expensive houses.
3. **Explore neighbourhood-level effects more deeply** — the one-hot encoded neighbourhood
   columns likely capture micro-market pricing that the current EDA only scratches the surface of.
4. **Cross-validate the probability estimates** using k-fold to confirm that
   P(SalePrice > median | OverallQual ≥ 8) ≈ 90% is not a sampling artefact.

---

*Report written as part of the ML Foundations Bootcamp Capstone Project.*
