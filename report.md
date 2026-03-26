# Capstone Project Report
## Ames Housing Dataset — Data Analysis Pipeline
**ML Foundations Bootcamp**

---

## 1. Introduction

**Dataset:** Ames Housing Dataset — 2,930 house sales in Ames, Iowa with 80+ features including lot size, quality ratings, neighbourhood, and sale price.

**Questions I set out to answer:**
- What factors most strongly influence house sale prices?
- How do quality and size interact to determine value?
- Are there differences between house age groups in terms of pricing?

---

## 2. Cleaning Summary

| Problem Found | How It Was Fixed |
|---|---|
| 19 columns with missing values | NaN in feature columns (PoolQC, Alley, etc.) means "not present" → filled with `'None'` or `0` |
| `LotFrontage` missing (~18%) | Filled with median value grouped by Neighborhood |
| `MSSubClass` stored as integer | Converted to string — it's a category, not a quantity |
| Extreme SalePrice outliers | Capped at 99th percentile (~$441,000) |
| Duplicate rows | Checked with `.duplicated()` — none found |
| Column names with spaces | Removed all spaces from column names for easier coding |

All cleaning steps were wrapped in a reusable `clean_data()` function, with 3 assertions to verify the output.

---

## 3. Feature Engineering Summary

| New Feature | Type | Why It's Useful |
|---|---|---|
| `price_per_sqft` | Ratio | Normalises price by living area for fair comparison |
| `total_bathrooms` | Aggregate | Combines full/half/basement bathrooms into one score |
| `qual_x_area` | Interaction | Captures the combined premium of quality × size |
| `LotArea_log` | Transform | Reduces right skew in LotArea for better analysis |
| `age_group` | Bin | Groups houses into New / Recent / Mid-Age / Old |
| `ExterQual` (encoded) | Ordinal | Converts quality text (Po→Ex) into ordered integers (1-5) |
| One-hot: `Neighborhood`, `SaleType` | Categorical | Encodes nominal categories for numerical analysis |
| `GrLivArea_scaled`, `LotArea_scaled` | Scaled | Standardised using StandardScaler for consistent ranges |

Highly correlated pairs (r > 0.95) were identified and removed to reduce redundancy.

---

## 4. Key Findings

### Finding 1 — Overall Quality is the Single Strongest Predictor
Quality-10 homes sell for an average of **4× more** than quality-4 homes. The correlation between `OverallQual` and `SalePrice` is r = 0.79 — the highest of any single feature.

**Chart Evidence:** The bar chart "Mean SalePrice by Overall Quality" shows a near-exponential increase from quality 4 ($108,000) to quality 10 ($438,000).

### Finding 2 — Size and Quality Have a Multiplier Effect
The interaction feature `qual_x_area` (quality × living area) is more correlated with SalePrice than either feature alone. A large house with low quality is not much more valuable than a small one — but a large *high-quality* house commands a significant premium.

**Chart Evidence:** The scatter plot "Living Area vs Sale Price" colored by quality shows that green points (high quality) consistently sit above the trend line, while red points (low quality) sit below, even at similar sizes.

### Finding 3 — Most High-Quality Houses Sell Above the Median
Over **90% of homes with OverallQual ≥ 8** sell above the dataset's median price (~$163,000). This confirms build quality is a reliable signal for premium pricing.

**Statistical Evidence:** P(SalePrice > median | OverallQual ≥ 8) = 0.92

---

## 5. What I Would Do Next

If I had more time, I would:

1. **Build a regression model** (e.g., Ridge Regression or XGBoost) using the engineered features to predict SalePrice and measure prediction accuracy with RMSE and R².

2. **Analyse Neighbourhood more deeply** — some neighbourhoods may carry price premiums independent of house features. A neighbourhood-level analysis could reveal location-based pricing patterns.

3. **Time-series analysis** on `YrSold` and `MoSold` to see if there are seasonal trends (e.g., summer vs winter sales) or annual trends in sale prices.

4. **Feature importance analysis** — use a tree-based model (Random Forest or XGBoost) to rank which engineered features contribute most to price predictions and validate our domain features.

5. **External data integration** — add economic indicators (interest rates, unemployment) or school district ratings to see how external factors affect house prices.

---

## Appendix: Technical Details

### Tools Used
- **Python 3.9+**
- **pandas** — data manipulation
- **numpy** — numerical computations
- **matplotlib & seaborn** — visualization
- **scikit-learn** — feature scaling and standardization

### Dataset Source
Ames Housing Dataset from Kaggle: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

### Project Structure
```
capstone/
├── capstone_complete.ipynb    # Complete analysis notebook
├── report.md                   # This report
└── data/
    └── raw/
        └── AmesHousing.csv     # Original dataset
```

---

**End of Report**
