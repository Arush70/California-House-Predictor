# California House Price Prediction

This project predicts the median house value of California districts from 1990 US Census data. It combines **unsupervised market segmentation** (K-Means + KNN) with **four regressors**: Linear Regression, Random Forest, SVR and a Keras MLP. Every model is evaluated under a shared, leakage-free protocol. The best model, a Random Forest, reaches **R² 0.82 (RMSE ≈ $49k)**.

![Geographic value map](figures/geographic_value_map.png)

## Results (20% held-out test set)

| Model | R² test | RMSE test | MAE test | R² train | Train–test gap |
|---|---|---|---|---|---|
| **Random Forest** | **0.824** | **$49,376** | **$32,100** | 0.973 | 0.149 |
| MLP (Keras) | 0.806 | $51,921 | $34,485 | 0.840 | 0.034 |
| SVR (RBF) | 0.784 | $54,690 | $35,360 | 0.830 | 0.045 |
| Linear Regression | 0.660 | $68,647 | $49,707 | 0.654 | −0.006 |

The Random Forest is the most accurate model but overfits the most. The MLP comes within 0.02 R² with a much smaller generalisation gap.

![Final comparison](figures/final_comparison_charts.png)

## Key findings

- **Median income dominates:** it accounts for about 0.43 of the Random Forest's permutation importance. Inland location, household size and coordinates come next.
- **Cluster-label ablation:** adding the K-Means market segment (k = 3) as a feature changed R² by less than 0.002 for every model. The location and income features already carry that information.
- **Capped-target ablation:** 4.7% of districts sit at the $500,001 census cap. Dropping them from training *lowered* test R² for all models, so they were kept.
- **MLP architecture search:** a 3-fold CV grid over depth, width, learning rate and dropout selected `[128, 64]`, lr 1e-3, dropout 0.1 (CV RMSE ≈ $50.9k).

## Pipeline

1. **EDA:** missing values, distribution of the capped target, correlations and geographic maps.
2. **Feature engineering:** rooms per household, bedrooms per room, population per household and one-hot `ocean_proximity`.
3. **Leak-free preprocessing:** an 80/20 split *before* imputation and scaling, which are fitted on the training set only.
4. **Clustering:** K-Means is chosen with the elbow and silhouette methods and fitted on the training set. KNN assigns clusters to test rows.
5. **Regression:** 5-fold CV plus randomised or grid hyperparameter search for Linear Regression, Random Forest and SVR.
6. **MLP:** standardised inputs and target, architecture search, a regularisation comparison and early stopping.
7. **Evaluation:** residual analysis, permutation importance, learning curves and ablations.

| Residuals | Learning curves | Permutation importance |
|---|---|---|
| ![](figures/residual_analysis.png) | ![](figures/learning_curves.png) | ![](figures/permutation_importance.png) |

## Run it

```bash
git clone https://github.com/Arush70/California-House-Predictor.git
cd California-House-Predictor
conda env create -f environment.yml
conda activate ecmm422
jupyter lab notebooks/Training_Models.ipynb
```

The notebook loads `data/housing.csv` and downloads the dataset automatically if the file is missing. It also runs on Google Colab.

## Project structure

```
├── notebooks/Training_Models.ipynb  # full pipeline
├── data/housing.csv                 # California Housing (1990 Census)
├── models/                          # saved LR, SVR and MLP models
├── figures/                         # plots and result tables (CSV)
└── environment.yml
```

## Data

The California Housing dataset contains 20,640 districts. Source: [ageron/data](https://github.com/ageron/data/raw/main/housing.tgz), from *Hands-On Machine Learning* by Aurélien Géron.

## Tech stack

Python · scikit-learn · TensorFlow / Keras · pandas · NumPy · matplotlib / seaborn / plotly

## Author

**Arush Kumar Vishwakarma**, [GitHub](https://github.com/Arush70). Originally built for ECMM422 Machine Learning coursework.
