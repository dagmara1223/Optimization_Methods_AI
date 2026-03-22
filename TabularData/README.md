# BMW Car Price Prediction — Regression Study
Machine learning project focused on predicting BMW vehicle selling prices using classical ML models and neural networks. The notebook follows a structured, experiment-driven approach — from exploratory analysis through baseline models to deep learning optimization. Whole amount of used & trained models: 16. <br>
## Project Overview 📂 

**Dataset** :[BMW Cars Dataset](https://www.kaggle.com/datasets/wardabilal/bmw-cars-dataset-analysis-with-visualizations) <br>
**Records** :10,781 vehicles  <br>
**Task** :Regression (predict selling price)  <br>
**Best model** :XGBoost (R² = 0.957, RMSE = 2312)   <br>

## Dataset Features 
The dataset contains 9 variables describing BMW vehicle listings:

`model` · `year` · `price` · `transmission` · `mileage` · `fuelType` · `tax` · `mpg` · `engineSize`

**Engineered features**: `car_age`, `log_mileage`, `engine_category`, `efficiency`, `log_price`

## Methodology 🔧 
### 1. Exploratory Data Analysis
- Distribution analysis of target variable (price) — strong right skew → log transformation applied
- Correlation heatmap: `year`, `mileage`, and `engineSize` identified as top predictors
- Feature distributions, boxplots by fuel type and transmission

### 2. Preprocessing
- One-hot encoding of categorical variables (`model`, `transmission`, `fuelType`)
- `StandardScaler` applied for neural network inputs
- Train / Validation / Test split: **70% / 15% / 15%**

### 3. Classical ML Models (Baseline)

| Model | Test RMSE | Test R² |
|---|---|---|
| Linear Regression | ~5200 | 0.859 |
| Lasso Regression | ~5200 | 0.859 |
| Decision Tree (default) | 2998 | 0.927 |
| Random Forest (100 trees) | 2434 | 0.952 |
| **XGBoost (tuned)** | **2312** | **0.957** |

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/5163ba83-df72-4de7-8e5e-8db8f95543dc" />


### 4. Neural Network Experiments

| Model | Architecture | Test RMSE | Test R² |
|---|---|---|---|
| NN-1 | Shallow MLP (1 layer) | 2753 | 0.940 |
| NN-2 | Medium MLP (3 layers) | ~2650 | ~0.944 |
| NN-3 | L2 + Dropout | 2819 | 0.936 |
| NN-4 | BatchNorm + Dropout | ~2700 | ~0.940 |
| NN-5 | Optimized + HP search | 2943 | 0.930 |

### 5. Hyperparameter Optimization
Grid search over learning rates `[1e-2, 1e-3, 5e-4, 1e-4]` and dropout rates `[0.2, 0.3, 0.4]`, evaluated via validation loss heatmap.

### 6. Cross-Validation
5-Fold CV on tree-based models to confirm generalization and stability.

## Key Findings

- **XGBoost outperforms all models** — tree-based ensembles are generally superior on structured tabular data, consistent with ML literature
- **Log transformation** of the target variable stabilizes training and reduces the effect of high-price outliers
- **Dropout behavior**: when Train Loss > Val Loss, this does NOT indicate underfitting — it's an expected artifact of Dropout randomly disabling neurons during training
- **Deeper ≠ better** for tabular data: NN-2 (medium MLP) outperformed both the deeper NN-3 and the heavily regularized NN-5
- **Efficiency, year, and car_age** are the most important predictors of vehicle price
