# House Price Prediction using Gradient Boosting Machine (GBM) Regression

This project predicts house sale prices using GBM regression on the `kc_house_data.csv` dataset. The model utilizes features such as square footage, number of bedrooms, grade, and location attributes to predict prices with high accuracy.

## Steps

### Step 1: Import Libraries
Load required libraries such as `pandas`, `scipy`, `xgboost`, `matplotlib`, and `sklearn` for data processing, visualization, and modeling.

### Step 2: Data Cleaning and Preprocessing
- Check for missing values and data types.
- Remove non-predictive columns (`id`, `date`).
- Rename `price` column to `SalePrice`.

### Step 3: Correlation Analysis
Identify features with the highest correlation to `SalePrice`, such as `sqft_living`, `grade`, and `sqft_above`, to help select relevant predictors.

### Step 4: Exploratory Data Analysis (EDA)
Visualize relationships between `SalePrice` and top features using scatter and box plots for insights into data distribution.

### Step 5: Train-Test Split
Split data into training and testing sets using an 80-20 ratio for effective model evaluation.

### Step 6: Model Training with GBM
Train the model using `GradientBoostingRegressor` with key parameters like:
- **learning_rate**
- **n_estimators**
- **min_samples_leaf**

### Step 7: Model Evaluation
- **R² Score**: 0.84, indicating good predictive performance.
- **Root Mean Squared Error (RMSE)**: 130,562, giving an estimate of prediction accuracy.

## Results
The GBM model performs well in predicting house prices, with significant accuracy based on the chosen features and hyperparameters.

