import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING & PREPROCESSING (Exact same logic as original)
# =============================================================================
print("Loading and preprocessing data...")
df = pd.read_excel('lentil_150.xlsx', sheet_name=0)

numeric_cols = ['DE', 'SV', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY', 'GY']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Remove outliers using IQR method (Exact same logic)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_clean = df.copy()
for col in ['GY', 'DE', 'SV', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY']:
    df_clean = remove_outliers(df_clean, col)
df_clean = df_clean.reset_index(drop=True)

# =============================================================================
# 2. FEATURE ENGINEERING (Exact same features as original)
# =============================================================================
print("Engineering features...")
# Interaction features
df_clean['PH_DF_Ratio'] = df_clean['PH'] / (df_clean['DF'] + 1)
df_clean['DM_DF_Difference'] = df_clean['DM'] - df_clean['DF']
df_clean['PH_DM_Interaction'] = df_clean['PH'] * df_clean['DM']
df_clean['DF_BP_Interaction'] = df_clean['DF'] * df_clean['BP']

# Polynomial features
df_clean['PH_Squared'] = df_clean['PH'] ** 2
df_clean['DF_Squared'] = df_clean['DF'] ** 2
df_clean['PP_Squared'] = df_clean['PP'] ** 2

new_numeric_cols = ['DE', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY',
                    'PH_DF_Ratio', 'DM_DF_Difference', 'PH_DM_Interaction', 
                    'DF_BP_Interaction', 'PH_Squared', 'DF_Squared', 'PP_Squared']

# Combine numeric and one-hot encoded categorical features
X_numeric_new = df_clean[new_numeric_cols]
X_categorical_encoded = pd.get_dummies(df_clean[['L', 'Gen']], columns=['L', 'Gen'], drop_first=False)
X = pd.concat([X_numeric_new, X_categorical_encoded], axis=1)
y = df_clean['GY']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =============================================================================
# 3. DEFINE MODELS & HYPERPARAMETERS (Exact same as original)
# =============================================================================
rf_base = RandomForestRegressor(random_state=42)
xgb_base = XGBRegressor(random_state=42, verbosity=0)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'Bayesian Ridge': BayesianRidge(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': rf_base,
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb_base,
    'SVR': SVR(),
    'Hybrid (RF+XGB)': VotingRegressor(estimators=[
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42, verbosity=0))
    ])
}

param_grids = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1]},
    'ElasticNet': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]},
    'Bayesian Ridge': {'alpha_1': [1e-6, 1e-4], 'lambda_1': [1e-6, 1e-4]},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Decision Tree': {'max_depth': [5, 10, None]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, None]},
    'Gradient Boosting': {'n_estimators': [100], 'learning_rate': [0.05, 0.1]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
    'SVR': {'C': [1, 10], 'kernel': ['rbf']},
    'Hybrid (RF+XGB)': {
        'rf__n_estimators': [100, 200],
        'xgb__learning_rate': [0.05, 0.1],
        'weights': [[1, 1], [2, 1], [1, 2]] 
    }
}

# =============================================================================
# 4. TRAIN, EVALUATE & CALCULATE MAPE
# =============================================================================
results = []
print("\nTraining models with GridSearchCV...")

for name, model in models.items():
    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # np.maximum prevents division by zero if any actual yield is exactly 0
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
    
    results.append({
        'Model': name,
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE (%)': round(mape, 2),
        'Best Params': grid_search.best_params_
    })

# =============================================================================
# 5. DISPLAY AND SAVE RESULTS
# =============================================================================
results_df = pd.DataFrame(results)
# Sort by R² (highest to lowest)
results_df = results_df.sort_values(by='R²', ascending=False).reset_index(drop=True)

print("\n" + "="*85)
print("MODEL PERFORMANCE COMPARISON (SORTED BY R²)")
print("="*85)
# Display main metrics cleanly
display_cols = ['Model', 'R²', 'RMSE', 'MAE', 'MAPE (%)']
print(results_df[display_cols].to_string(index=False))
print("="*85)

# Save full results (including best params) to CSV
results_df.to_csv('model_comparison_results_with_mape.csv', index=False)
print("\n✅ Results successfully saved to 'model_comparison_results_with_mape.csv'")