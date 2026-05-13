# =============================================================================
# LENTIL YIELD PREDICTION & STABILITY ANALYSIS PIPELINE
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.base import clone
import umap
import shap
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath='lentil_150_11_05_26.xlsx'):
    """Load and preprocess the lentil dataset with One-Hot Encoding"""
    print("="*80)
    print("STEP 1: DATA LOADING & PREPROCESSING (ONE-HOT ENCODING)")
    print("="*80)
    
    # Load data
    df = pd.read_excel(filepath, sheet_name=0)
        
    # Location mapping
    location_mapping = {
        'L1': 'Bashontopur',
        'L2': 'Shyampur',
        'L3': 'Nachole',
        'L4': 'Sapaher',
        'L5': 'Lalpur'
    }
    
    # Add location names
    df['Location_Name'] = df['L'].map(location_mapping)
    
    # Handle missing values
    numeric_cols = ['DE', 'SV', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY', 'GY']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with median
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    df_clean = df.copy()
    for col in ['GY','DE', 'SV', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY']:
        df_clean = remove_outliers(df_clean, col)

    df_clean = df_clean.reset_index(drop=True)       
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} outliers")
    
    # Categorical columns for One-Hot Encoding
    categorical_cols = ['L', 'Gen']
    
    # Numerical feature columns
    numeric_feature_cols = ['DE', 'SV', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY']
    
    target_col = 'GY'
    
    # Separate features and target
    X_numeric = df_clean[numeric_feature_cols].copy()
    X_categorical = df_clean[categorical_cols].copy()
    y = df_clean[target_col]
    
    # Apply One-Hot Encoding
    print("\n🔧 Applying One-Hot Encoding...")
    X_categorical_encoded = pd.get_dummies(X_categorical, columns=['L', 'Gen'], drop_first=False)
    
    # Combine numeric and encoded categorical features
    X = pd.concat([X_numeric, X_categorical_encoded], axis=1)
    
    # Store metadata
    metadata = {
        'df_clean': df_clean,
        'numeric_feature_cols': numeric_feature_cols,
        'categorical_cols': categorical_cols,
        'encoded_feature_cols': X.columns.tolist(),
        'target_col': target_col,
        'location_mapping': location_mapping,
        'X_categorical_encoded': X_categorical_encoded,
        'dropped_columns': ['Rep', 'Treat']
    }
    
    print(f"\n📊 Feature Information:")
    print(f"  Numeric features: {len(numeric_feature_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  One-Hot encoded features: {X_categorical_encoded.shape[1]}")
    print(f"  Total features after encoding: {X.shape[1]}")
    print(f"  Dropped columns (low correlation): {metadata['dropped_columns']}")
    
    # Basic statistics
    print("\n📊 Basic Statistics:")
    print(df_clean[numeric_cols].describe().round(2))
    
    # Show correlation with target
    print("\n📈 Correlation with Target (GY):")
    corr_df = df_clean[numeric_feature_cols + ['GY']].corr()['GY'].sort_values(ascending=False)
    print(corr_df.round(3))
    
    return X, y, metadata

# =============================================================================
# STEP 2.1: VISUALIZATION - CORRELATION HEATMAP
# =============================================================================

def plot_feature_distributions_and_correlation(X_scaled, metadata, save_plots=False):
    """
    Generates histograms and correlation heatmap.
    FIX: Reconstructs the full dataset to include engineered features for plotting.
    """
    print("\n" + "="*80)
    print("STEP 3: VISUALIZATION - HISTOGRAMS & CORRELATION HEATMAP")
    print("="*80)
    
    df_clean = metadata['df_clean']
    target_col = metadata['target_col']
    scaler = metadata.get('scaler')
    
    # -------------------------------------------------------------------------
    # FIX: Reconstruct a dataframe with ALL features (including engineered ones)
    # -------------------------------------------------------------------------
    try:
        # 1. Inverse transform X_scaled to get features back to original scale
        # Note: X_scaled contains both numeric and one-hot encoded features
        X_original_array = scaler.inverse_transform(X_scaled)
        
        # 2. Create a DataFrame with these features
        df_features = pd.DataFrame(X_original_array, 
                                   columns=X_scaled.columns, 
                                   index=df_clean.index)
        
        # 3. Add the Target variable (GY) which is in df_clean
        if target_col in df_clean.columns:
            df_features[target_col] = df_clean[target_col]
            
        # Use this combined dataframe for plotting
        df_to_plot = df_features

        # We look for columns that are NOT categorical dummies
        all_cols = df_to_plot.columns.tolist()
        categorical_dummy_cols = [c for c in all_cols if c.startswith('L_') or c.startswith('Gen_')]
        
        # Filter numeric features
        numeric_cols_for_hist = [c for c in all_cols if c not in categorical_dummy_cols and c != target_col]
        
    except Exception as e:
        print(f"⚠️ Warning: Could not reconstruct feature data ({e}).")
        print("   Falling back to original dataframe (engineered features might be missing).")
        df_to_plot = df_clean
        numeric_cols_for_hist = metadata['numeric_feature_cols']
        categorical_dummy_cols = []


    # -------------------------------------------------------------------------
    # PLOT 1: Correlation Heatmap of All Features
    # -------------------------------------------------------------------------
    print("\n🔥 Generating correlation heatmap...")
    
    # Select columns for heatmap (Numeric + Target)
    # Exclude One-Hot Encoded columns to keep the heatmap readable
    heatmap_cols = [c for c in numeric_cols_for_hist if c not in categorical_dummy_cols] + [target_col]
    
    # If too many features, select only the top correlated ones to prevent clutter
    if len(heatmap_cols) > 25:
        print(f"  ⚠️  {len(heatmap_cols)} features detected - showing top 20 correlated with {target_col}")
        corr_with_target = df_to_plot[numeric_cols_for_hist].corrwith(df_to_plot[target_col]).abs().sort_values(ascending=False)
        top_features = corr_with_target.head(24).index.tolist()
        heatmap_cols = top_features + [target_col]
    
    if len(heatmap_cols) > 1:
        corr_matrix = df_to_plot[heatmap_cols].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f', 
                    cmap='viridis', 
                    center=0,
                    square=True, 
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Correlation"},
                    annot_kws={"size": 8})
        
        plt.title('Feature Correlation Heatmap (with Engineered Features)', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('feature_correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
            print("  ✓ Saved: feature_correlation_heatmap_full.png")
        plt.show()
    else:
        print("  ⚠️ Not enough data to generate heatmap.")

    print("\n✅ Visualization step completed!")
    return None



# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def feature_engineering(X, y, metadata):
    """Create new features, drop low correlation features before scaling"""
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    df = metadata['df_clean'].copy()
    
    # Create interaction features (EXCLUDING low correlation ones)
    df['PH_DF_Ratio'] = df['PH'] / (df['DF'] + 1)
    df['DM_DF_Difference'] = df['DM'] - df['DF']
    df['PH_DM_Interaction'] = df['PH'] * df['DM']
    df['DF_BP_Interaction'] = df['DF'] * df['BP']
    
    # Polynomial features
    df['PH_Squared'] = df['PH'] ** 2
    df['DF_Squared'] = df['DF'] ** 2
    df['PP_Squared'] = df['PP'] ** 2
    
    # Updated numeric feature columns (excluding low correlation)
    new_numeric_cols = ['DE', 'PH', 'DF', 'DM', 'BP', 'PP', 'SP', 'TSW', 'BY',
                       'PH_DF_Ratio', 'DM_DF_Difference', 
                       'PH_DM_Interaction', 'DF_BP_Interaction',
                       'PH_Squared', 'DF_Squared', 'PP_Squared']
    
    # Columns dropped due to low correlation
    dropped_engineered_cols = ['PP_BP_Ratio', 'SP_PP_Product']  # These were found to have very low correlation with GY
    
    # Combine with one-hot encoded categorical features
    X_numeric_new = df[new_numeric_cols]
    X_categorical_encoded = metadata['X_categorical_encoded']
    
    # Combine all features
    X_new = pd.concat([X_numeric_new, X_categorical_encoded], axis=1)
    
    # Show correlation analysis
    print("\n📈 Correlation Analysis - New Engineered Features with GY:")
    engineered_features = ['PH_DF_Ratio', 'DM_DF_Difference', 'PH_DM_Interaction', 
                          'DF_BP_Interaction', 'PH_Squared', 'DF_Squared', 'PP_Squared']
    
    corr_with_gy = df[engineered_features + ['GY']].corr()['GY'].sort_values(ascending=False)
    print(corr_with_gy.round(3))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_new)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X_new.columns)
    
    print(f"\n📊 Feature Summary:")
    print(f"  Original numeric features: {len(metadata['numeric_feature_cols'])}")
    print(f"  Engineered numeric features: {len(new_numeric_cols)}")
    print(f"  One-Hot encoded categorical features: {X_categorical_encoded.shape[1]}")
    print(f"  Total features after engineering: {X_scaled.shape[1]}")
    print(f"  Dropped columns (low correlation): {metadata['dropped_columns'] + dropped_engineered_cols}")
    
    # Feature correlation with target (top 15)
    print("\n📈 Top 15 Feature Correlations with Target (GY):")
    all_features_corr = df[new_numeric_cols].corrwith(df['GY']).sort_values(ascending=False)
    print(all_features_corr.head(15).round(3))
    
    metadata['scaler'] = scaler
    metadata['new_numeric_cols'] = new_numeric_cols
    metadata['encoded_feature_cols'] = X_scaled.columns.tolist()
    metadata['dropped_engineered_cols'] = dropped_engineered_cols
    
    return X_scaled, y, metadata

# =============================================================================
# STEP 3: MANIFOLD LEARNING (UMAP & t-SNE)
# =============================================================================

def manifold_learning(X_scaled, y, metadata):
    """Apply UMAP and t-SNE for dimensionality reduction (2D & 3D)"""
    print("\n" + "="*80)
    print("STEP 3: MANIFOLD LEARNING (UMAP & t-SNE) - 2D & 3D")
    print("="*80)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    df = metadata['df_clean'].copy()
    
    
    # ========== 3D EMBEDDINGS ==========
    print("\n🔵 Applying UMAP (3D)...")
    umap_model_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_embedding_3d = umap_model_3d.fit_transform(X_scaled)
    
    print("🔴 Applying t-SNE (3D)...")
    tsne_model_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_embedding_3d = tsne_model_3d.fit_transform(X_scaled)
    
    
    # ========== 3D VISUALIZATION ==========
    print("\n🎨 Creating 3D Visualizations...")
    
    # UMAP 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter_3d = ax.scatter(umap_embedding_3d[:, 0], umap_embedding_3d[:, 1], 
                           umap_embedding_3d[:, 2], c=y, cmap='viridis', 
                           alpha=0.7, s=50, depthshade=True)
    
    ax.set_title('UMAP 3D Embedding (Colored by Grain Yield)', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    ax.set_zlabel('UMAP Component 3', fontsize=12)
    ax.view_init(elev=20, azim=45)
    plt.colorbar(scatter_3d, ax=ax, label='Grain Yield', shrink=0.6)
    plt.savefig('manifold_umap_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter_3d = ax.scatter(tsne_embedding_3d[:, 0], tsne_embedding_3d[:, 1], 
                           tsne_embedding_3d[:, 2], c=y, cmap='viridis', 
                           alpha=0.7, s=50, depthshade=True)
    
    ax.set_title('t-SNE 3D Embedding (Colored by Grain Yield)', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12)
    ax.view_init(elev=20, azim=45)
    plt.colorbar(scatter_3d, ax=ax, label='Grain Yield', shrink=0.6)
    plt.savefig('manifold_tsne_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 3D BY LOCATION ==========
    print("\n📍 Creating 3D Location-Based Visualizations...")
    
    location_colors = ['red', 'blue', 'green', 'orange', 'purple']
    locations = df['L'].unique()
    
    # UMAP 3D by Location
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, loc in enumerate(locations):
        mask = df['L'] == loc
        ax.scatter(umap_embedding_3d[mask, 0], umap_embedding_3d[mask, 1], 
                  umap_embedding_3d[mask, 2], c=location_colors[i], 
                  label=loc, alpha=0.7, s=50, depthshade=True)
    
    ax.set_title('UMAP 3D by Location', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    ax.set_zlabel('UMAP Component 3', fontsize=12)
    ax.view_init(elev=20, azim=45)
    ax.legend()
    plt.savefig('manifold_umap_3d_location.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE 3D by Location
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, loc in enumerate(locations):
        mask = df['L'] == loc
        ax.scatter(tsne_embedding_3d[mask, 0], tsne_embedding_3d[mask, 1], 
                  tsne_embedding_3d[mask, 2], c=location_colors[i], 
                  label=loc, alpha=0.7, s=50, depthshade=True)
    
    ax.set_title('t-SNE 3D by Location', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12)
    ax.view_init(elev=20, azim=45)
    ax.legend()
    plt.savefig('manifold_tsne_3d_location.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== STORE EMBEDDINGS ==========

    metadata['umap_embedding_3d'] = umap_embedding_3d
    metadata['tsne_embedding_3d'] = tsne_embedding_3d
    
    print("\n✅ Manifold learning completed!")
    print(f"UMAP 3D shape: {umap_embedding_3d.shape}")
    print(f"t-SNE 3D shape: {tsne_embedding_3d.shape}")
    
    return metadata

# =============================================================================
# STEP 4: MACHINE LEARNING MODELS (WITH HYPERPARAMETER TUNING)
# =============================================================================

def train_ml_models(X_scaled, y, metadata):
    """Train multiple ML models with hyperparameter tuning"""
    print("\n" + "="*80)
    print("STEP 4: MACHINE LEARNING MODELS")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Define the individual base models for the hybrid
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
            ('rf', rf_base),
            ('xgb', xgb_base)
        ])
    }
    
    param_grids = {
        'Linear Regression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1]},
        'ElasticNet': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]},
        'Bayesian Ridge': {
            'alpha_1': [1e-6, 1e-4],
            'lambda_1': [1e-6, 1e-4]
        },
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
    
    results = {}
    
    print("\n🔧 Training models with hyperparameter tuning...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'model': best_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'best_params': grid_search.best_params_,
            'param_grid': param_grids[name]
        }
        
        print(f"  Best R² (Test): {test_r2:.4f}")
        print(f"  Best RMSE (Test): {test_rmse:.2f}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train R²': [results[m]['train_r2'] for m in results],
        'Test R²': [results[m]['test_r2'] for m in results],
        'Train RMSE': [results[m]['train_rmse'] for m in results],
        'Test RMSE': [results[m]['test_rmse'] for m in results],
        'Train MAE': [results[m]['train_mae'] for m in results],
        'Test MAE': [results[m]['test_mae'] for m in results]
    })
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.8)
    axes[0, 0].bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(x - width/2, results_df['Train RMSE'], width, label='Train RMSE', alpha=0.8)
    axes[0, 1].bar(x + width/2, results_df['Test RMSE'], width, label='Test RMSE', alpha=0.8)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar(x - width/2, results_df['Train MAE'], width, label='Train MAE', alpha=0.8)
    axes[1, 0].bar(x + width/2, results_df['Test MAE'], width, label='Test MAE', alpha=0.8)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
    best_model_r2 = results_df['Test R²'].max()
    
    colors = ['lightgray'] * len(results_df)
    colors[results_df['Model'].tolist().index(best_model_name)] = 'green'
    
    axes[1, 1].bar(results_df['Model'], results_df['Test R²'], color=colors, alpha=0.8)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].set_title(f'Best Model: {best_model_name} (R² = {best_model_r2:.4f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1, 1].axhline(y=best_model_r2, color='red', linestyle='--', label='Best Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    metadata['ml_results'] = results
    metadata['X_train'] = X_train
    metadata['X_test'] = X_test
    metadata['y_train'] = y_train
    metadata['y_test'] = y_test
    
    return metadata

# =============================================================================
# STEP 5: NESTED CROSS-VALIDATION
# =============================================================================

def nested_cross_validation(X_scaled, y, metadata):
    """Perform nested cross-validation for robust performance estimation"""
    print("\n" + "="*80)
    print("STEP 5: NESTED CROSS-VALIDATION")
    print("="*80)
    
    best_model_name = max(metadata['ml_results'].items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = metadata['ml_results'][best_model_name]['model']
    
    print(f"\n🎯 Using best model: {best_model_name}")
    
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    nested_scores = []
    
    print("\n🔄 Running Nested Cross-Validation...")
    
    param_grid = metadata['ml_results'][best_model_name]['param_grid']
    
    for outer_train_idx, outer_test_idx in outer_cv.split(X_scaled):
        X_outer_train = X_scaled.iloc[outer_train_idx]
        X_outer_test = X_scaled.iloc[outer_test_idx]
        y_outer_train = y.iloc[outer_train_idx]
        y_outer_test = y.iloc[outer_test_idx]
        
        model_to_tune = clone(best_model)
        grid_search = GridSearchCV(
            model_to_tune, 
            param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_outer_train, y_outer_train)
        outer_score = grid_search.score(X_outer_test, y_outer_test)
        nested_scores.append(outer_score)
        
        print(f"  Fold R²: {outer_score:.4f}")
    
    nested_scores = np.array(nested_scores)
    
    print(f"\n📊 Nested CV Results:")
    print(f"  Mean R²: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    print(f"  Min R²: {nested_scores.min():.4f}")
    print(f"  Max R²: {nested_scores.max():.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].boxplot(nested_scores, vert=True, patch_artist=True)
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Nested CV Performance Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(range(1, len(nested_scores)+1), nested_scores, alpha=0.8, color='steelblue')
    axes[1].axhline(y=nested_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {nested_scores.mean():.4f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Performance per Fold', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(1, len(nested_scores)+1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('nested_cv_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    metadata['nested_cv_scores'] = nested_scores
    metadata['nested_cv_mean'] = nested_scores.mean()
    metadata['nested_cv_std'] = nested_scores.std()
    
    print("\n✅ Nested cross-validation completed!")
    
    return metadata

# =============================================================================
# STEP 6: PREDICTION RESULTS
# =============================================================================

def prediction_results(X_scaled, y, metadata):
    """Generate prediction results and visualizations (Train/Test R² only)"""
    print("\n" + "="*80)
    print("STEP 6: PREDICTION RESULTS")
    print("="*80)
    
    best_model_name = max(metadata['ml_results'].items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = metadata['ml_results'][best_model_name]['model']
    
    X_train = metadata['X_train']
    X_test = metadata['X_test']
    y_train = metadata['y_train']
    y_test = metadata['y_test']
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    # ✅ UPDATED: 1 row, 2 columns for Train & Test Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training Set
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=50, color='blue')
    axes[0].plot([y_train.min(), y_train.max()], 
                 [y_train.min(), y_train.max()], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Grain Yield')
    axes[0].set_ylabel('Predicted Grain Yield')
    axes[0].set_title(f'Training Set (R² = {r2_score(y_train, y_train_pred):.4f})', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test Set
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='green')
    axes[1].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[1].set_xlabel('Actual Grain Yield')
    axes[1].set_ylabel('Predicted Grain Yield')
    axes[1].set_title(f'Test Set (R² = {r2_score(y_test, y_test_pred):.4f})', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📈 Prediction Statistics:")
    print(f"  Test R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
    print(f"  Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
    print(f"  Mean Residual: {test_residuals.mean():.2f}")
    print(f"  Residual Std: {test_residuals.std():.2f}")
    
    metadata['y_test_pred'] = y_test_pred
    metadata['test_residuals'] = test_residuals
    
    return metadata

# =============================================================================
# STEP 7: SHAP EXPLAINABILITY
# =============================================================================

def shap_explainability(X_scaled, y, metadata):
    """Generate SHAP values for model interpretability"""
    print("\n" + "="*80)
    print("STEP 7: SHAP EXPLAINABILITY")
    print("="*80)
    
    best_model_name = max(metadata['ml_results'].items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = metadata['ml_results'][best_model_name]['model']
    
    X_train = metadata['X_train']
    
    print(f"\n🔍 Generating SHAP values for {best_model_name}...")
    
    # Detect linear models by checking for sklearn's `coef_` attribute 
    # or by matching common linear model names
    is_linear = hasattr(best_model, 'coef_') or any(
        kw in best_model_name.lower() for kw in ['linear', 'ridge', 'lasso', 'elastic', 'sgd', 'bayesian']
    )
    
    if is_linear:
        print("  ↳ Using shap.LinearExplainer (optimized for linear models)")
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_train)
    else:
        print("  ↳ Using shap.TreeExplainer (optimized for tree-based models)")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_train)
        
    # Handle multi-class outputs (SHAP returns a list of arrays for classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    plot_data = X_train
    
    # 1. Global Feature Importance (Bar Plot)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, plot_data, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Global) - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Feature Impact Distribution (Dot/Beeswarm Plot)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, plot_data, plot_type="dot", show=False)
    plt.title(f'SHAP Summary Plot (Feature Impact) - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_dot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Extract & Rank SHAP Importance
    feature_names = plot_data.columns.tolist()
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': shap_importance
    }).sort_values('SHAP Importance', ascending=False)
    
    print("\n📊 Top 15 SHAP Feature Importance Ranking:")
    print(importance_df.head(15).to_string(index=False))
    
    metadata['shap_values'] = shap_values
    metadata['shap_importance'] = importance_df
    metadata['explainer'] = explainer
    
    print("\n✅ SHAP explainability analysis completed!")
    return metadata

# =============================================================================
# STEP 8: STABILITY ANALYSIS (AMMI & GGE)
# =============================================================================

def stability_analysis(metadata):
    """Perform AMMI and GGE biplot analysis for genotype stability"""
    print("\n" + "="*80)
    print("STEP 8: STABILITY ANALYSIS (AMMI & GGE)")
    print("="*80)
    
    df = metadata['df_clean'].copy()
    
    ge_matrix = df.pivot_table(
        index='Gen', 
        columns='L', 
        values='GY', 
        aggfunc='mean'
    ).fillna(df['GY'].mean())
    
    print(f"\n📊 Genotype × Environment Matrix Shape: {ge_matrix.shape}")
    print(ge_matrix)
    
    print("\n🔬 Performing AMMI Analysis...")
    
    grand_mean = ge_matrix.values.mean()
    genotype_means = ge_matrix.mean(axis=1)
    environment_means = ge_matrix.mean(axis=0)
    
    interaction = ge_matrix.values - genotype_means.values.reshape(-1, 1) - \
                  environment_means.values.reshape(1, -1) + grand_mean
    
    U, S, Vt = np.linalg.svd(interaction, full_matrices=False)
    
    n_ipca = min(2, len(S))
    ipca1_gen = U[:, 0] * S[0]
    ipca1_env = Vt[0, :] * S[0]
    
    if n_ipca > 1:
        ipca2_gen = U[:, 1] * S[1]
        ipca2_env = Vt[1, :] * S[1]
    else:
        ipca2_gen = np.zeros_like(ipca1_gen)
        ipca2_env = np.zeros_like(ipca1_env)
    
    gge_data = ge_matrix.values - environment_means.values.reshape(1, -1)
    U_gge, S_gge, Vt_gge = np.linalg.svd(gge_data, full_matrices=False)
    
    pc1_gen = U_gge[:, 0] * S_gge[0]
    pc2_gen = U_gge[:, 1] * S_gge[1] if len(S_gge) > 1 else np.zeros_like(pc1_gen)
    pc1_env = Vt_gge[0, :] * S_gge[0]
    pc2_env = Vt_gge[1, :] * S_gge[1] if len(S_gge) > 1 else np.zeros_like(pc1_env)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    for i, gen in enumerate(ge_matrix.index):
        ax1.scatter(ipca1_gen[i], ipca2_gen[i], c='blue', s=100, label='Genotype' if i==0 else "")
        ax1.annotate(gen, (ipca1_gen[i], ipca2_gen[i]), fontsize=10, ha='right')
    
    for j, env in enumerate(ge_matrix.columns):
        ax1.scatter(ipca1_env[j], ipca2_env[j], c='red', s=150, marker='^', 
                   label='Environment' if j==0 else "")
        ax1.annotate(env, (ipca1_env[j], ipca2_env[j]), fontsize=12, fontweight='bold', ha='left')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('IPCA1')
    ax1.set_ylabel('IPCA2')
    ax1.set_title('AMMI Biplot', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for i, gen in enumerate(ge_matrix.index):
        ax2.scatter(pc1_gen[i], pc2_gen[i], c='green', s=100, label='Genotype' if i==0 else "")
        ax2.annotate(gen, (pc1_gen[i], pc2_gen[i]), fontsize=10, ha='right')
    
    for j, env in enumerate(ge_matrix.columns):
        ax2.scatter(pc1_env[j], pc2_env[j], c='orange', s=150, marker='^', 
                   label='Environment' if j==0 else "")
        ax2.annotate(env, (pc1_env[j], pc2_env[j]), fontsize=12, fontweight='bold', ha='left')
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('GGE Biplot', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    stability_df = pd.DataFrame({
        'Genotype': ge_matrix.index,
        'Mean_Yield': genotype_means.values,
        'IPCA1_Score': ipca1_gen,
        'IPCA2_Score': ipca2_gen,
        'ASV': np.sqrt(ipca1_gen**2 + ipca2_gen**2),
        'PC1_Score': pc1_gen,
        'PC2_Score': pc2_gen
    })
    
    stability_df['Yield_Rank'] = stability_df['Mean_Yield'].rank(ascending=False)
    stability_df['Stability_Rank'] = stability_df['ASV'].rank(ascending=True)
    stability_df['Overall_Rank'] = (stability_df['Yield_Rank'] + stability_df['Stability_Rank']) / 2
    stability_df = stability_df.sort_values('Overall_Rank')
    
    print("\n📈 Genotype Stability Parameters:")
    print(stability_df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    points = np.column_stack([pc1_gen, pc2_gen])
    hull = ConvexHull(points)
    
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=2)
    
    ax.scatter(pc1_gen, pc2_gen, c='blue', s=200, alpha=0.6)
    for i, gen in enumerate(ge_matrix.index):
        ax.annotate(gen, (pc1_gen[i], pc2_gen[i]), fontsize=12, ha='center', 
                   fontweight='bold')
    
    for j, env in enumerate(ge_matrix.columns):
        ax.arrow(0, 0, pc1_env[j], pc2_env[j], 
                head_width=0.1, head_length=0.15, fc='red', ec='red', linewidth=2)
        ax.annotate(env, (pc1_env[j]*1.1, pc2_env[j]*1.1), 
                   fontsize=14, fontweight='bold', color='red')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('GGE Biplot - Which-Won-Where Pattern', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('gge_which_won_where.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ✅ FIXED: Best Genotype per Environment - proper key conversion
    print("\n🏆 Best Genotype per Environment:")
    location_mapping = metadata['location_mapping']
    for env in ge_matrix.columns:
        best_gen = ge_matrix[env].idxmax()
        best_yield = ge_matrix[env].max()
        
        # FIX: Convert integer env to string format matching location_mapping
        if isinstance(env, (int, np.integer)):
            env_key = f'L{env}'
        else:
            env_key = str(env)
        
        # Get location name with fallback
        loc_name = location_mapping.get(env_key, f'Location {env}')
        print(f"  {loc_name}: {best_gen} ({best_yield:.2f})")
    
    metadata['stability_df'] = stability_df
    metadata['ge_matrix'] = ge_matrix
    
    print("\n✅ Stability analysis completed!")
    
    return metadata

# =============================================================================
# STEP 9: COMPARISON & INSIGHTS
# =============================================================================

def comparison_and_insights(metadata):
    """Generate final comparison and insights"""
    print("\n" + "="*80)
    print("STEP 9: COMPARISON & INSIGHTS")
    print("="*80)
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print("-"*80)
    
    results_df = pd.DataFrame({
        'Model': list(metadata['ml_results'].keys()),
        'Test R²': [metadata['ml_results'][m]['test_r2'] 
                   for m in metadata['ml_results']],
        'Test RMSE': [metadata['ml_results'][m]['test_rmse'] 
                     for m in metadata['ml_results']]
    }).sort_values('Test R²', ascending=False)
    
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['Model']
    best_r2 = results_df.iloc[0]['Test R²']
    
    print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")
    print(f"\n✅ Nested CV Validation: R² = {metadata['nested_cv_mean']:.4f} ± {metadata['nested_cv_std']:.4f}")
    
    print("\n🔍 KEY FEATURE INSIGHTS (from SHAP):")
    print("-"*80)
    
    shap_df = metadata['shap_importance'].head(10)
    for idx, row in shap_df.iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['SHAP Importance']:.4f}")
    
    print("\n🌾 GENOTYPE STABILITY INSIGHTS:")
    print("-"*80)
    
    stability_df = metadata['stability_df']
    top_genotypes = stability_df.head(3)
    
    print("Top 3 Stable & High-Yielding Genotypes:")
    for idx, row in top_genotypes.iterrows():
        print(f"  {row['Genotype']}: Mean Yield = {row['Mean_Yield']:.2f}, "
              f"ASV = {row['ASV']:.2f}, Overall Rank = {row['Overall_Rank']:.2f}")
    
    # ✅ FIXED: Location insights with proper key conversion
    print("\n📍 LOCATION INSIGHTS:")
    print("-"*80)
    
    location_performance = metadata['ge_matrix'].mean(axis=0).sort_values(ascending=False)
    location_mapping = metadata['location_mapping']
    
    for loc, yield_val in location_performance.items():
        # FIX: Convert integer loc to string format
        if isinstance(loc, (int, np.integer)):
            loc_key = f'L{loc}'
        else:
            loc_key = str(loc)
        
        loc_name = location_mapping.get(loc_key, f'Location {loc}')
        print(f"  {loc_name}: Average Yield = {yield_val:.2f}")
    
    # ✅ REMOVED: final_insights plot section (as requested)
    
    print("\n💾 Saving results...")
    
    results_df.to_csv('model_performance.csv', index=False)
    stability_df.to_csv('genotype_stability.csv', index=False)
    shap_df.to_csv('feature_importance.csv', index=False)
    
    print("  ✅ model_performance.csv")
    print("  ✅ genotype_stability.csv")
    print("  ✅ feature_importance.csv")
    print("  ✅ All visualization files saved")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    
    return metadata

# =============================================================================
# STEP 10: ENHANCED VISUALIZATIONS & SUMMARY TABLES
# =============================================================================

def enhanced_visualizations_and_tables(metadata):
    """Generate focused plots (SHAP + G×L Heatmap) and comprehensive summary tables"""
    print("\n" + "="*80)
    print("STEP 10: FOCUSED VISUALIZATIONS & SUMMARY TABLES")
    print("="*80)
    
    import matplotlib.gridspec as gridspec
    
    df = metadata['df_clean'].copy()
    best_model_name = max(metadata['ml_results'].items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = metadata['ml_results'][best_model_name]['model']
    
    X_test = metadata['X_test'].reset_index(drop=True)
    y_test = metadata['y_test'].reset_index(drop=True)
    y_test_pred = metadata['y_test_pred']
    
    # ========================================================================
    # 🔍 PLOT 1: SHAP Sample Explanations 
    # ========================================================================
    print("\n🔍 Creating SHAP Sample Explanations...")
    
    try:
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(best_model)
            shap_values_all = explainer.shap_values(X_test)
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[0]
        else:
            # Fallback for non-tree models (use subset for speed)
            explainer = shap.KernelExplainer(best_model.predict, X_test.iloc[:50])
            shap_values_all = explainer.shap_values(X_test.iloc[:50])
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sample_labels = ['Low Yield', 'Medium Yield', 'High Yield']
        
        # Select representative samples across yield distribution
        y_test_sorted_idx = y_test.sort_values().index.tolist()
        sample_positions = [
            y_test_sorted_idx[0], 
            y_test_sorted_idx[len(y_test_sorted_idx)//2], 
            y_test_sorted_idx[-1]
        ]
        
        for idx, (sample_pos, label) in enumerate(zip(sample_positions, sample_labels)):
            X_sample = X_test.iloc[[sample_pos]]
            y_actual = y_test.iloc[sample_pos]
            y_pred = y_test_pred[sample_pos]
            
            # Get SHAP values for this sample
            shap_idx = min(sample_pos, len(shap_values_all)-1)
            shap_imp = shap_values_all[shap_idx]
            
            feature_names = X_test.columns.tolist()
            top_idx = np.argsort(np.abs(shap_imp))[-10:][::-1]  # Top 10 features
            
            # Color: red = positive contribution, blue = negative
            colors = ['red' if v > 0 else 'blue' for v in shap_imp[top_idx]]
            axes[idx].barh(range(10), shap_imp[top_idx], color=colors, alpha=0.8, edgecolor='black')
            axes[idx].set_yticks(range(10))
            
            # Truncate long feature names for readability
            display_names = []
            for i in top_idx:
                name = feature_names[i]
                if len(name) > 15:
                    name = name.split('_')[-1] if '_' in name else name[:12] + '...'
                display_names.append(name)
            
            axes[idx].set_yticklabels(display_names, fontsize=9, fontweight='bold')
            axes[idx].set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10, fontweight='bold')
            axes[idx].set_title(f'{label}\nActual: {y_actual:.0f} | Pred: {y_pred:.0f}', 
                               fontsize=12, fontweight='bold', pad=12)
            axes[idx].axvline(x=0, color='gray', linestyle='--', linewidth=1)
            axes[idx].grid(True, alpha=0.3, axis='x', linestyle='--')
            axes[idx].tick_params(axis='both', labelsize=8)
        
        plt.suptitle(f'SHAP Explanations: {best_model_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('shap_sample_explanations.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: shap_sample_explanations.png")
        
    except Exception as e:
        print(f"⚠️ SHAP plot skipped due to: {str(e)}")
        print("   Continuing with other visualizations...")
    
    # ========================================================================
    # 🌾 PLOT 2: Genotype × Location Heatmap
    # ========================================================================
    print("\n🌾 Creating Genotype × Location Heatmap...")
    
    ge_matrix = metadata['ge_matrix']
    location_mapping = metadata['location_mapping']
    
    # Rename columns with full location names
    ge_matrix_renamed = ge_matrix.copy()
    ge_matrix_renamed.columns = [
        location_mapping.get(f'L{col}', location_mapping.get(str(col), f'Location {col}'))
        for col in ge_matrix_renamed.columns
    ]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(ge_matrix_renamed, annot=True, fmt='.0f', cmap='YlOrRd', 
                linewidths=0.8, linecolor='white', cbar_kws={'label': 'Grain Yield (kg/ha)', 'shrink': 0.8})
    plt.title('Genotype × Location Yield Interaction', fontsize=16, fontweight='bold', pad=25)
    plt.xlabel('Location', fontsize=13, fontweight='bold')
    plt.ylabel('Genotype', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('genotype_location_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: genotype_location_heatmap.png")
    
    # ========================================================================
    # 📋 SUMMARY TABLES (all preserved)
    # ========================================================================
    print("\n📋 Generating Summary Tables...")
    
    # --- Table 1: Model Performance ---
    results_df = pd.DataFrame({
        'Model': list(metadata['ml_results'].keys()),
        'Train R²': [metadata['ml_results'][m]['train_r2'] for m in metadata['ml_results']],
        'Test R²': [metadata['ml_results'][m]['test_r2'] for m in metadata['ml_results']],
        'Train RMSE': [metadata['ml_results'][m]['train_rmse'] for m in metadata['ml_results']],
        'Test RMSE': [metadata['ml_results'][m]['test_rmse'] for m in metadata['ml_results']],
        'Train MAE': [metadata['ml_results'][m]['train_mae'] for m in metadata['ml_results']],
        'Test MAE': [metadata['ml_results'][m]['test_mae'] for m in metadata['ml_results']]
    }).sort_values('Test R²', ascending=False).reset_index(drop=True)
    
    model_summary = results_df.copy()
    model_summary['Overfitting Gap'] = model_summary['Train R²'] - model_summary['Test R²']
    model_summary['RMSE/MAE Ratio'] = model_summary['Test RMSE'] / model_summary['Test MAE']
    model_summary = model_summary.round(4)
    
    print("\n📊 TABLE 1: MODEL PERFORMANCE SUMMARY")
    print("="*100)
    print(model_summary.to_string(index=False))
    model_summary.to_csv('table_1_model_performance.csv', index=False)
    
    # --- Table 2: Genotype Ranking ---
    stability_df = metadata['stability_df']
    genotype_summary = stability_df[['Genotype', 'Mean_Yield', 'ASV', 'Yield_Rank', 'Stability_Rank', 'Overall_Rank']].copy()
    genotype_summary = genotype_summary.round(2)
    genotype_summary.columns = ['Genotype', 'Mean Yield (kg/ha)', 'ASV (Stability)', 'Yield Rank', 'Stability Rank', 'Overall Rank']
    
    print("\n🌾 TABLE 2: TOP GENOTYPES BY STABILITY & YIELD")
    print("="*80)
    print(genotype_summary.head(10).to_string(index=False))
    genotype_summary.to_csv('table_2_genotype_ranking.csv', index=False)
    
    # --- Table 3: SHAP Feature Importance ---
    shap_summary = metadata['shap_importance'].head(20).copy()
    shap_summary['Relative Importance (%)'] = (shap_summary['SHAP Importance'] / shap_summary['SHAP Importance'].sum() * 100).round(2)
    shap_summary = shap_summary.round(4)
    
    print("\n🔍 TABLE 3: TOP 20 FEATURES BY SHAP IMPORTANCE")
    print("="*70)
    print(shap_summary.to_string(index=False))
    shap_summary.to_csv('table_3_feature_importance.csv', index=False)
    
    # --- Table 4: Location Performance ---
    location_summary = pd.DataFrame({
        'Location Code': metadata['ge_matrix'].columns,
        'Location Name': [
            location_mapping.get(f'L{loc}', location_mapping.get(str(loc), f'Location {loc}'))
            for loc in metadata['ge_matrix'].columns
        ],
        'Average Yield': metadata['ge_matrix'].mean(axis=0).values.round(2),
        'Yield Std Dev': metadata['ge_matrix'].std(axis=0).values.round(2),
        'Best Genotype': [metadata['ge_matrix'][loc].idxmax() for loc in metadata['ge_matrix'].columns],
        'Best Yield': [metadata['ge_matrix'][loc].max() for loc in metadata['ge_matrix'].columns]
    })
    location_summary = location_summary.sort_values('Average Yield', ascending=False)
    
    print("\n📍 TABLE 4: LOCATION PERFORMANCE SUMMARY")
    print("="*90)
    print(location_summary.to_string(index=False))
    location_summary.to_csv('table_4_location_performance.csv', index=False)
    
    # --- Table 5: Prediction Error Analysis ---
    from sklearn.metrics import mean_squared_error, r2_score
    
    test_indices = metadata['X_test'].index
    df_test = df.loc[test_indices].copy()
    df_test['Predicted_GY'] = metadata['y_test_pred']
    df_test['Absolute Error'] = np.abs(df_test['GY'] - df_test['Predicted_GY'])
    df_test['Relative Error (%)'] = (df_test['Absolute Error'] / df_test['GY'] * 100).round(2)
    
    error_summary = pd.DataFrame({
        'Metric': ['Mean Absolute Error', 'RMSE', 'MAPE (%)', 'R² Score', 'Max Error', 'Min Error'],
        'Value': [
            round(df_test['Absolute Error'].mean(), 2),
            round(np.sqrt(mean_squared_error(df_test['GY'], df_test['Predicted_GY'])), 2),
            round(df_test['Relative Error (%)'].mean(), 2),
            round(r2_score(df_test['GY'], df_test['Predicted_GY']), 4),
            round(df_test['Absolute Error'].max(), 2),
            round(df_test['Absolute Error'].min(), 2)
        ]
    })
    
    print("\n📈 TABLE 5: PREDICTION ERROR ANALYSIS (TEST SET)")
    print("="*50)
    print(error_summary.to_string(index=False))
    error_summary.to_csv('table_5_prediction_errors.csv', index=False)
    
    # ========================================================================
    # 🏆 FINAL REPORT CARD
    # ========================================================================
    print("\n" + "🏆"*40)
    print("FINAL REPORT CARD")
    print("🏆"*40)
    
    report = f"""
📊 DATASET SUMMARY:
• Total Samples: {len(df)} (after outlier removal)
• Features: {len(metadata['encoded_feature_cols'])} (after one-hot encoding)
• Target Variable: Grain Yield (GY)
• Locations: {len(df['L'].unique())} | Genotypes: {len(df['Gen'].unique())}

🤖 BEST MODEL: {best_model_name}
• Test R²: {metadata['ml_results'][best_model_name]['test_r2']:.4f}
• Test RMSE: {metadata['ml_results'][best_model_name]['test_rmse']:.2f} kg/ha
• Nested CV R²: {metadata['nested_cv_mean']:.4f} ± {metadata['nested_cv_std']:.4f}

🌟 TOP 3 RECOMMENDED GENOTYPES:
"""
    
    for idx, row in stability_df.head(3).iterrows():
        report += f"   {idx+1}. {row['Genotype']}: Yield={row['Mean_Yield']:.1f}, ASV={row['ASV']:.1f}, Rank={row['Overall_Rank']:.1f}\n"
    
    report += f"""
🔑 KEY DRIVERS OF YIELD (SHAP):
"""
    for idx, row in metadata['shap_importance'].head(5).iterrows():
        report += f"   • {row['Feature']}: {row['SHAP Importance']:.2f}\n"
    
    report += f"""
📍 BEST PERFORMING LOCATION: {location_summary.iloc[0]['Location Name']} ({location_summary.iloc[0]['Average Yield']:.1f} kg/ha)

💡 RECOMMENDATIONS:
• Prioritize genotypes {', '.join(stability_df.head(3)['Genotype'])} for stable high yields
• Focus breeding on {', '.join(metadata['shap_importance'].head(3)['Feature'])} (top SHAP features)
• {location_summary.iloc[0]['Location Name']} shows highest yield potential
• Overfitting gap: {model_summary.iloc[0]['Overfitting Gap']:.3f} (good generalization if <0.1)
"""
    
    print(report)
    
    with open('final_report_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n✅ All visualizations and tables saved!")
    print("📁 Output files:")
    print("   • shap_sample_explanations.png  ← Model interpretability")
    print("   • genotype_location_heatmap.png ← G×E interaction insights")
    print("   • table_1_model_performance.csv")
    print("   • table_2_genotype_ranking.csv")
    print("   • table_3_feature_importance.csv")
    print("   • table_4_location_performance.csv")
    print("   • table_5_prediction_errors.csv")
    print("   • final_report_summary.txt")
    
    return metadata

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the complete pipeline"""
    print("\n" + "="*80)
    print("LENTIL YIELD PREDICTION & STABILITY ANALYSIS PIPELINE")
    print("="*80)
    print("\nStarting complete analysis...\n")
    
    X, y, metadata = load_and_preprocess_data('lentil_150_11_05_26.xlsx')
    X_scaled, y, metadata = feature_engineering(X, y, metadata)
    plot_feature_distributions_and_correlation(X_scaled, metadata, save_plots=True)
    metadata = manifold_learning(X_scaled, y, metadata)
    metadata = train_ml_models(X_scaled, y, metadata)
    metadata = nested_cross_validation(X_scaled, y, metadata)
    metadata = prediction_results(X_scaled, y, metadata)
    metadata = shap_explainability(X_scaled, y, metadata)
    metadata = stability_analysis(metadata)
    metadata = comparison_and_insights(metadata)
    metadata = enhanced_visualizations_and_tables(metadata)
    return metadata

if __name__ == "__main__":
    metadata = main()