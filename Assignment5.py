# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set plot style for better visualization
plt.style.use('seaborn-v0_8') 

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the dataset
# Construct path to Housing.csv in the same directory
csv_path = os.path.join(script_dir, 'Housing.csv')
df = pd.read_csv(csv_path)

# --- Data Cleaning ---
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# No missing values found in the dataset, but let's confirm data types
print("\nData Types:\n", df.dtypes)

# Convert categorical columns to appropriate type
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Check for outliers in numerical columns using IQR method
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']  # Excluded 'price'
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    if not outliers.empty:
        print(f"\nOutliers in {col}:\n", outliers)
        # Cap outliers to reduce their impact
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# --- Exploratory Data Analysis (EDA) ---
# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Visualize price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig(os.path.join(script_dir, 'price_distribution.png'))
plt.close()

# Correlation matrix for numerical features including price
corr_cols = numerical_cols + ['price']  # Include price for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig(os.path.join(script_dir, 'correlation_matrix.png'))
plt.close()

# Boxplot of price vs. furnishing status
plt.figure(figsize=(10, 6))
sns.boxplot(x='furnishingstatus', y='price', data=df)
plt.title('Price vs. Furnishing Status')
plt.savefig(os.path.join(script_dir, 'price_vs_furnishing.png'))
plt.close()

# Scatter plot of area vs. price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price', hue='bedrooms', size='bathrooms', data=df)
plt.title('Area vs. Price (Colored by Bedrooms, Sized by Bathrooms)')
plt.savefig(os.path.join(script_dir, 'area_vs_price.png'))
plt.close()

# --- Feature Engineering ---
# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Create new feature: total rooms (bedrooms + bathrooms)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Create interaction term: area per room
df['area_per_room'] = df['area'] / df['total_rooms']

# --- Data Preprocessing ---
# Define features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# --- Model Training and Evaluation ---
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Cross-Validation R2 Scores: {cv_scores}")
    print(f"Average CV R2 Score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# --- Hyperparameter Tuning for Random Forest ---
# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nRandom Forest Hyperparameter Tuning Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best R2 Score: {grid_search.best_score_:.2f}")

# Train final model with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)

# Final evaluation
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
final_r2 = r2_score(y_test, final_predictions)
final_mae = mean_absolute_error(y_test, final_predictions)

print("\nFinal Random Forest Model Results:")
print(f"RMSE: {final_rmse:.2f}")
print(f"R2 Score: {final_r2:.2f}")
print(f"MAE: {final_mae:.2f}")

# --- Feature Importance ---
# Plot feature importance for Random Forest
feature_importance = pd.Series(final_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.savefig(os.path.join(script_dir, 'feature_importance.png'))
plt.close()

# --- Save Results ---
# Save model performance to a CSV file
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(script_dir, 'model_performance.csv'))

print("\nEDA plots and model performance have been saved to the House_Price_Prediction folder.")
print("You can view the plots: price_distribution.png, correlation_matrix.png, price_vs_furnishing.png, area_vs_price.png, feature_importance.png")
print("Model performance metrics are saved in model_performance.csv")