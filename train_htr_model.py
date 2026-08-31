import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from data_processing import load_data, preprocess_data, feature_engineer

# Define file paths
files = ['E0 .csv', 'E01.csv', 'E02.csv']
MODEL_PATH = 'epl_htr_model.pkl'

# Load and process data
df = load_data(files)
df = preprocess_data(df)
df, _ = feature_engineer(df)

# Select features and target
features = [
    'HomeTeamForm', 'AwayTeamForm', 'H2H_Advantage',
    'HomeAvgGoals', 'HomeAvgShots', 'HomeAvgShotsTarget', 'HomeAvgCorners',
    'AwayAvgGoals', 'AwayAvgShots', 'AwayAvgShotsTarget', 'AwayAvgCorners',
    'HomeAvgYellows', 'HomeAvgReds', 'AwayAvgYellows', 'AwayAvgReds',
    'HomeElo', 'AwayElo',
    'NormProb_H', 'NormProb_D', 'NormProb_A'
]
target = 'HTR'

# Handle missing values
df_clean = df.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Train model
print("Starting hyperparameter tuning for Half-Time Result model...")
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

print("\nBest Hyperparameters found:")
print(grid_search.best_params_)

# Evaluate model
y_pred = best_model.predict(X_test)
print("\nHalf-Time Result Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and encoder
model_data = {'model': best_model, 'encoder': le}
joblib.dump(model_data, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')
