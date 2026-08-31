import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from data_processing import load_data, preprocess_data, feature_engineer

# Define file paths
files = ['E0 .csv', 'E01.csv', 'E02.csv']
MODEL_PATH = 'epl_goals_model.pkl'

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
    'NormProb_H', 'NormProb_D', 'NormProb_A',
    'NormProb_O2.5', 'NormProb_U2.5'
]

# Define goal buckets
def to_buckets(goals):
    if goals <= 1:
        return '0-1'
    elif goals <= 3:
        return '2-3'
    else:
        return '4+'

df['Total_Goals_Bucket'] = (df['FTHG'] + df['FTAG']).apply(to_buckets)
target = 'Total_Goals_Bucket'

# Handle missing values
df_clean = df.dropna(subset=features + [target])

X = df_clean[features]
y_str = df_clean[target]
le = LabelEncoder()
y = le.fit_transform(y_str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate sample weights to handle class imbalance
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize model
model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_log_loss')

# Train model
print("Starting hyperparameter tuning for Total Goals model...")
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# Get the best model
best_model = grid_search.best_estimator_

print("\nBest Hyperparameters found:")
print(grid_search.best_params_)

# Evaluate model
y_pred_encoded = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)
y_test_labels = le.inverse_transform(y_test)
print("Total Goals Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_labels, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred))

# Save the best model
model_and_encoder = {'model': best_model, 'encoder': le}
joblib.dump(model_and_encoder, MODEL_PATH)
print(f"\nModel and encoder saved to {MODEL_PATH}")
