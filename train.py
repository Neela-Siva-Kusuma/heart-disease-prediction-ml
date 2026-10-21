import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Standard UCI Cleveland Dataset Column Names
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load data, replacing '?' missing values with NaN
df = pd.read_csv('heart.csv', names=column_names, header=None, na_values='?')

# If first row contains string headers accidentally, drop it
if not str(df.iloc[0]['age']).replace('.', '', 1).isdigit():
    df = df.iloc[1:].reset_index(drop=True)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target'].astype(float).apply(lambda x: 1 if x > 0 else 0)

# Impute missing values with column medians
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Pipeline Artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Model, Scaler, and Imputer successfully saved!")
