import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from preprocessing import load_and_preprocess_data, preprocess_privacy_data
from models import (train_classification_model, evaluate_model,
                    train_anomaly_detection_model, predict_anomalies,
                    train_privacy_preserving_model, evaluate_privacy_model)

ARFF_FILE_PATH = "/home/sriya/Downloads/ai_network/network_analysis/KDDTest+.arff"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("--- Starting Network Traffic Analysis ---")
X_train, X_test, y_train, y_test, preprocessor, df = load_and_preprocess_data(
    arff_file_path=ARFF_FILE_PATH,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

if X_train is None:
    print("Failed to load and preprocess main data. Exiting.")
    exit()

print("\n--- Training Main Traffic Classification Model ---")
best_classifier_model, grid_results = train_classification_model(X_train, y_train, preprocessor)
evaluate_model(best_classifier_model, X_test, y_test)

categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns

preprocessor_anomaly = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

anomaly_model, fitted_anomaly_preprocessor = train_anomaly_detection_model(X_train, y_train, preprocessor_anomaly)

y_anomaly_predictions = predict_anomalies(anomaly_model, fitted_anomaly_preprocessor, X_test)

print("\nSample Anomaly Detection Output:")
sample_output = pd.DataFrame({'Actual Class': y_test.values[:10], 'Anomaly Prediction': y_anomaly_predictions.values[:10]})
print(sample_output)

print("\n--- Simulating and Preprocessing Privacy-Preserving Data ---")
np.random.seed(RANDOM_STATE)
privacy_df = pd.DataFrame({
    'FlowDuration': np.random.normal(500, 150, 1000),
    'PacketCount': np.random.randint(5, 50, 1000),
    'AvgPacketSize': np.random.normal(300, 50, 1000),
    'TCP_Flag_Count': np.random.randint(0, 5, 1000),
    'Label': np.random.choice(['Benign', 'Suspicious'], size=1000, p=[0.8, 0.2])
})

Xp_train_scaled, Xp_test_scaled, yp_train, yp_test, privacy_scaler = preprocess_privacy_data(
    privacy_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

if Xp_train_scaled is None:
     print("Failed to preprocess privacy data. Exiting.")
     exit()

print("\n--- Training Privacy-Preserving Model ---")
privacy_model = train_privacy_preserving_model(Xp_train_scaled, yp_train)
evaluate_privacy_model(privacy_model, Xp_test_scaled, yp_test)

print("\n--- Network Traffic Analysis Complete ---")
