import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def train_classification_model(X_train, y_train, preprocessor):
    """
    Trains and optimizes a RandomForestClassifier using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        preprocessor (ColumnTransformer): The fitted preprocessor object.

    Returns:
        tuple: A tuple containing:
            - best_model (Pipeline): The trained and optimized classification model pipeline.
            - grid_search_results (dict): Dictionary containing best parameters and training time.
    """
    pipeline_grid = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=42))])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5]
    }

    grid = GridSearchCV(pipeline_grid,
                        param_grid,
                        cv=3,
                        scoring='f1_macro',
                        verbose=1)

    print("\nStarting Grid Search for Classification Model...")
    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()

    grid_search_results = {
        'training_time': end - start,
        'best_params': grid.best_params_
    }

    print(f"Grid Search Training Time: {grid_search_results['training_time']:.2f} seconds")
    print("Best Parameters Found:", grid_search_results['best_params'])

    best_model = grid.best_estimator_
    return best_model, grid_search_results

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on the test set and prints a classification report.

    Args:
        model: The trained machine learning model (e.g., a Pipeline or Classifier).
        X_test: Testing features.
        y_test: Testing labels.

    Returns:
        None
    """
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def train_anomaly_detection_model(X_train, y_train, preprocessor_anomaly):
    """
    Trains an Isolation Forest model for anomaly detection.

    Args:
        X_train (pd.DataFrame): Training features (original, not yet processed for anomaly model).
        y_train (pd.Series): Training labels.
        preprocessor_anomaly (ColumnTransformer): The preprocessor configured for the anomaly model.

    Returns:
        IsolationForest: The trained Isolation Forest model.
        ColumnTransformer: The fitted preprocessor for anomaly detection.
    """
    print("\nTraining Isolation Forest Model for Anomaly Detection...")
    X_train_processed = preprocessor_anomaly.fit_transform(X_train)
    anomaly_model = IsolationForest(contamination=0.01, random_state=42)
    anomaly_model.fit(X_train_processed)
    print("Isolation Forest Model trained.")
    return anomaly_model, preprocessor_anomaly


def predict_anomalies(anomaly_model, preprocessor_anomaly, X_test):
    """
    Predicts anomalies using the trained Isolation Forest model.

    Args:
        anomaly_model (IsolationForest): The trained Isolation Forest model.
        preprocessor_anomaly (ColumnTransformer): The fitted preprocessor for anomaly detection.
        X_test: Testing features (original, not yet processed for anomaly model).

    Returns:
        pd.Series: Predicted anomaly labels ('Anomaly' or 'Normal').
    """
    print("\nDetecting Anomalies on Test Data...")
    X_test_processed = preprocessor_anomaly.transform(X_test)
    y_anomaly_pred_raw = anomaly_model.predict(X_test_processed)
    y_anomaly_label = pd.Series(['Anomaly' if val == -1 else 'Normal' for val in y_anomaly_pred_raw],
                                 index=X_test.index)
    print("Anomaly detection prediction complete.")
    return y_anomaly_label


def train_privacy_preserving_model(Xp_train_scaled, yp_train):
    """
    Trains a RandomForestClassifier for the privacy-preserving analysis.

    Args:
        Xp_train_scaled (np.ndarray): Scaled training features for the privacy model.
        yp_train (pd.Series): Training labels for the privacy model.

    Returns:
        RandomForestClassifier: The trained privacy-preserving model.
    """
    print("\nTraining Privacy-Preserving Model...")
    clf_priv = RandomForestClassifier(random_state=42)
    clf_priv.fit(Xp_train_scaled, yp_train)
    print("Privacy-Preserving Model trained.")
    return clf_priv

def evaluate_privacy_model(model, Xp_test_scaled, yp_test):
    """
    Evaluates the privacy-preserving model on the test set.

    Args:
        model (RandomForestClassifier): The trained privacy-preserving model.
        Xp_test_scaled (np.ndarray): Scaled testing features for the privacy model.
        yp_test (pd.Series): Testing labels for the privacy model.

    Returns:
        None
    """
    print("\nEvaluating Privacy-Preserving Model Performance...")
    yp_pred = model.predict(Xp_test_scaled)
    print("\nPrivacy-Preserving Model Report:")
    print(classification_report(yp_test, yp_pred))
