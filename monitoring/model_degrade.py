import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from prometheus_client import start_http_server, Gauge
import time
import joblib  # For loading the model file

# Load the trained model
MODEL_PATH = "../model.pkl"  # Replace with the path to your model file
model = joblib.load(MODEL_PATH)

# Get the number of features expected by the model
N_FEATURES = model.n_features_in_  # Number of features the model was trained on

# Prometheus metrics
accuracy_metric = Gauge('model_accuracy', 'Accuracy of the model')
precision_metric = Gauge('model_precision', 'Precision of the model')
f1_metric = Gauge('model_f1_score', 'F1 Score of the model')
drift_factor_metric = Gauge('model_drift_factor', 'Drift factor applied to the data')  # Metric for drift factor

# Function to generate synthetic data
def generate_data(drift_factor=0.0):
    """
    Generates synthetic data with an optional drift factor.
    """
    np.random.seed(42)
    n_samples = 100

    # Base data (no drift)
    X = np.random.normal(0, 1, (n_samples, N_FEATURES))  # Use N_FEATURES
    y = np.random.randint(0, 2, n_samples)  # Binary classification

    # Introduce data drift by shifting the mean of the features
    X_drifted = X + np.random.normal(drift_factor, 0.1, (n_samples, N_FEATURES))

    return X_drifted, y

# Function to update metrics
def update_metrics(current_drift_factor):
    """
    Updates model metrics based on the current data drift.
    """
    # Generate data with drift
    X, y_true = generate_data(current_drift_factor)

    # Make predictions using the loaded model
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Update Prometheus metrics
    accuracy_metric.set(accuracy)
    precision_metric.set(precision)
    f1_metric.set(f1)
    drift_factor_metric.set(current_drift_factor)  # Update drift factor metric

    # Print metrics to console (for debugging)
    print(f"Drift factor: {current_drift_factor:.2f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}")

# Main function
if __name__ == '__main__':
    # Start Prometheus HTTP server
    start_http_server(8000)
    print("Prometheus metrics server started on port 8000.")

    # Simulate data drift over time
    drift_factor = 0.0  # Start with no drift
    while True:
        update_metrics(drift_factor)

        # Gradually increase drift
        drift_factor += 0.01  # Increase drift over time
        time.sleep(10)  # Update metrics every 10 seconds