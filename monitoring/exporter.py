from prometheus_client import start_http_server, Gauge
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import load

# Define Prometheus metrics
accuracy_metric = Gauge('model_accuracy', 'Accuracy of the ML model')
precision_metric = Gauge('model_precision', 'Precision of the ML model')
recall_metric = Gauge('model_recall', 'Recall of the ML model')
f1_score_metric = Gauge('model_f1_score', 'F1 Score of the ML model')

def update_metrics():
    model = load("../model.pkl") 
    x_train, x_test, y_train, y_test = load("../preprocessed_data.pkl")
    
    # Predict labels
    y_pred = model.predict(x_test)
    
    # Handle binary or multi-class predictions
    if len(y_pred.shape) > 1:  # Multi-class case
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:  # Binary classification
        y_pred_labels = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)

    # Log and update metrics
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    accuracy_metric.set(accuracy)
    precision_metric.set(precision)
    recall_metric.set(recall)
    f1_score_metric.set(f1)

if __name__ == '__main__':
    # Start Prometheus HTTP server
    start_http_server(8000, addr='0.0.0.0')
    while True:
        update_metrics()
        time.sleep(60)  # Update metrics every minute
