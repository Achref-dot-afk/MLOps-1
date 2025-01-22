from prometheus_client import start_http_server, Gauge
import time

metric = Gauge('model_accuracy', 'Accuracy of the ML model')

def update_metrics():
    # Logic to calculate model accuracy
    accuracy = 0.4
    metric.set(accuracy)

if __name__ == '__main__':
    start_http_server(8000,addr='0.0.0.0')
    while True:
        update_metrics()
        time.sleep(60)  # Update every minute
