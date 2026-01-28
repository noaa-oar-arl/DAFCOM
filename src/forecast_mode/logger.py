import logging

import mlflow


class Logger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def info(self, msg):
        logging.info(msg)

    def error(self, msg):
        logging.error(msg)


class PerformanceTracker:
    def log_matrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
