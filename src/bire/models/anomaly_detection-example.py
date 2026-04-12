import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination=0.05):
        """
        contamination: expected percentage of anomalies
        """
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, data):
        """
        Train model on normal data
        """
        self.model.fit(data)

    def predict(self, data):
        """
        Predict anomalies
        Returns:
            -1 = anomaly
             1 = normal
        """
        return self.model.predict(data)

    def predict_with_scores(self, data):
        """
        Returns anomaly labels + anomaly scores
        """
        labels = self.model.predict(data)
        scores = self.model.decision_function(data)
        return labels, scores


if __name__ == "__main__":
    # Example usage with fake patient data

    data = pd.DataFrame({
        "heart_rate": [72, 75, 78, 120, 77, 74, 200],
        "respiratory_rate": [16, 18, 17, 30, 16, 17, 35],
        "temperature": [98.6, 98.7, 98.5, 101.2, 98.6, 98.7, 103.5],
    })

    detector = AnomalyDetector(contamination=0.2)
    detector.fit(data)

    labels, scores = detector.predict_with_scores(data)

    data["anomaly"] = labels
    data["score"] = scores

    print(data)
