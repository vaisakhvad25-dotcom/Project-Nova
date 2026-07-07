import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors


class BenchmarkModel:
    """Container for regression, classification, and nearest-neighbor models."""

    def __init__(self, n_neighbors=5, random_state=42):
        # Create the underlying machine-learning components used by the app.
        self.regression_model = LinearRegression()
        self.classifier = DecisionTreeClassifier(random_state=random_state)
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X_reg, y_reg, y_class, X_knn):
        # Train the regression, classifier, and k-NN models on the provided data.
        self.regression_model.fit(X_reg, y_reg)
        self.classifier.fit(X_reg, y_class)
        self.knn_model.fit(X_knn)

    def predict(self, feature_values):
        # Estimate benchmark scores from hardware features using the regression model.
        return self.regression_model.predict(np.array(feature_values, dtype=float))

    def predict_score(self, feature_values):
        # Estimate a single benchmark score from hardware features.
        return float(self.predict([feature_values])[0])

    def classify_score(self, score, thresholds):
        # Convert a benchmark score into a category index using the provided thresholds.
        if score < thresholds[0]:
            return 0
        if score < thresholds[1]:
            return 1
        return 2

    def predict_with_category(self, feature_values, thresholds):
        # Predict a benchmark score and map it to a category index.
        score = self.predict_score(feature_values)
        category_index = self.classify_score(score, thresholds)
        return score, category_index

    def predict_category(self, feature_values):
        # Classify a feature vector into a category label index.
        return int(self.classifier.predict(np.array([feature_values], dtype=float))[0])

    def find_neighbors(self, feature_values, k=5):
        # Return distances and indices for the closest neighbors.
        return self.knn_model.kneighbors([feature_values], n_neighbors=k, return_distance=True)
