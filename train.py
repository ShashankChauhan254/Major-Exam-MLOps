import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class IrisDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        # Load iris dataset and initialize attributes
        iris = load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.target = iris.target
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.train_data = None
        self.test_data = None
        self.train_target = None
        self.test_target = None

    def prepare_data(self):
        # Scale features
        self.data[self.data.columns] = self.scaler.fit_transform(self.data)
        
        # Split data into training and test sets
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=self.random_state
        )
        
    def get_feature_stats(self):
        # Basic statistical analysis (mean, std, min, max) for the dataset
        stats = self.data.describe().T[['mean', 'std', 'min', 'max']]
        return stats

class IrisExperiment:
    def __init__(self, data_processor):
        # Initialize with a data processor and define models
        self.data_processor = data_processor
        self.models = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier()
        }
        self.results = {}

    def run_experiment(self):
        # Ensure data is prepared
        self.data_processor.prepare_data()
        X_train, y_train = self.data_processor.train_data, self.data_processor.train_target
        
        for model_name, model in self.models.items():
            # Start MLflow experiment for each model
            with mlflow.start_run(run_name=model_name):
                # Perform cross-validation predictions
                y_pred = cross_val_predict(model, X_train, y_train, cv=5)
                
                # Calculate metrics
                accuracy = accuracy_score(y_train, y_pred)
                precision = precision_score(y_train, y_pred, average='weighted')
                recall = recall_score(y_train, y_pred, average='weighted')
                
                # Save results
                self.results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall
                }
                
                # Log results to MLflow
                self.log_results(model_name, accuracy, precision, recall)
                
                # Train the model and log the trained model itself
                model.fit(X_train, y_train)
                mlflow.sklearn.log_model(model, model_name)

    def log_results(self, model_name, accuracy, precision, recall):
        # Log metrics to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"Logged metrics for {model_name} to MLflow.")

# Example of usage
data_processor = IrisDataProcessor()
experiment = IrisExperiment(data_processor)
experiment.run_experiment()

# Display experiment results
print("Experiment Results:\n", experiment.results)
