import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class IrisModelOptimizer:
    def __init__(self, model):
        """
        Initialize the optimizer with a pre-trained model.
        :param model: Trained model instance (e.g., LogisticRegression)
        """
        self.model = model
        self.quantized_coef_ = None
        self.quantized_intercept_ = None

    def quantize_model(self):
        """
        Quantize the model weights by converting them to 8-bit integers.
        Note: LogisticRegression doesn't have built-in quantization support, so we mimic it by scaling.
        """
        # Scale weights and biases to 8-bit integer range and store them as new attributes.
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            # Quantize weights (coef_) and intercepts to 8-bit integers
            self.quantized_coef_ = (self.model.coef_ * 127 / np.max(np.abs(self.model.coef_))).astype(np.int8)
            self.quantized_intercept_ = (self.model.intercept_ * 127 / np.max(np.abs(self.model.intercept_))).astype(np.int8)
        else:
            raise ValueError("Model has not been trained or does not have coef_ and intercept_ attributes.")

    def run_tests(self):
        """
        Run unit tests to verify the quantization process.
        """
        test_results = {
            "Quantized coefficients dtype": None,
            "Quantized intercept dtype": None,
            "Coefficient approximation test": None,
            "Intercept approximation test": None
        }

        # Test 1: Check if quantized weights and intercepts are 8-bit integers
        test_results["Quantized coefficients dtype"] = np.issubdtype(self.quantized_coef_.dtype, np.int8)
        test_results["Quantized intercept dtype"] = np.issubdtype(self.quantized_intercept_.dtype, np.int8)

        # Test 2: Check if the quantization retains the approximate structure of the original model
        approx_coef = self.quantized_coef_.astype(float) * np.max(np.abs(self.model.coef_)) / 127
        approx_intercept = self.quantized_intercept_.astype(float) * np.max(np.abs(self.model.intercept_)) / 127

        test_results["Coefficient approximation test"] = np.allclose(self.model.coef_, approx_coef, atol=0.1)
        test_results["Intercept approximation test"] = np.allclose(self.model.intercept_, approx_intercept, atol=0.1)

        return test_results

# Example usage
# Load iris dataset and train a logistic regression model
iris = load_iris()
X, y = iris.data, iris.target
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
pipeline.fit(X, y)

# Initialize the optimizer with the trained model
optimizer = IrisModelOptimizer(pipeline.named_steps['logisticregression'])

# Perform quantization
optimizer.quantize_model()

# Print quantized model coefficients and intercepts
print("Quantized coefficients:", optimizer.quantized_coef_)
print("Quantized intercept:", optimizer.quantized_intercept_)

# Run tests to verify quantization and print test results
test_results = optimizer.run_tests()
print("Test results:", test_results)
