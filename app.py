import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class IrisDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the IrisDataProcessor with options for test size and random state.
        Loads and scales the Iris dataset.
        """
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
        """
        Scales features and splits the dataset into training and test sets.
        """
        # Scale features
        self.data[self.data.columns] = self.scaler.fit_transform(self.data)
        
        # Split data into training and test sets
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=self.random_state
        )

    def get_feature_stats(self):
        """
        Provides basic statistical analysis (mean, std, min, max) for each feature.
        Returns:
            A DataFrame containing statistics for each feature in the dataset.
        """
        stats = self.data.describe().T[['mean', 'std', 'min', 'max']]
        return stats

# Initialize the processor
processor = IrisDataProcessor(test_size=0.3, random_state=123)

# Prepare data (scaling and splitting)
processor.prepare_data()

# Access training and testing data and targets
X_train, X_test = processor.train_data, processor.test_data
y_train, y_test = processor.train_target, processor.test_target

# Get statistical information on the features
feature_stats = processor.get_feature_stats()
print(feature_stats)
