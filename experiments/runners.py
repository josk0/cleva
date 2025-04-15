"""The Run class to prepare model and data together, exposing behavior for fit, predict, and score"""

from typing import Callable, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from models.utils import get_pipeline, bounded_train_test_split
from evaluation.metrics import plot_roc_curve


class Run:
    """Class to manage model experiments including data preparation, training, and evaluation."""
    
    def __init__(
        self, 
        model: BaseEstimator, 
        data_loader: Callable[[], Tuple[pd.DataFrame, np.ndarray]],
        experiment_name: str = "Prototyping Experiment"
    ) -> None:
        """
        Initialize a model experiment run.
        
        Args:
            model: A scikit-learn compatible model
            data_loader: Function that returns (X, y) when called
            experiment_name: Name for this experiment
        """
        self.model = model
        self.model_name = type(model).__name__
        # For TabPFN which has a 10k sample limit
        self.model_max_sample_length = 10000 if "TabPFN" in self.model_name else None
        
        print("Loading data...")
        self.X, self.y = data_loader()
        print(f"Loaded {len(self.X)} samples with {len(self.X.columns)} features.")
        print(f"And length of Y is: {len(self.y)}")

        print(f"Splitting data for {self.model_name}...")
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(
            self.X, self.y, test_size=0.33, random_state=42
        )

        # Free up memory after split
        del self.X
        del self.y

        print(f"Setting up pipeline for {self.model_name}...")
        self.pipe = get_pipeline(self)

        # Initialize predictions as None
        self.predictions = None
        self.prob_predictions = None

    def fit(self) -> 'Run':
        """
        Fit the model to the training data.
        
        Returns:
            self: For method chaining
        """
        print(f"Fitting {self.model_name} on {len(self.X_train)} samples...")
        self.pipe.fit(self.X_train, self.y_train)
        print(f"DONE: {self.model_name} fit")
        return self

    def predict(self, classes_only: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on test data.
        
        Args:
            classes_only: If True, only return class predictions, not probabilities
            
        Returns:
            Either class predictions only (if classes_only=True) or a tuple of
            (class_predictions, probability_predictions)
        """
        self.predictions = self.pipe.predict(self.X_test)
        print(f"DONE: {self.model_name} class prediction")
        
        if not classes_only:
            try:
                if hasattr(self.pipe, 'predict_proba'):
                    self.prob_predictions = self.pipe.predict_proba(self.X_test)
                    print(f"DONE: {self.model_name} probability prediction")
                else:
                    print(f"Warning: {self.model_name} does not support predict_proba")
            except Exception as e:
                print(f"Error in probability prediction: {e}")

        if classes_only:
            return self.predictions
        else:
            return self.predictions, self.prob_predictions

    def score(self) -> None:
        """Evaluate model performance with various metrics and visualizations."""
        if self.predictions is None:
            self.predict()
            
        report_dict = classification_report(self.y_test, self.predictions, output_dict=True)
        print(f"Accuracy: {report_dict['accuracy']:.4f}")
        print(f"F1 Score (macro avg): {report_dict['macro avg']['f1-score']:.4f}")

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions, digits=2, output_dict=False))

        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, ax=ax)
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.tight_layout()

        # Save the figure
        confusion_matrix_file = f"confusion_matrix_{self.model_name}.png"
        plt.savefig(confusion_matrix_file)
        plt.close(fig)

    def _split_data(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        test_size: float = 0.5, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets based on model constraints.
        
        Args:
            X: Feature data
            y: Target data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test: Split data
        """
        if self.model_max_sample_length is not None:
            # Use the version for models like TabPFN that have a limited sample length
            return bounded_train_test_split(
                X, y, 
                max_length=self.model_max_sample_length,  
                test_size=test_size, 
                random_state=random_state,
                stratify=y if len(np.unique(y)) < 10 else None  # Stratify if not too many classes
            )
        else:
            # Use regular train_test_split for other models
            return train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y if len(np.unique(y)) < 10 else None  # Stratify if not too many classes
            )