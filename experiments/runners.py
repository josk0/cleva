"""The run class to prepare model and data together, exposing behavior for fit, predict, and score"""

from models.utils import get_pipeline, subsample_train_test_split
from evaluation.metrics import plot_roc_curve
# import evaluation.reporter as reporter # <-- not working, will maybe scrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


class run:
  def __init__(self, model, data_loader, experiment_name="Prototying Experiment"):
    self.model = model
    self.model_name = type(model).__name__
    # For logging/reporting: need to implement a way to set dataset_name
    # self.dataset_name = 
    self.model_max_sample_length = 10000 if "TabPFN" in self.model_name else None
    print("loading data")
    self.X, self.y = data_loader()
    # reporter.setup_mlflow(experiment_name) # not sure how that behaves if we have several instances of this class!!
    print("trying get pipeline")
    self.pipe = get_pipeline(self)
    self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(self.X, self.y, test_size=0.33, random_state=42)

    # Free up memory and avoid leaks after split
    del self.X
    del self.y

  def fit(self):
    print("Fitting model...")
    # with mlflow.start_run():
    # mlflow.set_tag("Dataset", dataset_name)
    # mlflow.set_tag("model", model_name)
    self.pipe.fit(self.X_train, self.y_train)
    print(f"DONE: {self.model_name} fit")

  def predict(self, classes_only=False):
    self.predictions = self.pipe.predict(self.X_test)
    print(f"DONE: {self.model_name} class prediction")
    
    if not classes_only and hasattr(self.model, 'predict_proba'):
       self.prob_predictions = self.pipe.predict_proba(self.X_test)
       print(f"DONE: {self.model_name} probability prediction")

  def score(self):
    report_dict = classification_report(self.y_test, self.predictions, output_dict=True)
    print(f"Accuracy: {report_dict['accuracy']:.4f}")
    print(f"F1 Score (macro avg): {report_dict['macro avg']['f1-score']:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(self.y_test, self.predictions, digits=3, output_dict=False))

    # Reporting not working currently, MLflow integration
    # reporter.log_classification_report_metrics(report_dict)

    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, ax=ax)
    plt.tight_layout()

    # Save and log the figure
    plt.savefig("confusion_matrix.png")
    # mlflow.log_artifact("confusion_matrix.png")

    # Draw and show ROC Plot -- not working, need to fix dimensions of array
    # roc_plot = plot_roc_curve(self.y_test, self.prob_predictions)
    # roc_plot.show()

  def _split_data(self, X, y, test_size=0.5, random_state=42):
    if self.model_max_sample_length is not None:
        # Use the version for models like TabPFN that have a limited sample length
        return subsample_train_test_split(
            X, y, 
            max_length=self.model_max_sample_length,  
            test_size=test_size, 
            random_state=random_state
        )
    else:
        # Use regular train_test_split for other models
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )