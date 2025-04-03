from models.utils import get_pipeline, subsample_train_test_split
# import evaluation.reporter as reporter
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
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
    # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
    self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(self.X, self.y, test_size=0.4, random_state=42)

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

  def predict(self):
    self.predictions = self.pipe.predict(self.X_test)
    print(f"DONE: {self.model_name} predict")

  def score(self):
    print("Accuracy", accuracy_score(self.y_test, self.predictions))

    # Calculate and print F1 score
    print("F1 Score:", f1_score(self.y_test, self.predictions, average='macro'))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(self.y_test, self.predictions, digits=3, output_dict=False))

    # Reporting not working currently
    # report_dict = classification_report(self.y_test, self.predictions, output_dict=True)
    # reporter.log_classification_report_metrics(report_dict)

    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, ax=ax)
    plt.tight_layout()

    # Save and log the figure
    plt.savefig("confusion_matrix.png")
    # mlflow.log_artifact("confusion_matrix.png")

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