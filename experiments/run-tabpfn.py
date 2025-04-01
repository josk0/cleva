from data import load_us_perm_visas 
import evaluation.reporter as reporter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier
import mlflow


def run_tabpfn():
  # Setup MLflow experiment
  reporter.setup_mlflow(model_name="TabPFN", 
                        dataset_name="US Visa data",
                        experiment_name="Prototying Experiment")

  X, y = load_us_perm_visas(for_tabpfn=True) # need to set manually true or false here because no use of pipelines

  # Drop irrelevant date columns
  irrelevant_date_columns = ['decision_date_year', 
                            'decision_date_month', 
                            'decision_date_day', 
                            'decision_date_dayofweek']
  for col in irrelevant_date_columns:
    if col in X.columns:
      X.drop(col, axis=1, inplace=True)

  # Ignore FutureWarning, of which TabPFNClassifier has a lot!
  warnings.filterwarnings("ignore", category=FutureWarning)

  ## Need to limit size of sample for TabPFN
  X_subset = X[:20000]
  y_subset = y[:20000]
  
  X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.5, random_state=42)

  # Start an MLflow run
  with mlflow.start_run():

    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    # Predict labels
    predictions = clf.predict(X_test)
    # about 27 min...
    # or 21 without dates

    print("Accuracy", accuracy_score(y_test, predictions))

    # Calculate and print F1 score
    print("F1 Score:", f1_score(y_test, predictions, average='macro'))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=3, output_dict=False))

    report_dict = classification_report(y_test, predictions, output_dict=True)
    reporter.log_classification_report_metrics(report_dict)

    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
    plt.tight_layout()

    # Save and log the figure
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

  print("DONE: TabPFT run")
  