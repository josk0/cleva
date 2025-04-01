from data import load_us_perm_visas 
import evaluation.reporter as reporter
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier
import mlflow
import pandas as pd


def run_run(model=None):
  dataset_name="US Visa data"
  # Setup MLflow experiment
  reporter.setup_mlflow(experiment_name="Prototying Experiment")

  if model is None:
    print("No model provided. Using default TabPFNClassifier.")
    model = TabPFNClassifier()
  if isinstance(model, TabPFNClassifier):
    model_name = "TabPFN"
    tabpfn_run = True
  elif isinstance(model, RandomForestClassifier):
   model_name = "RandomForest"
   tabpfn_run = False
  else:
    raise ValueError("Model must be either TabPFN or RandomForest")

  X, y = load_us_perm_visas(for_tabpfn=tabpfn_run) # need to set manually true or false here because no use of pipelines

  # Drop datetime columns
  datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_dtype(X[col])]
  for col in datetime_cols:
    X = X.drop(columns=[col])

  # Drop irrelevant date columns, I think this only applies to TabPFN data 
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
  if tabpfn_run:
    X_subset = X[:20000]
    y_subset = y[:20000]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.5, random_state=42)
    # classifier is just the model
    clf = model
  
  else: # For RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify column types
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    numerical_cols = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Create classifier pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

  
  # Start an MLflow run
  with mlflow.start_run():
    mlflow.set_tag("Dataset", dataset_name)
    mlflow.set_tag("model", model_name)
    # Run pipeline
    clf.fit(X_train, y_train)

    # Predict labels
    predictions = clf.predict(X_test)
    # TabPFN takes about 27 min or 21 without dates
    # RandomForest takes about 12 min

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

  print(f"DONE: {model_name} run")
  