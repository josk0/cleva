import mlflow

def setup_mlflow(model_name="TabPFN", dataset_name="US Visa data", experiment_name="MLflow Quickstart"):
  mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
  mlflow.set_experiment(experiment_name)
  mlflow.set_tag("Dataset", dataset_name)
  mlflow.set_tag("model", model_name)

def log_classification_report_metrics(report_dict):
    """
    Log each metric from a classification report dictionary to MLflow.
    
    Parameters:
    -----------
    report_dict : dict
        The dictionary returned by sklearn.metrics.classification_report with output_dict=True
    """
    # Loop through the top level dictionary
    for class_key, metrics in report_dict.items():
        # Check if the value is a dictionary (for class metrics)
        if isinstance(metrics, dict):
            # Log metrics for each class
            for metric_name, value in metrics.items():
                # Create a named metric with class and metric info
                metric_full_name = f"{metric_name}_{class_key}"
                mlflow.log_metric(metric_full_name, value)
        else:
            # For overall metrics like accuracy that aren't nested
            mlflow.log_metric(class_key, metrics)