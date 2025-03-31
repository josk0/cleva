from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def compute_f1_score(y_true, y_pred, pos_label=None, average='binary'):
    """
    Compute F1 score handling both numeric and string labels.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    pos_label : str or int, default=None
        The label of the positive class. When None, if binary classification and
        string labels are detected, uses the most common string label in y_true.
    average : {'binary', 'micro', 'macro', 'weighted', None}, default='binary'
        This parameter is required for multiclass/multilabel targets.
        
    Returns:
    --------
    f1 : float
        F1 score of the positive class (binary case) or averaged F1 score (multiclass).
    """
    
    # Convert inputs to numpy arrays for easier handling
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Check if string labels are present
    has_string_labels = any(isinstance(label, str) for label in np.unique(y_true_array))
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate((y_true_array, y_pred_array)))
    is_multiclass = len(unique_classes) > 2
    
    if has_string_labels:
        print("String labels detected. Converting to numeric values for F1 computation.")
        le = LabelEncoder()
        le.fit(unique_classes)
        y_true_encoded = le.transform(y_true_array)
        y_pred_encoded = le.transform(y_pred_array)
        
        # If pos_label is provided and it's a string, convert it to numeric
        if pos_label is not None and isinstance(pos_label, str):
            pos_label = le.transform([pos_label])[0]
        
        # For binary classification with no pos_label specified, use the less frequent class
        if not is_multiclass and pos_label is None and average == 'binary':
            from collections import Counter
            # Find the less frequent class in the original labels
            class_counts = Counter(y_true)
            # Get the numeric label for the less frequent class
            less_frequent_class = min(class_counts.items(), key=lambda x: x[1])[0]
            pos_label = le.transform([less_frequent_class])[0]
            print(f"No pos_label specified. Using '{less_frequent_class}' (encoded as {pos_label}) as positive class.")
        
        if is_multiclass and average == 'binary':
            print("Multiple classes detected. Changing average to 'weighted'.")
            average = 'weighted'
            
        return f1_score(y_true_encoded, y_pred_encoded, pos_label=pos_label, average=average)
    else:
        # Numeric labels, use f1_score directly
        if is_multiclass and average == 'binary':
            print("Multiple classes detected. Changing average to 'weighted'.")
            average = 'weighted'
        
        return f1_score(y_true, y_pred, pos_label=pos_label, average=average)