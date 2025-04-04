"""Methods to calculate metrics —— NOT USED CURRENTLY"""

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(y_test, prob_predictions):
    # Convert y_test to binary format if it's multiclass
    # For binary classification, this isn't needed
    if len(np.unique(y_test)) > 2:  # Multiclass
        # One-vs-Rest approach for multiclass
        n_classes = len(np.unique(y_test))
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            # Create binary labels for current class
            y_binary = (y_test == i).astype(int)
            
            # Get probabilities for current class
            if prob_predictions.ndim > 1:
                class_probs = prob_predictions[:, i]
            else:
                # If only one set of probabilities is given (binary case)
                class_probs = prob_predictions
            
            # Calculate ROC curve and AUC
            fpr[i], tpr[i], _ = roc_curve(y_binary, class_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curve for current class
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        # Calculate macro average AUC
        macro_auc = roc_auc_score(y_test, prob_predictions, multi_class='ovr', average='macro')
        print(f"Macro Average AUC: {macro_auc:.4f}")
            
    else:  # Binary classification
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, prob_predictions)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        # Print AUC score
        print(f"AUC: {roc_auc:.4f}")
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    return plt

