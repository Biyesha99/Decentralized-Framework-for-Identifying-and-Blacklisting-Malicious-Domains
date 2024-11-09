import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Ensure the preprocessing script can be found
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_preprocess_data

def main():
    # Get list of all CSV files
    file_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../data/url/results/*.csv'))

    # Load and preprocess the data
    X, y, label_encoders = load_and_preprocess_data(file_paths)

    print(f"Original data shapes: X.shape = {X.shape}, y.shape = {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train data shapes: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"Test data shapes: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: X_train_res.shape = {X_train_res.shape}, y_train_res.shape = {y_train_res.shape}")

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # Make predictions with probabilities
    y_pred_prob = model.predict_proba(X_test)

    # Check class probabilities
    print(y_pred_prob[:5])  # Print the first 5 predictions with probabilities

    # Set a custom threshold for classification 
    threshold = 0.5  # Default threshold
    y_pred_custom = (y_pred_prob[:, 1] >= threshold).astype(int)  # Class 1 is malicious

    # Evaluate the model using the custom predictions
    accuracy = accuracy_score(y_test, y_pred_custom)
    report = classification_report(y_test, y_pred_custom, zero_division=1)

    print(f'Accuracy with custom threshold: {accuracy}')
    print('Classification Report:')
    print(report)

    def plot_confusion_matrix(y_test, y_pred):
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Benign', 'Malicious'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    plot_confusion_matrix(y_test, y_pred_custom)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_res, y_train_res)

    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_}')

    # Save the trained model and label encoders
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'trained_model.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

if __name__ == '__main__':
    main()

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Ensure the preprocessing script can be found
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_preprocess_data

def main():
    # Get list of all CSV files
    file_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../data/url/results/*.csv'))

    # Load and preprocess the data
    X, y, label_encoders = load_and_preprocess_data(file_paths)

    print(f"Original data shapes: X.shape = {X.shape}, y.shape = {y.shape}")

    # Train-test split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train data shapes: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"Test data shapes: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: X_train_res.shape = {X_train_res.shape}, y_train_res.shape = {y_train_res.shape}")

    # Initializing a RandomForestClassifier for GridSearch
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_res, y_train_res)

    # Using the best parameters found in GridSearch
    best_params = grid_search.best_params_
    print(f'Best parameters found: {best_params}')
    print(f'Best cross-validation score: {grid_search.best_score_}')

    # Retrain model with best parameters
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train_res, y_train_res)

    # Make predictions on test data
    y_pred_prob = best_model.predict_proba(X_test)

    # Custom threshold for predicting class based on probability
    threshold = 0.5  # Adjust threshold if necessary
    y_pred_custom = (y_pred_prob[:, 1] >= threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_custom)
    report = classification_report(y_test, y_pred_custom, zero_division=1)

    print(f'Accuracy with custom threshold: {accuracy}')
    print('Classification Report:')
    print(report)

    # Plot the confusion matrix
    def plot_confusion_matrix(y_test, y_pred):
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Benign', 'Malicious'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    plot_confusion_matrix(y_test, y_pred_custom)

    # Save the trained model and label encoders
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'optimized_trained_model.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

if __name__ == '__main__':
    main()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Ensure the preprocessing script can be found
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_preprocess_data

def main():
    # Get list of all CSV files
    file_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../data/url/results/*.csv'))
    
    # Load and preprocess the data
    X, y, label_encoders = load_and_preprocess_data(file_paths)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Define a parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Grid search to find the best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_res, y_train_res)

    # Apply the best model found in grid search
    best_model = grid_search.best_estimator_
    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_}')

    # Train the model with best parameters on the full resampled training set
    best_model.fit(X_train_res, y_train_res)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    
    # Plot the confusion matrix
    def plot_confusion_matrix(y_test, y_pred):
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Benign', 'Malicious'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    plot_confusion_matrix(y_test, y_pred)

    # Save the trained model and label encoders
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'trained_model.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

if __name__ == '__main__':
    main()
"""