import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import joblib
from gig_worker_analysis import generate_gig_worker_data, engineer_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme()

def plot_learning_curves(model, X, y, title="Learning Curves"):
    """Plot learning curves to show how the model performs with different training set sizes."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve for the model."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Generate data
    print("Generating dataset...")
    df = generate_gig_worker_data(n_samples=10000)
    df = engineer_features(df)
    
    # Split features and target
    X = df.drop('will_change_job', axis=1)
    y = df['will_change_job']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply preprocessing
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE to training data only
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    # Create and train XGBoost classifier with optimized parameters
    print("Training model...")
    classifier = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        random_state=42
    )
    
    # Train the model on balanced data
    classifier.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = classifier.predict(X_test_processed)
    y_pred_proba = classifier.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(classifier, X_test_processed, y_test)
    
    # Plot ROC curve
    print("Generating ROC curve...")
    plot_roc_curve(classifier, X_test_processed, y_test)
    
    # Plot feature importance
    print("Generating feature importance plot...")
    feature_names = (
        list(numeric_features) +
        [f"{feature}_{val}" for feature, vals in 
         zip(categorical_features, preprocessor.named_transformers_['cat'].categories_) 
         for val in vals]
    )
    plot_feature_importance(classifier, feature_names)
    
    # Save the model and preprocessor
    print("\nSaving model and preprocessor...")
    joblib.dump(classifier, 'model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    
    # Generate learning curves
    print("Generating learning curves...")
    plot_learning_curves(classifier, X_train_processed, y_train)
    
    print("Analysis complete. Check the generated plots for detailed insights.")

if __name__ == "__main__":
    main() 