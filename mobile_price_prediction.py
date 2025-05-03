import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('dataset.csv')

# Exploratory Data Analysis
print("Dataset Shape:", df.shape)
print("\nFeature Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df['price_range'].value_counts())

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='price_range', data=df)
plt.title('Distribution of Price Range Categories')
plt.savefig('price_range_distribution.png')
plt.close()

# Analyze correlations
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Top features correlation with target
target_correlation = correlation_matrix['price_range'].sort_values(ascending=False)
print("\nFeature Correlation with Price Range:")
print(target_correlation)

# Visualize top 5 features with highest correlation
top_features = target_correlation[1:6].index.tolist()
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='price_range', y=feature, data=df)
    plt.title(f'{feature} vs Price Range')
plt.tight_layout()
plt.savefig('top_features_boxplot.png')
plt.close()

# Prepare data for modeling
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(4), yticklabels=range(4))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    return pipeline, accuracy

# Train different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}
for name, model in models.items():
    trained_model, accuracy = train_evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results[name] = {
        'model': trained_model,
        'accuracy': accuracy
    }

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Feature importance for Random Forest (if it's the best model)
if best_model_name == 'Random Forest':
    feature_importances = best_model.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Hyper-parameter tuning for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'SVM':
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto', 0.1, 0.01],
        'model__kernel': ['rbf', 'linear']
    }

print(f"\nTuning hyperparameters for {best_model_name}...")
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_report = classification_report(y_test, y_pred_tuned)

print(f"\nTuned {best_model_name} Results:")
print(f"Accuracy: {tuned_accuracy:.4f}")
print("\nClassification Report:")
print(tuned_report)

# Plot the confusion matrix for the tuned model
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=range(4), yticklabels=range(4))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - Tuned {best_model_name}')
plt.savefig(f'confusion_matrix_tuned_{best_model_name.replace(" ", "_").lower()}.png')
plt.close()

print("\nMobile Phone Price Prediction Analysis Complete.") 