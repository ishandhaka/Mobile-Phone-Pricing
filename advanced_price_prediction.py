import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
df = pd.read_csv('dataset.csv')

# Feature Engineering
# Create new features that might be useful for prediction
df['px_area'] = df['px_width'] * df['px_height']  # Screen pixel area
df['sc_area'] = df['sc_w'] * df['sc_h']  # Screen area in cm²
df['ram_per_core'] = df['ram'] / df['n_cores']  # RAM per core
df['battery_per_area'] = df['battery_power'] / df['px_area'].replace(0, 1)  # Battery power per pixel area
df['performance_score'] = df['ram'] * df['clock_speed'] * df['n_cores']  # Overall performance score
df['camera_total'] = df['pc'] + df['fc']  # Total camera megapixels
df['is_high_end'] = ((df['ram'] > df['ram'].median()) & 
                     (df['battery_power'] > df['battery_power'].median()) & 
                     (df['px_area'] > df['px_area'].median())).astype(int)

# Print new features
print("Added new engineered features:")
for feature in ['px_area', 'sc_area', 'ram_per_core', 'battery_per_area', 
                'performance_score', 'camera_total', 'is_high_end']:
    print(f"- {feature}")

# Exploratory Data Analysis
print("\nData Overview:")
print(f"Dataset Shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")

# Check for missing values
print("\nMissing Values:", df.isnull().sum().sum())

# Display target distribution
print("\nTarget Variable Distribution:")
print(df['price_range'].value_counts(normalize=True) * 100)

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='price_range', data=df)
plt.title('Distribution of Price Range Categories')
plt.savefig('price_range_distribution.png')
plt.close()

# Correlation heatmap of top features
plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
corr_with_target = abs(corr_matrix['price_range']).sort_values(ascending=False)
top_features = corr_with_target[1:16].index  # Top 15 correlated features

# Create a heatmap of top features
sns.heatmap(df[list(top_features) + ['price_range']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Top Features')
plt.tight_layout()
plt.savefig('top_features_correlation.png')
plt.close()

# Pairplot of top 4 numerical features against target
top_4_features = top_features[:4]
plt.figure(figsize=(20, 15))
for i, feature in enumerate(top_4_features, 1):
    plt.subplot(2, 2, i)
    for j in range(4):  # 4 price ranges
        subset = df[df['price_range'] == j]
        plt.scatter(subset[feature], subset['price_range'], 
                    alpha=0.6, label=f'Price Range {j}')
    plt.xlabel(feature)
    plt.ylabel('Price Range')
    plt.title(f'{feature} vs Price Range')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('top_features_scatter.png')
plt.close()

# Data Preparation
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=20)  # Select top 20 features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("\nTop 20 Selected Features:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# Define the neural network model
def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')  # 4 price ranges
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Learning rate reduction on plateau
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.2,
    min_lr=0.00001
)

# Train the neural network
nn_model = create_nn_model(X_train_selected.shape[1])
history = nn_model.fit(
    X_train_selected, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

# Save the neural network model
nn_model.save('nn_model.h5')

# Evaluate the neural network
nn_predictions = nn_model.predict(X_test_selected)
nn_predictions = np.argmax(nn_predictions, axis=1)
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_report = classification_report(y_test, nn_predictions)

print("\nNeural Network Results:")
print(f"Accuracy: {nn_accuracy:.4f}")
print("\nClassification Report:")
print(nn_report)

# Create and train a Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Create and train a Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_predictions = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Print results for all models
print("\nModel Accuracies:")
print(f"Neural Network: {nn_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")
print(f"Gradient Boosting: {gb_accuracy:.4f}")

# Visualize learning curves for Neural Network
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig('nn_learning_curves.png')
plt.close()

# Confusion Matrix for Neural Network
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, nn_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(4), yticklabels=range(4))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Neural Network')
plt.savefig('confusion_matrix_nn.png')
plt.close()

# Feature Importance from Random Forest
plt.figure(figsize=(12, 10))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.close()

# Create an ensemble model (Voting Classifier)
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft'
)
voting_model.fit(X_train_scaled, y_train)
voting_predictions = voting_model.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, voting_predictions)
voting_report = classification_report(y_test, voting_predictions)

print("\nEnsemble Model Results:")
print(f"Accuracy: {voting_accuracy:.4f}")
print("\nClassification Report:")
print(voting_report)

# Save models for future use
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(voting_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

# Function to make predictions for new data
def predict_price_range(new_data):
    """
    Make price range predictions for new mobile phone data.
    
    Args:
        new_data: DataFrame with the same features as the training data
    
    Returns:
        Predictions using the ensemble model
    """
    # Engineer features for the new data
    new_data['px_area'] = new_data['px_width'] * new_data['px_height']
    new_data['sc_area'] = new_data['sc_w'] * new_data['sc_h']
    new_data['ram_per_core'] = new_data['ram'] / new_data['n_cores']
    new_data['battery_per_area'] = new_data['battery_power'] / new_data['px_area'].replace(0, 1)
    new_data['performance_score'] = new_data['ram'] * new_data['clock_speed'] * new_data['n_cores']
    new_data['camera_total'] = new_data['pc'] + new_data['fc']
    new_data['is_high_end'] = ((new_data['ram'] > df['ram'].median()) & 
                              (new_data['battery_power'] > df['battery_power'].median()) & 
                              (new_data['px_area'] > df['px_area'].median())).astype(int)
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = voting_model.predict(new_data_scaled)
    
    return predictions

print("\nMobile Phone Price Range Analysis Complete.")

# Example of model use with a sample phone
print("\nExample Prediction for a Sample Phone:")
sample_phone = pd.DataFrame({
    'battery_power': [1500],
    'blue': [1],
    'clock_speed': [2.0],
    'dual_sim': [1],
    'fc': [8],
    'four_g': [1],
    'int_memory': [64],
    'm_dep': [0.5],
    'mobile_wt': [150],
    'n_cores': [8],
    'pc': [16],
    'px_height': [1920],
    'px_width': [1080],
    'ram': [4000],
    'sc_h': [15],
    'sc_w': [8],
    'talk_time': [15],
    'three_g': [1],
    'touch_screen': [1],
    'wifi': [1]
})

# Add engineered features
sample_phone['px_area'] = sample_phone['px_width'] * sample_phone['px_height']
sample_phone['sc_area'] = sample_phone['sc_w'] * sample_phone['sc_h']
sample_phone['ram_per_core'] = sample_phone['ram'] / sample_phone['n_cores']
sample_phone['battery_per_area'] = sample_phone['battery_power'] / sample_phone['px_area']
sample_phone['performance_score'] = sample_phone['ram'] * sample_phone['clock_speed'] * sample_phone['n_cores']
sample_phone['camera_total'] = sample_phone['pc'] + sample_phone['fc']
sample_phone['is_high_end'] = ((sample_phone['ram'] > df['ram'].median()) & 
                              (sample_phone['battery_power'] > df['battery_power'].median()) & 
                              (sample_phone['px_area'] > df['px_area'].median())).astype(int)

# Make prediction
sample_scaled = scaler.transform(sample_phone)
predicted_price_range = voting_model.predict(sample_scaled)[0]
print(f"Sample Phone Predicted Price Range: {predicted_price_range}")
print(f"Price Category: {'Low' if predicted_price_range == 0 else 'Medium' if predicted_price_range == 1 else 'High' if predicted_price_range == 2 else 'Very High'}") 