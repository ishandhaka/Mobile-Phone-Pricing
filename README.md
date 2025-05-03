# Mobile Phone Price Prediction

This project predicts the price range category of mobile phones based on their features. The price ranges are categorized into 4 classes (0-3).

## Dataset Information

The dataset (`dataset.csv`) contains various mobile phone features, including:

- `battery_power`: Total energy a battery can store (mAh)
- `blue`: Whether the phone has Bluetooth (0/1)
- `clock_speed`: Speed at which microprocessor executes instructions
- `dual_sim`: Whether dual SIM is supported (0/1)
- `fc`: Front Camera megapixels
- `four_g`: Whether 4G is supported (0/1)
- `int_memory`: Internal Memory in GB
- `m_dep`: Mobile Depth in cm
- `mobile_wt`: Weight of mobile phone
- `n_cores`: Number of processor cores
- `pc`: Primary Camera megapixels
- `px_height`: Pixel Resolution Height
- `px_width`: Pixel Resolution Width
- `ram`: Random Access Memory in MB
- `sc_h`: Screen Height of mobile in cm
- `sc_w`: Screen Width of mobile in cm
- `talk_time`: Longest time a single battery charge will last
- `three_g`: Whether 3G is supported (0/1)
- `touch_screen`: Whether touchscreen is present (0/1)
- `wifi`: Whether WiFi is supported (0/1)
- `price_range`: Target variable with values 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost)

## Setup Instructions

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the analysis scripts:
   ```
   # Basic analysis
   python mobile_price_prediction.py
   
   # Advanced analysis with neural networks
   python advanced_price_prediction.py
   
   # Interactive prediction interface
   python predict_price.py
   ```

## Project Components

### 1. Basic Analysis (`mobile_price_prediction.py`)

- Exploratory Data Analysis (EDA)
- Basic model building (Logistic Regression, Random Forest, SVM)
- Model evaluation and hyperparameter tuning

### 2. Advanced Analysis (`advanced_price_prediction.py`)

- Feature engineering to create new predictive features:
  - Pixel area, screen area, RAM per core, etc.
  - Performance score combining multiple specifications
  - High-end classification based on key features
- Deep learning with neural networks using TensorFlow
- Ensemble modeling (Random Forest + Gradient Boosting)
- Advanced model evaluation and visualization
- Model persistence for future use

### 3. Prediction Interface (`predict_price.py`)

- Interactive command-line interface for price predictions
- Option to use sample data or enter custom specifications
- Displays predictions from multiple models
- Provides explanations for price range predictions

## Analysis Overview

The scripts perform the following analyses:

1. **Exploratory Data Analysis (EDA)**
   - Basic dataset statistics
   - Missing value detection
   - Feature correlation analysis
   - Visual representation of important features

2. **Feature Engineering**
   - Creating derived features to improve model performance
   - Feature selection to identify the most impactful specifications

3. **Model Building and Evaluation**
   - Training multiple model types:
     - Traditional ML: Logistic Regression, Random Forest, SVM
     - Ensemble Methods: Gradient Boosting, Voting Classifier
     - Deep Learning: Neural Networks with TensorFlow
   - Evaluation metrics: accuracy, classification report, confusion matrix
   - Identifying the best performing models

4. **Hyperparameter Tuning**
   - Optimizing models with grid search and randomized search
   - Early stopping and learning rate scheduling for neural networks
   - Final model evaluation

## Output Files

The analysis generates several visualization files:
- `price_range_distribution.png`: Distribution of the target variable
- `correlation_matrix.png`: Heat map of feature correlations
- `top_features_boxplot.png`: Relationship between top features and price range
- `confusion_matrix_*.png`: Confusion matrices for each model
- `feature_importance_rf.png`: Feature importance rankings for Random Forest
- `nn_learning_curves.png`: Training and validation learning curves for neural network

## Model Files

The advanced analysis saves several model files that can be used for future predictions:
- `nn_model.h5`: TensorFlow neural network model
- `random_forest_model.pkl`: Scikit-learn Random Forest model
- `gradient_boosting_model.pkl`: Scikit-learn Gradient Boosting model
- `ensemble_model.pkl`: Voting Classifier ensemble model
- `scaler.pkl`: StandardScaler for feature normalization
- `feature_selector.pkl`: Feature selector used for the neural network

## Results

The analysis identifies the key factors that influence mobile phone pricing and creates accurate models for predicting price ranges. The models achieve high accuracy by leveraging both the original features and engineered features that capture important relationships in the data.

## Future Improvements

Potential enhancements to the project:
- Web-based user interface for predictions
- Integration with current market data for more accurate pricing
- Support for comparative analysis of multiple phone models
- Deployment as a web service/API 