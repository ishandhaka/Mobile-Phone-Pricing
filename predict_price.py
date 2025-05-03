import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

# Check if models exist
def check_models_exist():
    required_models = [
        'ensemble_model.pkl',
        'scaler.pkl',
        'nn_model.h5'
    ]
    
    all_exist = True
    for model_file in required_models:
        if not os.path.exists(model_file):
            all_exist = False
            print(f"Missing model file: {model_file}")
    
    if not all_exist:
        print("\nPlease run advanced_price_prediction.py first to generate the model files.")
        return False
    
    return True

# Load the models
def load_models():
    ensemble_model = joblib.load('ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    nn_model = tf.keras.models.load_model('nn_model.h5')
    
    # Try to load feature selector if it exists
    try:
        selector = joblib.load('feature_selector.pkl')
    except:
        selector = None
    
    return ensemble_model, scaler, nn_model, selector

# Generate engineered features
def generate_features(data):
    # Calculate engineered features
    data['px_area'] = data['px_width'] * data['px_height']
    data['sc_area'] = data['sc_w'] * data['sc_h']
    data['ram_per_core'] = data['ram'] / data['n_cores'] if data['n_cores'] > 0 else 0
    data['battery_per_area'] = data['battery_power'] / data['px_area'] if data['px_area'] > 0 else 0
    data['performance_score'] = data['ram'] * data['clock_speed'] * data['n_cores']
    data['camera_total'] = data['pc'] + data['fc']
    
    # Since we don't have the global dataset medians in this script, we'll set high_end based on thresholds
    data['is_high_end'] = 1 if (data['ram'] > 2000 and data['battery_power'] > 1000) else 0
    
    return data

# Get user input
def get_user_input():
    print("\n=== Mobile Phone Price Predictor ===")
    print("Please enter the specifications of your mobile phone:")
    
    try:
        battery_power = float(input("Battery power (mAh): "))
        has_bluetooth = int(input("Has Bluetooth (0/1): "))
        clock_speed = float(input("Clock speed (GHz): "))
        has_dual_sim = int(input("Has dual SIM (0/1): "))
        front_camera_mp = float(input("Front Camera megapixels: "))
        has_4g = int(input("Has 4G (0/1): "))
        internal_memory = float(input("Internal Memory (GB): "))
        mobile_depth = float(input("Mobile Depth (cm): "))
        mobile_weight = float(input("Weight (g): "))
        num_cores = int(input("Number of processor cores: "))
        primary_camera_mp = float(input("Primary Camera megapixels: "))
        pixel_height = float(input("Screen pixel height: "))
        pixel_width = float(input("Screen pixel width: "))
        ram = float(input("RAM (MB): "))
        screen_height = float(input("Screen height (cm): "))
        screen_width = float(input("Screen width (cm): "))
        talk_time = float(input("Talk time (hours): "))
        has_3g = int(input("Has 3G (0/1): "))
        has_touch_screen = int(input("Has touchscreen (0/1): "))
        has_wifi = int(input("Has WiFi (0/1): "))
        
        # Create a dictionary with the input values
        phone_data = {
            'battery_power': battery_power,
            'blue': has_bluetooth,
            'clock_speed': clock_speed,
            'dual_sim': has_dual_sim,
            'fc': front_camera_mp,
            'four_g': has_4g,
            'int_memory': internal_memory,
            'm_dep': mobile_depth,
            'mobile_wt': mobile_weight,
            'n_cores': num_cores,
            'pc': primary_camera_mp,
            'px_height': pixel_height,
            'px_width': pixel_width,
            'ram': ram,
            'sc_h': screen_height,
            'sc_w': screen_width,
            'talk_time': talk_time,
            'three_g': has_3g,
            'touch_screen': has_touch_screen,
            'wifi': has_wifi
        }
        
        return pd.DataFrame([phone_data])
    except ValueError as e:
        print(f"Error: {e}. Please enter a valid number.")
        return None

# Sample data for quick testing
def get_sample_data():
    return pd.DataFrame([{
        'battery_power': 1500,
        'blue': 1,
        'clock_speed': 2.0,
        'dual_sim': 1,
        'fc': 8,
        'four_g': 1,
        'int_memory': 64,
        'm_dep': 0.5,
        'mobile_wt': 150,
        'n_cores': 8,
        'pc': 16,
        'px_height': 1920,
        'px_width': 1080,
        'ram': 4000,
        'sc_h': 15,
        'sc_w': 8,
        'talk_time': 15,
        'three_g': 1,
        'touch_screen': 1,
        'wifi': 1
    }])

# Predict price range using ensemble model
def predict_price_range(data, ensemble_model, scaler, nn_model, selector):
    # Generate engineered features
    data = generate_features(data)
    
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Make predictions with ensemble model
    ensemble_pred = ensemble_model.predict(data_scaled)[0]
    
    # Make predictions with neural network if selector exists
    if selector is not None:
        data_selected = selector.transform(data_scaled)
        nn_pred_probs = nn_model.predict(data_selected)
        nn_pred = np.argmax(nn_pred_probs, axis=1)[0]
    else:
        # Use the neural network directly if there's no selector
        nn_pred_probs = nn_model.predict(data_scaled)
        nn_pred = np.argmax(nn_pred_probs, axis=1)[0]
    
    # Return both predictions
    return ensemble_pred, nn_pred

# Map prediction to price range description
def get_price_range_description(prediction):
    price_ranges = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    return price_ranges.get(prediction, "Unknown")

# Main function
def main():
    # Check if models exist
    if not check_models_exist():
        return
    
    # Load models
    print("Loading prediction models...")
    ensemble_model, scaler, nn_model, selector = load_models()
    
    while True:
        # Ask user if they want to use sample data or enter their own
        choice = input("\nChoose an option:\n1. Use sample phone data\n2. Enter your own phone specifications\n3. Exit\nYour choice (1-3): ")
        
        if choice == '3':
            print("Exiting...")
            break
        
        if choice == '1':
            print("Using sample phone data...")
            phone_data = get_sample_data()
        elif choice == '2':
            phone_data = get_user_input()
            if phone_data is None:
                continue
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue
        
        # Make predictions
        ensemble_pred, nn_pred = predict_price_range(phone_data, ensemble_model, scaler, nn_model, selector)
        
        # Get price range descriptions
        ensemble_desc = get_price_range_description(ensemble_pred)
        nn_desc = get_price_range_description(nn_pred)
        
        # Print results
        print("\n=== Price Prediction Results ===")
        print(f"Ensemble Model Prediction: {ensemble_pred} - {ensemble_desc}")
        print(f"Neural Network Prediction: {nn_pred} - {nn_desc}")
        
        # If models disagree, explain
        if ensemble_pred != nn_pred:
            print("\nNote: The models have different predictions. This suggests the phone has features that place it at the boundary between price categories.")
        
        # Print key features
        print("\nKey Features of This Phone:")
        print(f"- RAM: {phone_data['ram'].values[0]} MB")
        print(f"- Battery: {phone_data['battery_power'].values[0]} mAh")
        print(f"- Primary Camera: {phone_data['pc'].values[0]} MP")
        print(f"- Screen Resolution: {phone_data['px_width'].values[0]}x{phone_data['px_height'].values[0]} pixels")

if __name__ == "__main__":
    main() 