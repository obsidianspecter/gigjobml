from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
from gig_worker_analysis import engineer_features
import os

app = Flask(__name__)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'gig_worker_model.joblib')
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def validate_input_data(data):
    required_fields = {
        'age': (float, 18, 100),
        'education_level': (str, None, None),
        'years_experience': (float, 0, 50),
        'current_income': (float, 0, None),
        'hours_per_week': (float, 0, 168),
        'job_satisfaction': (float, 1, 10),
        'work_life_balance': (float, 1, 10),
        'skill_diversity': (float, 1, 10),
        'market_demand': (float, 1, 10),
        'job_stability': (float, 1, 10),
        'benefits_score': (float, 1, 10),
        'commute_time': (float, 0, 180),
        'flexibility_score': (float, 1, 10),
        'training_opportunities': (float, 1, 10),
        'industry': (str, None, None)
    }
    
    valid_education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    valid_industries = ['Tech', 'Healthcare', 'Education', 'Retail', 'Transportation']
    
    errors = []
    
    for field, (field_type, min_val, max_val) in required_fields.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
            
        try:
            value = field_type(data[field])
            
            if field == 'education_level' and value not in valid_education_levels:
                errors.append(f"Invalid education level. Must be one of: {', '.join(valid_education_levels)}")
            elif field == 'industry' and value not in valid_industries:
                errors.append(f"Invalid industry. Must be one of: {', '.join(valid_industries)}")
            elif min_val is not None and value < min_val:
                errors.append(f"{field} must be greater than or equal to {min_val}")
            elif max_val is not None and value > max_val:
                errors.append(f"{field} must be less than or equal to {max_val}")
                
        except (ValueError, TypeError):
            errors.append(f"Invalid value for {field}. Expected {field_type.__name__}")
    
    return errors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    # Return a 204 No Content response for favicon requests
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Create input array for prediction
        input_data = np.array([[
            float(data['age']),
            float(data['job_satisfaction']),
            float(data['work_life_balance']),
            float(data['stress_level']),
            float(data['daily_working_hours']),
            float(data['years_of_experience']),
            float(data['monthly_income'])
        ]])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # Get feature importance
        feature_importance = model.feature_importances_
        feature_names = ['Age', 'Job Satisfaction', 'Work Life Balance', 'Stress Level',
                        'Daily Working Hours', 'Years of Experience', 'Monthly Income']
        
        # Create a list of (feature_name, importance) tuples
        importance_pairs = list(zip(feature_names, feature_importance))
        # Sort by importance in descending order
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 most important factors
        key_factors = [{'factor': pair[0], 'importance': float(pair[1])} 
                      for pair in importance_pairs[:3]]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'key_factors': key_factors,
            'feature_importance': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(feature_names, feature_importance)
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 