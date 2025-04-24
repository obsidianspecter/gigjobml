# Gig Worker Job Change Predictor

A machine learning-based web application that predicts the likelihood of gig workers changing jobs based on various factors such as job satisfaction, work-life balance, and market conditions.

## Features

- **Interactive Web Interface**: Modern, responsive UI built with Bootstrap 5
- **Real-time Predictions**: Instant job change probability calculations
- **Data Visualization**: 
  - Probability bar showing likelihood of job change
  - Feature importance chart
  - Key factors affecting the prediction
- **Comprehensive Input Fields**:
  - Personal Information (age, education, experience)
  - Job Satisfaction Metrics (satisfaction, work-life balance, skill diversity)
  - Market Factors (demand, stability, benefits)
  - Additional Factors (commute time, flexibility, training opportunities)
- **Input Validation**: Client-side and server-side validation for all fields
- **Error Handling**: Robust error handling and user feedback
- **Mobile Responsive**: Works seamlessly on all device sizes

## Technology Stack

- **Backend**:
  - Python 3.8+
  - Flask 2.0.1
  - scikit-learn 1.0.2
  - XGBoost 1.4.2
  - pandas 1.3.3
  - numpy 1.24.3

- **Frontend**:
  - HTML5
  - Bootstrap 5.1.3
  - Chart.js
  - Bootstrap Icons

## Project Structure

```
.
├── app.py                 # Flask application
├── gig_worker_analysis.py # Model training and feature engineering
├── gig_worker_model.joblib # Trained model file
├── requirements.txt       # Python dependencies
└── templates/
    └── index.html        # Web interface
```

## Local Development Setup

1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:8080`

## Model Training

The model was trained using the following process:

1. **Data Generation**: Synthetic data generation with realistic patterns
2. **Feature Engineering**: Creation of interaction features and composite scores
3. **Model Selection**: XGBoost classifier with optimized hyperparameters
4. **Training Process**:
   - Data preprocessing (scaling, encoding)
   - SMOTE for handling class imbalance
   - Cross-validation for model evaluation
   - Feature importance analysis

## API Endpoints

### GET /
- Serves the main web interface

### POST /predict
- Accepts JSON data with worker information
- Returns prediction results including:
  - Prediction (0 or 1)
  - Probability
  - Key factors
  - Feature importance

## Input Parameters

### Personal Information
- `age`: 18-100
- `education_level`: ["High School", "Bachelor", "Master", "PhD"]
- `years_experience`: 0-50
- `current_income`: > 0
- `hours_per_week`: 0-168
- `industry`: ["Tech", "Healthcare", "Education", "Retail", "Transportation"]

### Job Satisfaction Metrics (1-10)
- `job_satisfaction`
- `work_life_balance`
- `skill_diversity`
- `market_demand`
- `job_stability`
- `benefits_score`

### Additional Factors
- `commute_time`: 0-180 minutes
- `flexibility_score`: 1-10
- `training_opportunities`: 1-10

## Error Handling

The application includes comprehensive error handling for:
- Invalid input values
- Missing required fields
- Model loading issues
- Prediction errors
- Server errors

## Performance Considerations

- Input validation happens on both client and server side
- Efficient data processing and model inference
- Optimized feature engineering pipeline
- Responsive UI with minimal latency

## Future Improvements

Potential enhancements for future versions:
1. User authentication and saved predictions
2. Historical data tracking
3. More detailed explanations of predictions
4. Additional visualization options
5. API rate limiting
6. Caching for frequently requested predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Bootstrap for the UI framework
- Chart.js for data visualization
- XGBoost for the machine learning model # gigjobml
