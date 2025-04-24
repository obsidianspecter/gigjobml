import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def generate_gig_worker_data(n_samples=1000, noise_factor=0.1):
    np.random.seed(42)
    
    # Base data generation with more realistic distributions
    data = {
        'age': np.random.normal(35, 8, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'years_experience': np.random.normal(5, 3, n_samples),
        'current_income': np.random.normal(2500, 500, n_samples),
        'hours_per_week': np.random.normal(25, 8, n_samples),
        'job_satisfaction': np.random.normal(7, 1.5, n_samples),
        'work_life_balance': np.random.normal(6, 1.5, n_samples),
        'skill_diversity': np.random.normal(7, 1.5, n_samples),
        'market_demand': np.random.normal(6, 1.5, n_samples),
        'job_stability': np.random.normal(6, 1.5, n_samples),
        'benefits_score': np.random.normal(5, 1.5, n_samples),
        'commute_time': np.random.normal(30, 15, n_samples),
        'flexibility_score': np.random.normal(7, 1.5, n_samples),
        'training_opportunities': np.random.normal(6, 1.5, n_samples),
        'industry': np.random.choice(['Tech', 'Healthcare', 'Education', 'Retail', 'Transportation'], n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add noise to create variations
    for col in df.select_dtypes(include=[np.number]).columns:
        noise = np.random.normal(0, noise_factor * df[col].std(), n_samples)
        df[col] = df[col] + noise
    
    # Generate target variable with stronger and more realistic relationships
    probability = (
        -0.4 * (df['job_satisfaction'] / df['job_satisfaction'].max()) +  # Even stronger weight on satisfaction
        -0.35 * (df['work_life_balance'] / df['work_life_balance'].max()) +  # Stronger weight on balance
        -0.3 * ((df['current_income'] - df['current_income'].mean()) / df['current_income'].std()) +  # Stronger income effect
        0.25 * (df['market_demand'] / df['market_demand'].max()) +
        0.2 * (df['skill_diversity'] / df['skill_diversity'].max()) +
        -0.25 * (df['job_stability'] / df['job_stability'].max()) +
        0.15 * (df['benefits_score'] / df['benefits_score'].max()) +
        -0.2 * (df['flexibility_score'] / df['flexibility_score'].max()) +
        0.15 * ((df['market_demand'] * df['skill_diversity']) / (df['market_demand'].max() * df['skill_diversity'].max())) +
        -0.15 * ((df['job_satisfaction'] * df['work_life_balance']) / (df['job_satisfaction'].max() * df['work_life_balance'].max()))
    )
    
    # Add stronger non-linear effects
    probability += -0.25 * (df['job_satisfaction'] < df['job_satisfaction'].quantile(0.3)).astype(float)
    probability += 0.25 * (df['market_demand'] > df['market_demand'].quantile(0.7)).astype(float)
    
    # Add education level effect
    education_effect = pd.Series({
        'High School': 0.2,
        'Bachelor': 0.1,
        'Master': -0.1,
        'PhD': -0.2
    })
    probability += df['education_level'].map(education_effect)
    
    # Add industry effect
    industry_effect = pd.Series({
        'Tech': 0.2,
        'Healthcare': 0.1,
        'Education': -0.1,
        'Retail': 0.15,
        'Transportation': 0.05
    })
    probability += df['industry'].map(industry_effect)
    
    # Normalize probability
    probability = 1 / (1 + np.exp(-probability))  # Sigmoid transformation
    df['will_change_job'] = (np.random.random(n_samples) < probability).astype(int)
    
    return df

def augment_data(df, n_augment=1):
    """Augment the dataset by creating variations of existing samples"""
    augmented_dfs = []
    
    for _ in range(n_augment):
        # Create a copy of the dataframe
        aug_df = df.copy()
        
        # Add random noise to numerical columns
        for col in aug_df.select_dtypes(include=[np.number]).columns:
            if col != 'will_change_job':  # Don't modify the target variable
                noise = np.random.normal(0, 0.1, len(aug_df))
                aug_df[col] = aug_df[col] + noise
        
        # Randomly swap some categorical values
        categorical_cols = ['education_level', 'industry']
        for col in categorical_cols:
            mask = np.random.random(len(aug_df)) < 0.1  # 10% chance of change
            aug_df.loc[mask, col] = np.random.choice(df[col].unique(), mask.sum())
        
        augmented_dfs.append(aug_df)
    
    return pd.concat([df] + augmented_dfs, ignore_index=True)

# Enhanced feature engineering function
def engineer_features(df):
    # Create more focused interaction features
    df['satisfaction_balance'] = df['job_satisfaction'] * df['work_life_balance']
    df['market_skill'] = df['market_demand'] * df['skill_diversity']
    df['income_stability'] = df['current_income'] / (df['job_stability'] + 1)
    
    # Create threshold-based features with more meaningful thresholds
    df['low_satisfaction'] = (df['job_satisfaction'] < 5).astype(float)
    df['high_market_demand'] = (df['market_demand'] > 7).astype(float)
    df['poor_work_life'] = (df['work_life_balance'] < 5).astype(float)
    
    # Create composite scores with weighted averages
    df['job_quality'] = (0.4 * df['job_satisfaction'] + 0.3 * df['work_life_balance'] + 0.3 * df['job_stability'])
    df['opportunity_score'] = (0.4 * df['market_demand'] + 0.3 * df['skill_diversity'] + 0.3 * df['training_opportunities'])
    df['compensation_score'] = (0.6 * df['current_income'] + 0.4 * df['benefits_score'])
    
    # Create relative measures with industry-specific normalization
    df['relative_income'] = df.groupby('industry')['current_income'].transform(lambda x: (x - x.mean()) / x.std())
    df['relative_satisfaction'] = df.groupby('industry')['job_satisfaction'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

# Main execution
if __name__ == "__main__":
    # Generate larger initial dataset
    print("Generating initial dataset...")
    df = generate_gig_worker_data(10000)
    
    # Augment the dataset
    print("Augmenting dataset...")
    df = augment_data(df, n_augment=2)
    
    print(f"Final dataset size: {len(df)} samples")
    
    # Feature engineering
    print("Performing feature engineering...")
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop('will_change_job', axis=1)
    y = df['will_change_job']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define numerical and categorical columns
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Create model pipeline with optimized parameters
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=8,
            min_child_weight=3,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            random_state=42
        ))
    ])
    
    # Preprocess the data first
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    # Train the model
    print("Training the model...")
    model.named_steps['classifier'].fit(X_train_balanced, y_train_balanced)
    
    # Save the trained model
    print("Saving the model...")
    joblib.dump(model, 'gig_worker_model.joblib')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print results
    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'feature': numerical_features.tolist() + [f"{feat}_{val}" for feat, vals in 
                   zip(categorical_features, preprocessor.named_transformers_['cat'].categories_) 
                   for val in vals[1:]],
        'importance': model.named_steps['classifier'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nFeature importance plot has been saved as 'feature_importance.png'") 