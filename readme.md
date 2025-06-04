# Smart Water System Consumption Prediction

## Project Overview
This project implements a machine learning model to predict water consumption based on various household and environmental factors. The model uses features like apartment type, temperature, humidity, income level, and appliance usage to make accurate predictions of water consumption patterns.

## Dataset
The project uses two main datasets:
- `train.csv`: Used for training and validating the model
- `test.csv`: Used for making predictions on new data

### Features
- Timestamp
- Residents
- Apartment_Type
- Temperature
- Humidity
- Water_Price
- Period_Consumption_Index
- Income_Level
- Guests
- Amenities
- Appliance_Usage
- Water_Consumption (target variable)

## Model Development
The notebook (`swm.ipynb`) implements several machine learning approaches:
1. Random Forest Regressor
2. PyCaret AutoML comparison
3. LightGBM Regressor (final selected model)

## Key Steps in the Notebook
1. Data Loading and Initial Exploration
2. Missing Value Treatment
   - Categorical variables filled with 'Missing'
   - Numerical variables filled with median values
3. Feature Engineering
   - Label encoding for categorical variables
   - Data type conversions
4. Model Training and Evaluation
   - Multiple models compared using PyCaret
   - Final LightGBM model selected based on RMSE
5. Model Persistence
   - Best model saved as 'model.pkl'
6. Prediction Generation
   - Predictions made on test dataset
   - Results saved in 'predictions.csv' and 'final.csv'

## Files Generated
- `model.pkl`: Serialized machine learning model
- `predictions.csv`: Raw predictions from the model
- `final.csv`: Final predictions with timestamps
- `logs.log`: Execution logs

## Requirements
The project requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- pycaret
- lightgbm
- joblib

## Model Performance
The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Custom scoring metric (100 - RMSE)

### Results
We tested multiple modeling approaches with the following results:

1. Initial Random Forest Model:
   - Used basic feature engineering
   - Handled missing values with simple imputation

2. PyCaret AutoML Comparison:
   - Tested multiple algorithms
   - Applied advanced preprocessing
   - Selected best performing model based on RMSE

3. Final LightGBM Model (Selected Solution):
   - Achieved best performance
   - RMSE Score: 11.59
   - Custom Score: 88.41
   - Features complete preprocessing pipeline
   - Robust to missing values
   - Generated predictions saved in 'final.csv'

## Usage
1. Ensure all required libraries are installed
2. Run the notebook cells sequentially
3. The model will automatically process the test data and generate predictions
4. Final predictions can be found in 'final.csv'

## Notes
- The model handles missing values automatically
- Categorical variables are encoded using Label Encoding
- The system is designed to work with both complete and incomplete data entries