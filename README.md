Titanic-AutoML-Project---AIDI1002

This project is part of the AIDI1002 course, focusing on applying AutoML techniques to the Titanic - Machine Learning from Disaster dataset. The goal is to predict the survival of passengers using machine learning models.

Project Structure
/Titanic - Machine Learning from Disaster/
├── Housing.csv                   # Additional dataset (possibly for comparison or analysis)
├── gender_submission.csv          # Example of submission format with predictions
├── housing_predictions.csv        # Predictions generated for Housing.csv (if applicable)
├── submission.csv                 # Final submission file in the required format
├── test.csv                       # Test data for model evaluation
├── titanic_predictions.csv        # Model predictions for the Titanic test data
└── train.csv                      # Training data for model training
Dataset Information

The project utilizes the Titanic - Machine Learning from Disaster dataset from Kaggle. The dataset consists of two main files:

    train.csv: Contains the training data with passenger details and survival labels.
    test.csv: Contains passenger details for which survival needs to be predicted.

Additional files:

    gender_submission.csv: A sample submission file provided by Kaggle.
    housing_predictions.csv and Housing.csv: May contain additional data or predictions for comparison or analysis, unrelated to the primary Titanic dataset.
    titanic_predictions.csv: The final predictions for the Titanic test dataset.

Requirements

Ensure you have the following installed:

    Python (version 3.7 or later)
    Required libraries (pandas, numpy, sklearn, etc.)

Installation
Clone the repository:

git clone https://github.com/yourusername/Titanic-AutoML-Project---AIDI1002.git
cd Titanic-AutoML-Project---AIDI1002

Install the dependencies: Install required libraries with:

    pip install -r requirements.txt

Usage

    Data Preprocessing: Load and preprocess train.csv and test.csv for model training and prediction.
    Model Training: Use AutoML frameworks or custom machine learning pipelines to train on train.csv.
    Make Predictions: Apply the trained model to test.csv and save predictions in titanic_predictions.csv.
    Submission: Format your predictions according to gender_submission.csv and save them in submission.csv for evaluation.

File Descriptions

    train.csv: Training data containing features and labels (survival status).
    test.csv: Test data for which survival predictions are made.
    gender_submission.csv: Sample submission format.
    submission.csv: Final formatted predictions.
    housing_predictions.csv and Housing.csv: Additional datasets or predictions, if used in extended analysis.

Results and Analysis

The model predictions and analysis are saved in titanic_predictions.csv and submission.csv. Performance evaluation metrics and detailed analysis can be found in the project report. 
