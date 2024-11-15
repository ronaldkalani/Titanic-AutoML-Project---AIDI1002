Titanic-AutoML-Project---AIDI1002

AutoGluon is an open-source AutoML framework designed to simplify the process of training high-accuracy machine learning models on tabular datasets. It automates tasks such as data preprocessing, model selection, hyperparameter tuning, and ensembling, allowing users to achieve robust predictive performance with minimal manual intervention.

Key Features of AutoGluon:

    Ease of Use: Requires only a single line of Python code to initiate model training on raw tabular data.
    Model Ensembling: Employs multi-layer stacking of various models, including decision trees, neural networks, and gradient boosting machines, to enhance predictive accuracy.
    Automatic Data Processing: Automatically identifies data types and applies appropriate preprocessing steps, such as handling missing values and encoding categorical variables.
    Flexible and Extensible: Supports customization and extension, enabling users to incorporate their own models or preprocessing steps into the AutoML pipeline.

Real-World Application Example:

Consider a real estate company aiming to predict house prices based on features like area, number of bedrooms, and location. Using AutoGluon, the company can input their dataset directly, and AutoGluon will handle data preprocessing, model training, and evaluation. By leveraging its ensembling techniques, AutoGluon can provide accurate price predictions, assisting the company in making informed pricing decisions.

For a detailed exploration of AutoGluon's capabilities and performance, refer to the research paper "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data" by Nick Erickson et al.

This project is part of the AIDI1002 course, focusing on applying AutoML techniques to the Titanic - Machine Learning from Disaster dataset. The goal is to predict the survival of passengers using machine learning models.
Project Structure

Titanic - Machine Learning from Disaster
Housing.csv 
gender_submission.csv
housing_predictions.csv
submission.csv 
test.csv 
titanic_predictions.csv 
train.csv 

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
