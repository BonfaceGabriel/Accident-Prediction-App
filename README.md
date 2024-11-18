# Accident prediction 

## 1. Introduction
This project aims to predict the severity of accidents based on various input features. We use the LightBBM classifier, an ensemble learning technique, to achieve accurate accident severity predictions. 

## 2. Project Overview
The main objective of the project is to predict the severity of accidents using the LightGBM classifier. The workflow includes the following steps:

1. Data Cleaning: Removing or correcting any missing or erroneous data points from the dataset.
2. Data Scaling: Scaling the numeric features using Min-Max Scaler to bring them within a specific range (e.g., [0, 1]).
3. Data Imputation: Handling missing values in the dataset using Simple Imputer to ensure all features have valid values.
4. Categorical Columns One-Hot Encoding: Converting categorical features into numerical form for the Random Forest classifier.
5. Random Forest Classifier: Training the LightGBM model on the preprocessed data and using it to predict accident severity.

## 3. Installation
To run the project, you need the following dependencies:

- Python (>= 3.6)
- scikit-learn
- pandas
- numpy

You can install the required packages using pip:

```bash
pip install scikit-learn pandas numpy
```

## 4. Usage
To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Prepare your dataset in a CSV format and place it in the project directory.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the script that executes the accident severity prediction using the Random Forest model.

## 5. Data Cleaning
Data cleaning is essential to ensure the dataset is free of missing or erroneous values. This step involves handling missing data points and possibly removing any irrelevant features from the dataset.

## 6. Data Scaling
Scaling the data helps bring all numeric features within the same range, avoiding the dominance of any particular feature during model training. We use Min-Max Scaler to scale the features to a specified range.

## 7. Data Imputation
In cases where there are missing values in the dataset, we use Simple Imputer to fill in those missing values with appropriate estimates, making the dataset ready for model training.

## 8. One-Hot Encoding
Since the Random Forest algorithm requires numerical inputs, we use One-Hot Encoding to convert categorical features into binary vectors, ensuring the algorithm can effectively use them during classification.

## 9. LightGBM
The heart of the project lies in the LightGBM classifier. We train the model on the preprocessed data and tune its hyperparameters if necessary to achieve the best accuracy.

## 10. Demo App
[Accident Severity](https://severity-accident-prediction.streamlit.app/)
