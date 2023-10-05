# Heart Failure Prediction Project

This project aims to predict heart failure events based on clinical and health-related features using various machine learning models. The dataset used in this project contains information about patients and their health attributes.

## Overview

Heart failure is a critical medical condition that requires early detection and intervention. Predictive models can assist healthcare professionals in identifying patients at risk of heart failure, allowing for timely treatment and improved patient outcomes.

In this project, we explore different machine learning algorithms to predict the likelihood of a heart failure event. We preprocess the data, train multiple models, and evaluate their performance. The best-performing models are used for predictions.

## Dataset

The dataset used in this project contains the following features:

- Age
- Anaemia
- Creatinine Phosphokinase
- Diabetes
- Ejection Fraction
- High Blood Pressure
- Platelets
- Serum Creatinine
- Serum Sodium
- Gender
- Smoking
- Follow-up Period (Time)
- Target Variable: Death Event (1 for death, 0 for no death)

## Models Explored

1. **Logistic Regression**: A linear classification model.
2. **K-Nearest Neighbors (KNN)**: A distance-based classification model.
3. **Decision Tree**: A tree-based classification model.
4. **Support Vector Machine (SVM)**: A powerful classification model.
5. **Random Forest**: An ensemble-based classification model.

## Results

After training and evaluating these models, the best-performing models are Support Vector Machine (SVM) and Logistic Regression, both achieving an accuracy of 90%.

## Usage

You can use the provided code and models to predict heart failure events for new patient data. Make sure to preprocess new data in the same way as the training data for accurate predictions.

## Dependencies

This project relies on the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
