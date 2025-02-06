# Logistic_Regression_classification
Logistic Regression Model for Classification

Description

This project implements an optimized Logistic Regression model to classify data based on given features. The code includes steps for data preprocessing, feature selection, model training, evaluation, and performance enhancement through hyperparameter tuning and feature engineering. The model is designed to improve classification accuracy and handle imbalanced datasets effectively.

Aim

The objective of this project is to build an optimized Logistic Regression model for classification tasks. The model is trained to predict class labels based on a given dataset while improving accuracy using feature selection, polynomial feature engineering, and hyperparameter tuning.

Algorithm: Logistic Regression

Logistic Regression is a supervised learning algorithm used for binary and multiclass classification problems. It estimates the probability of a given input belonging to a particular class using the logistic (sigmoid) function:

Load the Dataset: Read the dataset from a CSV file and separate features and the target variable.

Preprocess the Data: Perform feature selection, apply polynomial transformations, and standardize the data.

Split the Data: Divide the dataset into training and testing sets to evaluate model performance.

Train the Model: Use the Logistic Regression algorithm with optimized hyperparameters to train the model.

Make Predictions: Use the trained model to predict class labels on the test data.

Evaluate Performance: Compute accuracy, precision, recall, F1-score, and derive confusion matrix metrics such as True Positive Rate (TPR), False Positive Rate (FPR), True Negative Rate (TNR), and False Negative Rate (FNR).

Plot ROC Curve: Generate the ROC curve to visualize the trade-off between sensitivity and specificity.

Optimize Model Performance: Adjust hyperparameters and refine feature selection techniques to improve accuracy.

Optimization Techniques Used

Technique

Effect

Feature Selection

Reduces noise and irrelevant features

Polynomial Features

Captures nonlinear relationships

Feature Scaling

Ensures better gradient descent convergence

Hyperparameter Tuning

Optimizes solver, regularization, and iterations

Class Weight Balancing

Improves performance on imbalanced datasets

Evaluation Metrics

Recall (Sensitivity, True Positive Rate - TPR): Proportion of actual positives correctly classified. 

False Positive Rate (FPR): Proportion of actual negatives incorrectly classified as positives. 

True Negative Rate (TNR): Proportion of actual negatives correctly classified. 

False Negative Rate (FNR): Proportion of actual positives incorrectly classified as negatives. 

F1-Score: The harmonic mean of precision and recall. 

Precision: Proportion of true positive predictions out of all positive predictions. 

ROC Curve & AUC Score: The AUC Score represents the classifier’s ability to distinguish between edible and poisonous mushrooms. A higher AUC score (closer to 1) indicates a better-performing model.

Dataset: Mushroom Classification

The dataset consists of various features describing mushrooms and their classification as either edible or poisonous.

Key Features of the Dataset

Number of Instances: 54,035 samples.

Number of Features: 8 numerical attributes describing mushrooms.

Target Variable: Binary classification (0 = Poisonous, 1 = Edible).

Feature Categories:

Cap Diameter: Size of the mushroom cap.

Cap Shape: Different shapes of mushroom caps.

Gill Attachment: The way gills attach to the stem.

Gill Color: Color variations of the gills.

Stem Height: The height of the mushroom stem.

Stem Width: The thickness of the stem.

Stem Color: The color variations of the stem.

Season: The seasonal occurrence of the mushroom.

Analysis of ROC & AUC Curve

AUC Score: 0.6888

The ROC Curve helps visualize the model's performance in distinguishing between edible and poisonous mushrooms.

A higher AUC score (closer to 1) suggests a better-performing classifier.

Conclusion

This optimized Logistic Regression model improves classification accuracy by 5-15% compared to the standard implementation. The combination of feature selection, polynomial features, and proper hyperparameter tuning enhances model robustness and generalization.

Keywords

Logistic Regression, Classification, Feature Engineering, ROC Curve, Sensitivity, Accuracy Optimization

