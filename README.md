# c627

# Healthcare Outcome Prediction

## Team Members
Lauren DeMaio
Rohit Kumaratchi
Jennifer Garcia

## Project Description

This project aims to predict healthcare outcomes using machine learning. We utilized a publicly available dataset and implemented Naive Bayes and Decision Tree models to classify and predict outcomes. The project follows the steps outlined in the CS627 Term Project Description.

## Dataset

* **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) 
* **Description:**
    * Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. [cite: 15, 4]
    * They describe characteristics of the cell nuclei present in the image. [cite: 15, 4]
    * Ten real-valued features are computed for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. [cite: 15, 4]
    * For each image, the mean, standard error, and "worst" (largest mean of the three largest values) of these features were computed, resulting in 30 features. [cite: 15, 4]
    * Outcome: diagnosis (M = malignant, B = benign) [cite: 15, 4]
    * Class distribution: 357 benign, 212 malignant [cite: 15, 4]


## Code Sections

### Imports

* This section includes all necessary Python libraries for data manipulation, visualization, model building, and evaluation.
    * Libraries used: `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, `sklearn` (and submodules).

### Data Loading and Exploration

* Loads the dataset from a CSV file.
* Performs initial data exploration, including:
    * Displaying the size and shape of the data.
    * Printing column names.
    * Visualizing the distribution of the target variable.

### Data Visualization

* Generates visualizations to understand the dataset.
    * Includes:
        * Bar chart of diagnosis counts.
        * Histogram of a numerical feature (e.g., radius mean).
        * Scatter plot to show the relationship between two features.
        * Box plot to compare feature distributions.

### Data Preparation

* Prepares the data for machine learning models.
    * Steps:
        * Selects features (X) and the target variable (y).
        * Encodes categorical variables (if any).
        * Normalizes numerical features using `StandardScaler`.
        * Splits the data into training and testing sets.

### Model Building and Evaluation

* Implements and evaluates the following machine learning models:
    * Naive Bayes (Gaussian Naive Bayes)
    * Decision Tree Classifier
* For each model, the code:
    * Trains the model.
    * Makes predictions on the test set.
    * Calculates and prints accuracy, precision, recall, and F1-score.
    * Generates and displays a confusion matrix.
    * Provides a detailed classification report with per-class metrics.

### Model Comparison

* Compares the performance of the Naive Bayes and Decision Tree models.
    * Compares accuracy, precision, recall, and F1-score.
    * States which model performed better and provides a brief explanation.



## Link to Project Description

* [CS627 Term Project Description](https://docs.google.com/document/d/1FNJsrxCRkVKYdUblwxTYGg2edbPFr4x9lukplNkRwG8/edit?usp=sharing)

## Notes

* All code is written in Python.
* For team-specific details, please refer to the report.