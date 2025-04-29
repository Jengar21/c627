# c627

# Healthcare Outcome Prediction

## Team Members

* \[Team Member 1 Name]: \[Specific Contributions, e.g., Data Loading, Naive Bayes Model]
* \[Team Member 2 Name]: \[Specific Contributions, e.g., Data Visualization, Decision Tree Model]
* \[Team Member 3 Name]: \[Specific Contributions, e.g., Data Preprocessing, Report Writing]
* \[Team Member 4 Name]: \[Specific Contributions, e.g., Model Evaluation, Code Optimization]

## Project Description

This project aims to predict healthcare outcomes using machine learning. We utilized a publicly available dataset and implemented Naive Bayes and Decision Tree models to classify and predict outcomes. The project follows the steps outlined in the CS627 Term Project Description.

## Dataset

* **Name:** \[Name of the dataset, e.g., Breast Cancer Wisconsin (Diagnostic) Dataset]
* **Source:** \[Link to the dataset source, e.g., [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)]
* **Description:** \[Briefly describe the dataset, its context, and what it contains. E.g., "This dataset contains features of breast masses and a diagnosis of malignant or benign."]

## Code Sections

### 1. Imports

* This section includes all necessary Python libraries for data manipulation, visualization, model building, and evaluation.
    * Libraries used: `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, `sklearn` (and submodules).

### 2. Data Loading and Exploration

* Loads the dataset from a CSV file.
* Performs initial data exploration, including:
    * Displaying the size and shape of the data.
    * Printing column names.
    * Visualizing the distribution of the target variable.

### 3. Data Visualization

* Generates visualizations to understand the dataset.
    * Includes:
        * Bar chart of diagnosis counts.
        * Histogram of a numerical feature (e.g., radius mean).
        * Scatter plot to show the relationship between two features.
        * Box plot to compare feature distributions.

### 4. Data Preparation

* Prepares the data for machine learning models.
    * Steps:
        * Selects features (X) and the target variable (y).
        * Encodes categorical variables (if any).
        * Normalizes numerical features using `StandardScaler`.
        * Splits the data into training and testing sets.

### 5. Model Building and Evaluation

* Implements and evaluates the following machine learning models:
    * Naive Bayes (Gaussian Naive Bayes)
    * Decision Tree Classifier
* For each model, the code:
    * Trains the model.
    * Makes predictions on the test set.
    * Calculates and prints accuracy, precision, recall, and F1-score.
    * Generates and displays a confusion matrix.
    * Provides a detailed classification report with per-class metrics.

### 6. Model Comparison

* Compares the performance of the Naive Bayes and Decision Tree models.
    * Compares accuracy, precision, recall, and F1-score.
    * States which model performed better and provides a brief explanation.

### 7. Conclusion

* Summarizes the project's purpose, key findings, and limitations.
* Suggests potential future work.

## Link to Project Description

* [CS627 Term Project Description]([Insert the actual link to the PDF])

## Notes

* All code is written in Python.
* Dataset file: `data.csv` (should be in the same directory as the script).
* For team-specific details, please refer to the report.