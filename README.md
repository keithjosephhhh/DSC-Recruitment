# DSC-Recruitment

This project aims to predict whether an individual has a part-time job or not based on various features such as age, gender, hours studied, academic performance, and more. We use machine learning algorithms to build a predictive model and evaluate its performance.

README.md: The project's documentation (you're reading it).
data.csv: The dataset containing information about individuals and their part-time job status.
DSC-Recruitment.ipynb: Jupyter Notebook containing the Python code for data preprocessing, model selection, and evaluation.
model.pkl: Trained Model

## Dataset
The dataset used for this project contains the following columns:

1. Name: The name of the individual (not used for modeling).
2. Age: Age of the individual.
3. Gender: Gender of the individual (encoded as 0 for Female and 1 for Male).
4. Hours_Studied: Number of hours studied by the individual.
5. IQ: IQ score of the individual (the target variable).
6. Physics_Marks: Marks obtained in physics.
7. Math_Marks: Marks obtained in math.
8. Chemistry_Marks: Marks obtained in chemistry.
9. Has_Part_Time_Job: Whether the individual has a part-time job (encoded as 0 for No and 1 for Yes).
10. Study_Hours_Group: Grouping of study hours (not used for modeling).

## Data Preprocessing
1. The dataset is loaded, and unnecessary columns like "Name" and "Study_Hours_Group" are dropped.
2. Categorical variables like "Gender", "Has_Part_Time_Job" and "Study_Hours_Group" are encoded into numerical values.
3. The data is split into training and testing sets for model evaluation.
4. Model Selection
5. Several classification algorithms are applied to predict "Has_Part_Time_Job."


### The algorithms tested include:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVM)
5. Model Evaluation

### Each model is evaluated using the following metrics:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix

## Results
The project compares the performance of the different algorithms to determine the best model for predicting whether an individual has a part-time job or not.

## Conclusion
The project provides insights into which machine learning algorithm performs best for the part-time job prediction task based on the dataset and evaluation metrics. The selected model can be used for future predictions.

## Technologies Used
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Pickle
How to Use
Clone this repository to your local machine.
Install the required libraries using pip install -r requirements.txt.
Open and run the Jupyter Notebook part_time_job_prediction.ipynb to execute the analysis.
Feel free to adapt and expand upon this README to provide more context or details specific to your project.
