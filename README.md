# DSC-Recruitment

This project aims to predict whether an individual has a part-time job or not based on various features such as age, gender, hours studied, academic performance, and more. We use machine learning algorithms to build a predictive model and evaluate its performance.

1. README.md: The project's documentation (you're reading it).
2. Student.csv: The dataset containing information about individuals and their part-time job status.
3. DSC-Recruitment.ipynb: Jupyter Notebook containing the Python code for data preprocessing, model selection, and evaluation.
4. model.pkl: Trained Model

## Dataset
The dataset used for this project contains the following columns:

1. Name: The name of the individual (not used for modeling).
2. Age: Age of the individual.
3. Gender: Gender of the individual (encoded as 0 for Female and 1 for Male).
4. Hours_Studied: Number of hours studied by the individual.
5. IQ: IQ score of the individual.
6. Physics_Marks: Marks obtained in physics.
7. Math_Marks: Marks obtained in math.
8. Chemistry_Marks: Marks obtained in chemistry.
9. Has_Part_Time_Job: Whether the individual has a part-time job (encoded as 0 for No and 1 for Yes) (the target variable).
10. Study_Hours_Group: Grouping of study hours (encoded as 0 for Low and 1 for High)

## Data Preprocessing
1. The dataset is loaded, and unnecessary columns like "Name" is dropped.
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
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

## Results
The project compares the performance of the different algorithms to determine the best model for predicting whether an individual has a part-time job or not.

## Conclusion
The project provides insights into which machine learning algorithm performs best for the part-time job prediction task based on the dataset and evaluation metrics. The selected model can be used for future predictions.

## Technologies Used
1. NumPy
2. Pandas
3. Scikit-learn
4. Matplotlib
5. Seaborn
6. Pickle

   
## How to Use
1. Clone this repository to your local machine.
2. Install the required libraries using pip install -r requirements.txt.
3. Open and run the Jupyter Notebook part_time_job_prediction.ipynb to execute the analysis.
4. Feel free to adapt and expand upon this README to provide more context or details specific to your project.
