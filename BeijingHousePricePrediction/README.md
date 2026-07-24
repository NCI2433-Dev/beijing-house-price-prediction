Project Title:
Beijing House Price Prediction using Machine Learning

Project Description:
This project aims to forecast the house prices in Beijing based on Machine Learning models. Three models of regression were trained and compared for the best performing model.

Project Flow:
Import all the necessary python libraries.
2. Load and upload the dataset about houses in Beijing.
3. Process features and class labels.
4. Partition your data set into training and testing sets.
6. Compute the parameters of the linear function for the data.7. Create a linear function to describe the data.
7. Fit the Linear Regression model.
8. Train the Random Forest model.
9. Train the XGBoost model.
9. Evaluate all the models by comparing their performance based on the criteria of MAE, RMSE and R².
Error Analysis: Carry out error analysis on the best model.
11. Save trained XGBoost model.
13. Use a Streamlit web app to deploy the model.

Programming Language:
Python 3

Libraries Used:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib
- streamlit

Dataset:
This project uses the publicly available Beijing Housing dataset.
    -https://www.kaggle.com/datasets/ruiqurm/lianjia

The cleaned dataset (beijing_housing_cleaned.csv) is derived from the public dataset and is not included in this submission, as per the assignment instructions.

Files Included:
- app.py
- feature_columns.pkl
- House_Price_Prediction.ipynb
- README.txt
- requirements.txt
- xgboost_house_price.pkl






How to Run:
To install required libraries use:
   You can install requirements by running the following command:

2. Run the notebook to train the models.

4. Then, run the Streamlit application using:
   streamlit run app.py