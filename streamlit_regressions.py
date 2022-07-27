import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


@st.cache
def loadData():
    dataset = pd.read_csv('Salary_Data.csv')
    return dataset

 
def preprocessing(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    return X_train, X_test, y_train, y_test

#@st.cache(suppress_st_warning=True)
def linear_regression(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Visualising the Training set results
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Salary vs Experience (Training set)'); plt.xlabel('Years of Experience')
    plt.ylabel('Salary'); plt.show()
    # Visualising the Test set results
    ax.scatter(X_test, y_test, color = 'red')
    ax.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Salary vs Experience (Test set, Linear regression)'); plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    st.pyplot(fig)
    
#@st.cache
def poly_regression(X_train, X_test, y_train, y_test):
    # Fitting Polynomial Regression to the dataset
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_train)
    # Visualising the Polynomial Regression results
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color = 'red')
    ax.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
    plt.title('Salary vs Experience (Training set, Polynomial regression)'); plt.xlabel('Years of Experience')
    plt.ylabel('Salary');st.pyplot(fig)
   
def main():
    st.title('Regression with different methods')
    st.subheader('Given some data we will try to predict a persion\'s salary, based on their years of experience') 
    data = loadData()
    choose_model = st.sidebar.selectbox("Choose the regression model",
		["Linear","Polynomial"])
    X_train, X_test, y_train, y_test = preprocessing(data)
    if(choose_model == "Linear"):
        linear_regression(X_train, X_test, y_train, y_test)
    if(choose_model == "Polynomial"):
        poly_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	main()