from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np

df_temp = pd.read_csv("datasetFinal.csv")
df = pd.get_dummies(df_temp,columns=["Location"])
y = df["Cost"]
X = df.drop(columns = ["ApartmentName","URL","Cost","Distance"])
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X = preprocessing.normalize(X, norm='l2')

def return_location_list(direction):
    direction_list = []
    if direction.lower() == "south":
        return [0,0,0,1,0]
    elif direction.lower() == "north":
        return [0,0,1,0,0]
    elif direction.lower() == "east":
        return [0,1,0,0,0]
    elif direction.lower() == "west":
        return [0,0,0,0,1]
    elif direction.lower() == "central":
        return [1,0,0,0,0]

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def home():
    if request.method == "POST":
        beds = int(request.form["beds"])
        baths = float(request.form["baths"])
        square_feet = int(request.form["square feet"])
        library = int(request.form["library"])
        fitness_center = int(request.form["fitness center"])
        direction = request.form["direction"]

        values_list = [beds, baths, square_feet, library, fitness_center] + return_location_list(direction)     
        rent = rent_calculator(values_list)
        return render_template('index.html', rent_value=rent)
    else:
        return render_template('index.html')


    

    # compute model accuracy
def rent_calculator(inputs):
    ntimes = 30
    avg_rent_value = 0
    for i in range(ntimes):
        # train model with 80% of the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # prediction
        model = linear_model.LinearRegression()
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        #X_predict = [[2,2.0,1200,0,1,0,0,0,1,0]]
        X_predict = [inputs]
        X_norm = preprocessing.normalize(X_predict, norm='l2')
        avg_rent_value += model.predict(X_norm)[0]
    avg_rent_value /= ntimes

    return round(avg_rent_value, 2)
