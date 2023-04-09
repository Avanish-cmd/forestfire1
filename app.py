import pickle 
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import ridge regressor model and standardscaler pickle 
ridge_model  = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


## Route for homepage
@app.route('/')
def index():
    return render_template('index.html') # when we go home page this will be running



@app.route('/predictdata', methods = ['GET', 'POST'])  # home.html file ko display krna is get request and home.html file ko submit krna is post request
def predict_datapoint():
    if request.method == 'POST': # here order of taking data is same as we are does in model training and prediction 
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))


        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result = result[0])

    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
