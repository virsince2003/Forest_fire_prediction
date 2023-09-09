import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

app=Flask(__name__,template_folder='template')

# import ridge regressor model and standard scaler pickle
ridge_model=pickle.load(open("Models/ridge.pkl","rb"))
standard_Scaler=pickle.load(open("Models/scaler.pkl","rb"))

### Route for homepage
@app.route("/")
def index():
    return render_template("Index.html")

@app.route("/",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template("index.html",result=result[0])
    else:
        return render_template("index.html")


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
