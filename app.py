# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
import pickle
import numpy as np

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
           # reading the inputs given by the user
           BMI = float(request.form['BMI'])
           log_BMI = np.log(BMI)
           Glucose = float(request.form['Glucose'])
           log_Glucose = np.log(Glucose)
           Age = float(request.form['Age'])
           log_Age = np.log(Age)
           Pregnancies = float(request.form['Pregnancies'])
           sq_Pregnancies = np.sqrt(Pregnancies)
           Insulin = float(request.form['Insulin'])
           Reci_Insulin = 1/Insulin
           SkinThickness = float(request.form['SkinThickness'])
           log_SkinThickness = np.log(SkinThickness)
           DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
           log_DiabetesPedigreeFunction = np.log(DiabetesPedigreeFunction)
           BloodPressure = float(request.form['BloodPressure'])
           log_BloodPressure = np.log(BloodPressure)
           filename = 'SVM_Prediction.sav'
           model = pickle.load(open(filename, 'rb'))
           scalefile = 'standardScalar.sav'
           scalar = pickle.load(open(scalefile, 'rb'))
           scaled_data = scalar.transform([[log_BMI,log_Glucose,log_Age,sq_Pregnancies,Reci_Insulin,log_SkinThickness,log_DiabetesPedigreeFunction,log_BloodPressure]])
           prediction = model.predict(scaled_data)
           print('prediction value is ', prediction)
           # showing the prediction results in a UI
           return render_template('predict.html', prediction = prediction)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

