from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
loaded_model = pickle.load(open('KNN_Model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')



@app.route('/prediction', methods=['POST'])
def predict():
    glucose = request.form['glucose']
    bmi = request.form['bmi']
    age = request.form['age']
    
    
    prediction = loaded_model.predict([[glucose, bmi, age]])[0]
    probability = loaded_model.predict_proba([[glucose, bmi, age]])
    probability = f"{np.round((np.max(probability) * 100), 2)}%"
    
    if prediction == 0:
        prediction = 'Not Diabetic'
        
    else:
        prediction = 'Diabetic'
    return render_template('index.html', output_prediction=prediction, output_proba = probability)


if __name__ == '__main__':
    app.run(debug=True)
    