from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template

# Define the file path to the saved pickle file
model = pickle.load(open('svm_model.pkl', 'rb'))

# Create a Flask instance
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/results', methods=['POST'])
def patient_input():
    if request.method == 'POST':
        age = int(request.form['Age'])
        restingbp = int(request.form['RestingBP'])
        cholesterol = int(request.form['Cholesterol'])
        fastingbs = int(request.form['FastingBS']) 
        maxHR = int(request.form['MaxHR'])
        oldpeak = float(request.form['Oldpeak'])
        sex = int(request.form['Sex'])
        chestpaintype = int(request.form['ChestPainType'])
        restingecg = int(request.form['RestingECG'])
        ExerciseAngina = int(request.form['ExerciseAngina']) 
        stslope = int(request.form['ST_Slope'])
        
        # Concatenate the one-hot encoded vectors with the numerical variables
        input_data=[[age, restingbp, cholesterol, fastingbs, maxHR, oldpeak, sex, chestpaintype, restingecg, ExerciseAngina, stslope]]      

        # Make predictions using the machine learning model
        prediction = model.predict(input_data)
    
        if prediction==[0]:
            status="Negative"
        else:
            status="Positive"

        return render_template('results.html', response = format(status))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
