from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('placement_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and match field names exactly
        data = [
            float(request.form['CGPA']),
            int(request.form['Internships']),
            int(request.form['Projects']),
            int(request.form['Workshops']),
            int(request.form['AptitudeTestScore']),
            float(request.form['SoftSkillsRating']),
            int(request.form['ExtracurricularActivities']),
            int(request.form['PlacementTraining']),
            float(request.form['SSC_Marks']),
            float(request.form['HSC_Marks'])
        ]
        
        # Scale input data
        scaled_data = scaler.transform([data])
        
        # Make prediction
        prediction = model.predict(scaled_data)
        result = "Placed 🎉🥳🤩" if prediction[0] == 1 else "Not Placed 🙁❌"

        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return render_template('result.html', prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
