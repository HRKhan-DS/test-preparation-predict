import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
print("Current Directory:", os.getcwd())
app = Flask(__name__)

# Load the pre-trained model
with open("LinearRegressionModel.pkl", "rb") as model_file:
    lr_pipeline = joblib.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from the form
            gender = int(request.form['gender'])
            race_ethnicity = int(request.form['race_ethnicity'])
            parental_level_of_education = int(request.form['parental_level_of_education'])
            lunch = int(request.form['lunch'])
            test_preparation_course = int(request.form['test_preparation_course'])
            math_score = float(request.form['math_score'])
            reading_score = float(request.form['reading_score'])
            writing_score = float(request.form['writing_score'])

            # Create a DataFrame with the user input
            user_data = pd.DataFrame({
                'gender': [gender],
                'race_ethnicity': [race_ethnicity],
                'parental_level_of_education': [parental_level_of_education],
                'lunch': [lunch],
                'test_preparation_course': [test_preparation_course],
                'math_score': [math_score],
                'reading_score': [reading_score],
                'writing_score': [writing_score]
            })

            # Use the preprocessor and the trained model to make predictions
            user_data_processed = lr_pipeline.named_steps['preprocessor'].transform(user_data)
            prediction = lr_pipeline.named_steps['regressor'].predict(user_data_processed)

            return render_template('index.html', prediction=f'Predicted Score: {prediction[0]:.2f}')
        except Exception as e:
            return render_template('index.html', prediction='Error in input. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)