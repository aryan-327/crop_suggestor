from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load CSV data
data = pd.read_csv("Crop_recommendation.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Combine input into a DataFrame
        user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Calculate Euclidean distance from all rows in dataset
        data['distance'] = ((data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] - user_input.iloc[0])**2).sum(axis=1)

        # Get top 1 closest match
        top_crop = data.sort_values('distance').iloc[0]['label']

        return render_template('result.html', crop=top_crop)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
