from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'Model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

data_path = os.path.join(os.path.dirname(__file__), 'data/madeby_me.csv')

data = pd.read_csv(data_path)

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found.")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file) 

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input (symptoms) from the form
        symptoms = request.form['symptoms'].lower()

        # Transform user input using the vectorizer
        user_input_vectorized = tfidf_vectorizer.transform([symptoms])

        try:
            # Make a prediction using the model
            prediction = model.predict(user_input_vectorized)

            # Filter dataset for matching rows
            matched_data = data[data['generic name'].str.contains(prediction[0], case=False, na=False)]

            if not matched_data.empty:
                result = matched_data.to_dict(orient='records')
            else:
                result = "No matching data found."

        except Exception as e:
            result = f"Error in model prediction: {str(e)}"

        return render_template('results.html', data=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
