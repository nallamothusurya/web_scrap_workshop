from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
with open('Final_Model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    name = int(request.form['name'])
    price = int(request.form['price'])
    offer = int(request.form['offer'])
    
    # Use the model to predict
    prediction = model.predict(np.array([[name, price, offer]]))
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
    
