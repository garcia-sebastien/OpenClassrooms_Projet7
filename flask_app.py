# flask_app.py
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the model
model = mlflow.pyfunc.load_model(model_uri="models:/finalModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = pd.DataFrame(data)
    
    # Make predictions
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.52).astype(int)
    
    response = {
        'probabilities': probabilities.tolist(),
        'predictions': predictions.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)