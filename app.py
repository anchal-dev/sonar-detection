from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
# try:
#     with open('sonar_model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None
# Load the trained model safely with an absolute path
try:
    model_path = os.path.join(os.path.dirname(__file__), 'sonar_model.pkl')
    print(f"🔍 Looking for model at: {model_path}")  # <--- ADD THIS LINE
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Model accuracy metrics (from your training)
MODEL_STATS = {
    'training_accuracy': 83.42,
    'test_accuracy': 76.19,
    'total_samples': 208,
    'training_samples': 187,
    'test_samples': 21,
    'features': 60,
    'rock_samples': 97,
    'mine_samples': 111
}

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict whether the sonar readings indicate a Rock or Mine
    Expects JSON with 'features' array of 60 values
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'No features provided',
                'message': 'Please provide 60 sonar readings'
            }), 400
        
        features = data['features']
        
        # Validate input
        if len(features) != 60:
            return jsonify({
                'error': 'Invalid input length',
                'message': f'Expected 60 features, got {len(features)}'
            }), 400
        
        # Convert to numpy array and reshape
        input_array = np.asarray(features, dtype=float)
        input_reshaped = input_array.reshape(1, -1)
        
        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure sonar_model.pkl is in the correct directory'
            }), 500
        
        prediction = model.predict(input_reshaped)
        prediction_proba = model.predict_proba(input_reshaped)
        
        # Get probability scores
        rock_probability = float(prediction_proba[0][1]) * 100  # Assuming R is class 1
        mine_probability = float(prediction_proba[0][0]) * 100  # Assuming M is class 0
        
        result = {
            'prediction': str(prediction[0]),
            'prediction_label': 'Rock' if prediction[0] == 'R' else 'Mine',
            'confidence': {
                'rock': round(rock_probability, 2),
                'mine': round(mine_probability, 2)
            },
            'message': f"The object is a {'Rock' if prediction[0] == 'R' else 'Mine'}"
        }
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input values',
            'message': 'All features must be numeric values'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify(MODEL_STATS), 200

@app.route('/api/sample/<sample_type>', methods=['GET'])
def get_sample(sample_type):
    """Get sample data for testing"""
    
    # Sample Rock data
    rock_sample = [0.0409, 0.0421, 0.0573, 0.013, 0.0183, 0.1019, 0.1054, 0.107, 
                   0.2302, 0.2259, 0.2373, 0.3323, 0.3827, 0.484, 0.6812, 0.7555, 
                   0.9522, 0.9826, 0.8871, 0.8268, 0.7561, 0.8217, 0.6967, 0.6444, 
                   0.6948, 0.8014, 0.6053, 0.6084, 0.8877, 0.8557, 0.5563, 0.2897, 
                   0.3638, 0.4786, 0.2908, 0.0899, 0.2043, 0.1707, 0.0407, 0.1286, 
                   0.1581, 0.2191, 0.1701, 0.0971, 0.2217, 0.2732, 0.1874, 0.1062, 
                   0.0665, 0.0405, 0.0113, 0.0028, 0.0036, 0.0105, 0.012, 0.0087, 
                   0.0061, 0.0061, 0.003, 0.0078]
    
    # Sample Mine data
    mine_sample = [0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 
                   0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 
                   0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550, 
                   0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604, 
                   0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744, 
                   0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 
                   0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 
                   0.0180, 0.0084, 0.0090, 0.0032]
    
    if sample_type.lower() == 'rock':
        return jsonify({
            'type': 'rock',
            'features': rock_sample,
            'label': 'R'
        }), 200
    elif sample_type.lower() == 'mine':
        return jsonify({
            'type': 'mine',
            'features': mine_sample,
            'label': 'M'
        }), 200
    else:
        return jsonify({
            'error': 'Invalid sample type',
            'message': 'Use "rock" or "mine"'
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    }), 200
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
