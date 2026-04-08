"""
Flask Server to Connect HTML GUI to Your Trained Model
=======================================================
Put this file as: app.py
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import io
import os

app = Flask(__name__)

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "models/leaf_cnn_model.h5"
IMG_SIZE = (128, 128)

# Load your trained model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Your 7 leaf classes
class_labels = ['bipinnate', 'entire', 'fascicle', 'lobed', 'palmate', 'pinnate', 'trifoliate']

# ==============================
# SERVE HTML PAGE
# ==============================
@app.route('/')
def index():
    return render_template('leaf_recognition_gui.html')

# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Remove the data URL prefix (e.g., "data:image/png;base64,")
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image (same as your training)
        image = image.resize(IMG_SIZE)
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        top_confidence = float(predictions[top_idx] * 100)
        predicted_class = class_labels[top_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {
                'type': class_labels[idx],
                'confidence': float(predictions[idx] * 100)
            }
            for idx in top3_idx
        ]
        
        # Return results
        return jsonify({
            'predicted': predicted_class,
            'confidence': top_confidence,
            'top3': top3
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    # Make sure templates folder exists
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*50)
    print("🍃 LEAFLET - AI Leaf Recognition System")
    print("="*50)
    print("\n📝 SETUP INSTRUCTIONS:")
    print("1. Move 'leaf_recognition_gui.html' to 'templates/' folder")
    print("2. Make sure 'models/leaf_cnn_model.h5' exists")
    print("3. Run: python app.py")
    print("4. Open browser: http://localhost:5000")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)