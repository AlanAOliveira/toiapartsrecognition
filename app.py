import pandas as pd
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import logging
import os
from model_loader import CarPartsClassifier

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
APP_NAME = "toiapartsrecognition"

# Load model
MODEL_URI = os.getenv('MODEL_URI', 'models:/toiapartsrecognition_model/Production')
USE_MLFLOW = os.getenv('USE_MLFLOW', 'false').lower() == 'true'

try:
    if USE_MLFLOW:
        classifier = CarPartsClassifier(model_path=MODEL_URI, use_mlflow=True)
    else:
        classifier = CarPartsClassifier(model_path=os.getenv('MODEL_PATH'))
    logger.info(f"[{APP_NAME}] Model loaded successfully")
except Exception as e:
    logger.error(f"[{APP_NAME}] Failed to load model: {e}")
    logger.warning("Falling back to base ResNet50 (untrained head)")
    classifier = CarPartsClassifier()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    try:
        return render_template('index.html', app_name=APP_NAME)
    except Exception as e:
        log.error(f"Error rendering index.html: {e}")
        return "Index file not found", 404


@app.route('/health')
def health():
    return jsonify({
        'app': APP_NAME,
        'status': 'healthy',
        'model_loaded': classifier is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use png, jpg, jpeg, or webp.'}), 400

        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Make prediction
        predictions = classifier.predict(image, top_k=3)

        # Encode image preview
        buffered = io.BytesIO()
        preview = image.copy()
        preview.thumbnail((400, 400))
        preview.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Optional: log predictions as a DataFrame
        df = pd.DataFrame(predictions)
        logger.info(f"[{APP_NAME}] Predictions:\n{df}")

        return jsonify({
            'app': APP_NAME,
            'predictions': predictions,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access (base64 input)."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing base64 image data'}), 400

        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        predictions = classifier.predict(image, top_k=data.get('top_k', 3))

        return jsonify({'app': APP_NAME, 'predictions': predictions})

    except Exception as e:
        logger.exception("API prediction error")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', os.getenv('DATABRICKS_APP_PORT', 8000)))

    print(f"Flask app '{APP_NAME}' starting on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)