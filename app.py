import pandas as pd
from flask import Flask, render_template, request
import logging
import os
import base64
import datetime


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        log.error(f"Error rendering index.html: {e}")
        return "Index file not found", 404

@app.route('/hello')
def hello_world():
    chart_data = pd.DataFrame({'Apps': [x for x in range(30)],
                               'Fun with data': [2 ** x for x in range(30)]})
    return f'<h1>Hello, World!</h1> <p>Here is some fun data:</p> <pre>{chart_data.to_string(index=False)}</pre>'

@app.route('/receiveimage', methods=['POST'])
def receive_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return {'error': 'No image provided'}, 400
    image_data = data['image'].split(',', 1)[1]
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return {'error': 'Invalid image data'}, 400
    filename = f"images/received_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    with open(filename, 'wb') as f:
        f.write(image_bytes)
    return {'message': 'Image saved', 'filename': filename}, 200



@app.route('/receivenumbers', methods=['POST'])
def receive_numbers():
    data = request.get_json()
    if not data or 'numro1' not in data or 'numro2' not in data:
        return {'error': 'Missing numbers'}, 400
    numro1 = data['numro1']
    numro2 = data['numro2']
    # Process the numbers (example: sum them)
    result = int(numro1) + int(numro2)
    return {'message': 'Numbers received', 'result': result}, 200

if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))

    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")

