#for production
import tflite_runtime.interpreter as tflite

# for test and development
# import tensorflow.lite as tflite

from flask import Flask, request, jsonify, render_template

from io import BytesIO
import urllib
from PIL import Image
import numpy as np

MODEL_NAME='model/cardmodel.tflite'

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

print('reading model done!...')

classes = ['ace of clubs',
 'ace of diamonds',
 'ace of hearts',
 'ace of spades',
 'eight of clubs',
 'eight of diamonds',
 'eight of hearts',
 'eight of spades',
 'five of clubs',
 'five of diamonds',
 'five of hearts',
 'five of spades',
 'four of clubs',
 'four of diamonds',
 'four of hearts',
 'four of spades',
 'jack of clubs',
 'jack of diamonds',
 'jack of hearts',
 'jack of spades',
 'joker',
 'king of clubs',
 'king of diamonds',
 'king of hearts',
 'king of spades',
 'nine of clubs',
 'nine of diamonds',
 'nine of hearts',
 'nine of spades',
 'queen of clubs',
 'queen of diamonds',
 'queen of hearts',
 'queen of spades',
 'seven of clubs',
 'seven of diamonds',
 'seven of hearts',
 'seven of spades',
 'six of clubs',
 'six of diamonds',
 'six of hearts',
 'six of spades',
 'ten of clubs',
 'ten of diamonds',
 'ten of hearts',
 'ten of spades',
 'three of clubs',
 'three of diamonds',
 'three of hearts',
 'three of spades',
 'two of clubs',
 'two of diamonds',
 'two of hearts',
 'two of spades']

def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def open_image(file_path):
    img = Image.open(file_path)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x.astype('float32')
    x /= 255.0  # Rescale to the range [0.0, 1.0]
    return x

def predict_simple(img):
    img = prepare_image(img, (224, 224))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Preprocess the input image
    x = preprocess_input(img_array)

    # Set input tensor
    interpreter.set_tensor(input_index, x)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    preds = interpreter.get_tensor(output_index)

    # Convert predictions to a dictionary
    float_predictions = preds[0].tolist()
    result = dict(zip(classes, float_predictions))
    top_10_predictions = dict(sorted([item for item in result.items() if item[1]>0.1], key=lambda x: x[1], reverse=True)[:10])
    print(top_10_predictions)
    
    return top_10_predictions

def predictions_to_str(predictions):
    s='PREDICTIONS:<br>'
    for cardclass in predictions.keys():
        s+=f'{cardclass}: {round(predictions[cardclass],5)}, <br>'
    return s



app = Flask('cardclass')


@app.route('/predict', methods=['POST'])
def predict():
    """ Flask routine for API prediction.
    will search for field 'url' in the request, and return a 
    json with top 10 classifications
    """
    url = request.get_json().get('url')

    if url is None:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        # Assuming you have a function predict_test_file defined
        img=download_image(url)
        predictions = predict_simple(img)

        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def main_get():
    """
    Flask routine for entering via browser, to get base page with form and script

    :return: main.html
    """
    return render_template('main.html')


@app.route('/predict_img', methods=['POST'])
def predict_img():
    try:
        img_file = request.files['file']
        img = Image.open(BytesIO(img_file.read()))

        predictions = predict_simple(img)

        return predictions_to_str(predictions)
    except Exception as e:
        return f'error{str(e)}'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)







