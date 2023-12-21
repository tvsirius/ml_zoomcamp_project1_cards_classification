# for production
# import tflite_runtime.interpreter as tflite

# for test
import tensorflow.lite as tflite


from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

MODEL_NAME='model/cardmodel.tflite'

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Define classes
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
    with request.urlopen(url) as resp:
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


def predict_url(url):
    img = download_image(url)
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
    top_10_predictions = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:10])
    print(top_10_predictions)
    
    return top_10_predictions

def predict_test_file(file_path):
    img = open_image(file_path)
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

    top_10_predictions = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:10])
    print(top_10_predictions)
    
    return top_10_predictions


url1 = 'https://cdn.britannica.com/56/129056-050-318DAD51/joker-jokes-playing-card.jpg'
url2='https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Playing_card_club_A.svg/614px-Playing_card_club_A.svg.png'

print('predictions from local files:')

result = predict_test_file('dataset/test/four of hearts/1.jpg')
result = predict_test_file('dataset/test/four of spades/1.jpg')
result = predict_test_file('dataset/test/four of clubs/1.jpg')
result = predict_test_file('dataset/test/four of diamonds/1.jpg')
result = predict_test_file('dataset/test/king of hearts/1.jpg')
result = predict_test_file('dataset/test/jack of spades/3.jpg')

print('predictions from urls:')
result = predict_url(url1)
result = predict_url(url2)