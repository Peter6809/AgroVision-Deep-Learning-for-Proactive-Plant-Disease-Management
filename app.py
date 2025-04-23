import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from difflib import get_close_matches
import pandas as pd
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import google.generativeai as genai

import torchvision.transforms.functional as TF
import CNN

app = Flask(__name__)

# Load disease and supplement data
disease_info = pd.read_csv('info/disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('info/supplement_info.csv', encoding='cp1252')

# Function to find the closest matching disease name
def find_closest_match(predicted_disease):
    predicted_disease_cleaned = predicted_disease.replace("___", " : ").replace("_", " ")
    disease_names = disease_info["disease_name"].tolist()
    closest_match = get_close_matches(predicted_disease_cleaned, disease_names, n=1, cutoff=0.7)
    return closest_match[0] if closest_match else "No close match found"

# Ensure column names match exactly (modify as needed)
expected_columns = ["Disease Name", "Description", "Possible Steps", "Image URL"]
disease_info.columns = [col.strip().lower().replace(" ", "_") for col in disease_info.columns]

# Configure your API key
api_key = "AIzaSyB4fgvYc8u8S9-yQkQmWp_CiACFbQkfdpE"
genai.configure(api_key=api_key)

##########################################ROUTES########################

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def indexPage():
    return render_template('index.html')

@app.route('/rnn')
def rnn_page():
    return render_template('rnn.html')

@app.route('/vision-transformer')
def vision_page():
    return render_template('vit.html')

@app.route('/capsule-network')
def capnet_page():
    return render_template('capnet.html')


@app.route('/chatbot')
def contact():
    return render_template('chat.html')

@app.route('/plant')
def ai_engine_page():
    return render_template('plant.html')





#####################################CNN##################################

# Define PyTorch CNN model
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(32 * 112 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Adjust the output for your number of classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense_layers(x)
        return x

# Load PyTorch model
cnn_model = PlantDiseaseModel()
cnn_model.load_state_dict(torch.load("outputs/cnn_model.pt"), strict=False)
cnn_model.eval()

# Image transformation for PyTorch model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# CNN
def predict_cnn(image_path):
    """ Predicts plant disease using the PyTorch CNN model """
    try:
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = cnn_model(img)
            predicted_class = torch.argmax(output)

        return predicted_class.item()
    except Exception as e:
        return {'error': str(e)}

# CNN Based
@app.route('/submit', methods=['POST'])
def submit():
    """ Handles CNN-based plant disease detection """
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(file_path)

    # Get prediction from CNN model
    pred = predict_cnn(file_path)

    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['possible_steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    return render_template('submit.html', title=title, desc=description, prevent=prevent,
                           image_url=image_url, pred=pred, sname=supplement_name,
                           simage=supplement_image_url, buy_link=supplement_buy_link)

#################################RNN###################################

# Load RNN model (Keras)
rnn_model = load_model("outputs/RNN_model.keras")

# Load class names
rnn_class_names = np.load("outputs/class_names.npy")

# RNN
def preprocess_image(img_path):
    """ Preprocess image for Keras model """
    img = keras_image.load_img(img_path, target_size=(128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# RNN Based Detection
@app.route('/rnn_submit', methods=['POST'])
def submit_rnn():
    """ Handles RNN-based plant disease detection """
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(file_path)

    # Preprocess image for RNN model
    img_array = preprocess_image(file_path)

    # Predict using RNN model
    prediction = rnn_model.predict(img_array)
    predict_class = rnn_class_names[np.argmax(prediction)]  # Get class name

    matched_disease = find_closest_match(predict_class)

    # Check if the matched disease exists in the dataframe
    disease_row = disease_info[disease_info["disease_name"] == matched_disease]

    if disease_row.empty:  # If no match is found, return an error message
        return jsonify({'error': f'No information found for {matched_disease}'}), 404

    disease_row = disease_row.iloc[0]  # Now safe to access first row

    title = predict_class
    description = disease_row.get("description", "No description available")
    prevent = disease_row.get("possible_steps", "No prevention steps available")
    image_url = disease_row.get("image_url", "#")

    # Check if supplement info exists for this disease
    supplement_row = supplement_info[supplement_info["supplement name"] == matched_disease]

    if not supplement_row.empty:
        supplement_row = supplement_row.iloc[0]
        supplement_name = supplement_row.get("supplement name", "No supplement available")
        supplement_image_url = supplement_row.get("supplement image", "#")
        supplement_buy_link = supplement_row.get("buy link", "#")
    else:
        supplement_name = "No supplement available"
        supplement_image_url = "#"
        supplement_buy_link = "#"

    return render_template('submit.html', title=title, desc=description, prevent=prevent,
                           image_url=image_url, pred=predict_class, sname=supplement_name,
                           simage=supplement_image_url, buy_link=supplement_buy_link)

##############################VISION TRANSFORMER#######################################

# Define the SimpleViT model (Ensure it matches your trained model)
class SimpleViT(nn.Module):
    def __init__(self, num_classes=38):
        super(SimpleViT, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 224 * 224, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the trained model
vit_model_path = "outputs/disease_vit_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chmod(vit_model_path, 0o777)

vit_model = SimpleViT().to(device)
vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
vit_model.eval()

# Image transformations
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (Ensure they match the trained model labels)
vit_class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

def predict_disease_vit(image_path):
    img = Image.open(image_path)
    img_tensor = vit_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = vit_model(img_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = vit_class_names[predicted.item()]
    return predicted_class

@app.route('/vit_submit', methods=['POST'])
def submit_vit():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(file_path)

    predict_class = predict_disease_vit(file_path)

    matched_disease = find_closest_match(predict_class)

    # Check if the matched disease exists in the dataframe
    disease_row = disease_info[disease_info["disease_name"] == matched_disease]

    if disease_row.empty:  # If no match is found, return an error message
        return jsonify({'error': f'No information found for {matched_disease}'}), 404

    disease_row = disease_row.iloc[0]  # Now safe to access first row

    title = predict_class
    description = disease_row.get("description", "No description available")
    prevent = disease_row.get("possible_steps", "No prevention steps available")
    image_url = disease_row.get("image_url", "#")

    # Check if supplement info exists for this disease
    supplement_row = supplement_info[supplement_info["supplement name"] == matched_disease]

    if not supplement_row.empty:
        supplement_row = supplement_row.iloc[0]
        supplement_name = supplement_row.get("supplement name", "No supplement available")
        supplement_image_url = supplement_row.get("supplement image", "#")
        supplement_buy_link = supplement_row.get("buy link", "#")
    else:
        supplement_name = "No supplement available"
        supplement_image_url = "#"
        supplement_buy_link = "#"

    return render_template('submit.html', title=title, desc=description, prevent=prevent,
                           image_url=image_url, pred=predict_class, sname=supplement_name,
                           simage=supplement_image_url, buy_link=supplement_buy_link)

####################################CAPSULENETWORK#####################

# Register the squash function with Keras for serialization (using tf.keras.utils.register_keras_serializable)
def squash(s, axis=-1):
    """Squash function for capsules"""
    squared_norm = tf.reduce_sum(tf.square(s), axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + 1e-8)
    return scale * s

# Load the trained Capsule Network model and class labels with custom_objects
capsule_model = tf.keras.models.load_model(
    "outputs/capsule_plant_disease.h5",
    custom_objects={'squash': squash}  # Specify the custom squash function here
)

# Load the class labels from the saved .npy file
capsule_class_labels = np.load("outputs/cap_lables.npy", allow_pickle=True)

# Function to preprocess the input image and predict the class
def predict_cap_disease(img_path, target_size=(128, 128)):
    """
    Predict the plant disease from an image.

    Parameters:
    - img_path (str): Path to the input image.
    - target_size (tuple): Target size to resize the image (default is (128, 128)).

    Returns:
    - predicted_class_label (str): Predicted class label of the disease.
    - confidence (float): Confidence score of the prediction.
    """
    # Load and preprocess the image
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Predict the class using the trained model
    prediction = capsule_model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])  # Index of the predicted class
    predicted_class_label = capsule_class_labels[predicted_class_index]  # Get the class label
    confidence = prediction[0][predicted_class_index]  # Confidence score for the predicted class

    return predicted_class_label

@app.route('/capnet_submit', methods=['POST'])
def capnet_submit():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(file_path)

    predict_class = predict_cap_disease(file_path)

    matched_disease = find_closest_match(predict_class)

    # Check if the matched disease exists in the dataframe
    disease_row = disease_info[disease_info["disease_name"] == matched_disease]

    if disease_row.empty:  # If no match is found, return an error message
        return jsonify({'error': f'No information found for {matched_disease}'}), 404

    disease_row = disease_row.iloc[0]  # Now safe to access first row

    title = predict_class
    description = disease_row.get("description", "No description available")
    prevent = disease_row.get("possible_steps", "No prevention steps available")
    image_url = disease_row.get("image_url", "#")

    # Check if supplement info exists for this disease
    supplement_row = supplement_info[supplement_info["supplement name"] == matched_disease]

    if not supplement_row.empty:
        supplement_row = supplement_row.iloc[0]
        supplement_name = supplement_row.get("supplement name", "No supplement available")
        supplement_image_url = supplement_row.get("supplement image", "#")
        supplement_buy_link = supplement_row.get("buy link", "#")
    else:
        supplement_name = "No supplement available"
        supplement_image_url = "#"
        supplement_buy_link = "#"

    return render_template('submit.html', title=title, desc=description, prevent=prevent,
                           image_url=image_url, pred=predict_class, sname=supplement_name,
                           simage=supplement_image_url, buy_link=supplement_buy_link)





################################################CHATBOT#############################
# API endpoint for generating responses (for Chat Bot functionality)
@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(user_input)

    if hasattr(response, 'text'):
        return jsonify({"response": response.text})
    else:
        return jsonify({"response": "No text found in the response."}), 500



########################################VGG16############################################
model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/submit-vgg', methods=['POST'])
def submitVGG16():
    if 'image' in request.files:
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        # Prediction function to detect disease
        pred = prediction(file_path)
        
        # Retrieve information based on prediction
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['possible_steps'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        # Print data for debugging
        response_data = {
            'title': title,
            'desc': description,
            'prevent': prevent,
            'buy_link': supplement_buy_link,
        }
        print("Response Data:", response_data)  # Debugging line
        return jsonify(response_data)
    else:
        return jsonify({'error': 'No image uploaded'}), 400





if __name__ == '__main__':
    app.run(debug=True)