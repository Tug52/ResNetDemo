import ssl
import certifi
import urllib.request
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
from flask import Flask, request, jsonify
import os

# Ensure SSL context uses certifi's certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)

app = Flask(__name__)

# Define transformations for the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(datasets.ImageFolder(root='C:\\Users\\st\\Desktop\\VeriSeti\\train').classes))
model.load_state_dict(torch.load('C:\\Users\\st\\Desktop\\animal_classifier_resnet18.pth'))
model = model.to(device)
model.eval()

# Class names
class_names = datasets.ImageFolder(root='C:\\Users\\st\\Desktop\\VeriSeti\\train').classes


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data['image_path']

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image path does not exist'}), 400

    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        prediction = class_names[preds[0]]

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)