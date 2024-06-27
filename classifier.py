import os
from PIL import Image
from torchvision import datasets
from main import classify_image
from enum import Enum

class Animal(Enum):
    Labrador = 1
    Albert = 2
    Kedi = 3
    Çitos = 4

selected_animal = Animal.Labrador

image_file_type = 'jpeg' if selected_animal.value == 2 else 'jpg'


image_path = 'C:\\Users\\st\\Desktop\\VeriSeti\\test\\{}.{}'.format(selected_animal.name, image_file_type)
model_path = 'C:\\Users\\st\\Desktop\\animal_classifier_resnet18.pth'
data_dir = 'C:\\Users\\st\\Desktop\\VeriSeti'
train_dir = os.path.join(data_dir, 'train')
class_names = datasets.ImageFolder(root=train_dir).classes

classify_image(image_path, model_path, class_names)