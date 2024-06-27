import ssl
import certifi
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow
import mlflow.pytorch

# Ensure SSL context uses certifi's certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)


def print_dir_structure(dir_path):
    """Prints the directory structure for the given path."""
    for root, dirs, files in os.walk(dir_path):
        level = root.replace(dir_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    """Trains and evaluates the model."""
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to MLflow
            mlflow.log_metrics({f"{phase}_loss": np.float64(epoch_loss), f"{phase}_acc": np.float64(epoch_acc)}, step=epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, class_names, device, num_images=6):
    """Visualizes the model's predictions."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
                plt.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def classify_image(image_path, model_path, class_names):
    """Classifies a single image using the trained model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        prediction = class_names[preds[0]]
        print(f'Predicted class: {prediction}')


def main():
    """Main function to train and evaluate the model."""
    # Start an MLflow run
    mlflow.start_run()

    # Enable autologging
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_every_n_step=None, log_models=True,
                           log_datasets=True, disable=False, exclusive=False, disable_for_unsupported_versions=False,
                           silent=False, registered_model_name=None, extra_tags=None, checkpoint=True,
                           checkpoint_monitor='val_loss', checkpoint_mode='min', checkpoint_save_best_only=True,
                           checkpoint_save_weights_only=False, checkpoint_save_freq='epoch')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'C:\\Users\\st\\Desktop\\VeriSeti'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    print("Training Directory Structure:")
    print_dir_structure(train_dir)

    print("\nValidation Directory Structure:")
    print_dir_structure(val_dir)

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=True, num_workers=0)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device,
                           num_epochs=3)

    torch.save(model_ft.state_dict(), 'animal_classifier_resnet18.pth')
    print('Model saved as animal_classifier_resnet18.pth')

    visualize_model(model_ft, dataloaders, class_names, device)

    # End the MLflow run
    mlflow.end_run()


if __name__ == '__main__':
    main()
