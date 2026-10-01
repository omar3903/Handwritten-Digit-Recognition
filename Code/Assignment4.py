"""
Author: Omar Abdeen
CP 467 Assignment 4
22/11/2024
inspired by the PyTorch transfer learning tutorial : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
import time
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, models

cudnn.benchmark = True

def load_data():
    """ 
    Load and preprocess our handwritten digit dataset 
    """
    train_imgs, train_labs = [], []
    val_imgs, val_labs = [], []
    test_imgs, test_labs = [], []
    
    # Data preprocessing transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    for digit in range(10):
        # Training set (1-8)
        for i in range(1, 9):
            img_path = f"{digit}_{i}.jpg"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = transform(img)
                train_imgs.append(img)
                train_labs.append(digit)

        # Validation set (9-10)
        for i in range(9, 11):
            img_path = f"{digit}_{i}.jpg"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = transform(img)
                val_imgs.append(img)
                val_labs.append(digit)

        # Test set (11-12)
        for i in range(11, 13):
            img_path = f"{digit}_{i}.jpg"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = transform(img)
                test_imgs.append(img)
                test_labs.append(digit)

    # Return the images and digits from each set
    return (torch.stack(train_imgs), torch.tensor(train_labs)), \
           (torch.stack(val_imgs), torch.tensor(val_labs)), \
           (torch.stack(test_imgs), torch.tensor(test_labs))

def prepare_model(mode='last_layer'):
    """
    Initialize the model in prep for transfer learning
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify final layer for 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    if mode == 'last_layer':
        model.fc.requires_grad = True
    elif mode == 'multi_layers':
        for layer in [model.layer4, model.fc]:
            for param in layer.parameters():
                param.requires_grad = True
    
    return model

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    """Train the model"""
    since = time.time()
    best_model_wts = model.state_dict().copy()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.2f}%')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNIST data transform
    mnist_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    results = {}

    # Task 1 & 2: Train on MNIST, test on both MNIST and custom test set
    print("\nTraining Model 1")
    model1 = prepare_model('last_layer').to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Load MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root='.', train=True,
                                           transform=mnist_transform, download=True)
    mnist_test = torchvision.datasets.MNIST(root='.', train=False,
                                          transform=mnist_transform, download=True)
    
    # Split training into train/val
    train_size = 50000
    val_size = len(mnist_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, 
                                                             [train_size, val_size])
    
    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=128)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    # Optimizer and scheduler
    optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model1
    model1 = train_model(model1, criterion, optimizer, scheduler, 
                        dataloaders, dataset_sizes, device, num_epochs=35)
    
    # Task 1: Test on MNIST test set
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128)
    correct = 0
    total = 0
    model1.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    mnist_test_acc = 100 * correct / total
    results['model1_mnist'] = mnist_test_acc
    print(f'Model 1 accuracy on MNIST test set: {mnist_test_acc:.2f}%')
    
    # Task 2: Test same model on custom test set
    _, _, test_data = load_data()
    model1.eval()
    with torch.no_grad():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        outputs = model1(test_images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == test_labels).sum().item()
        custom_test_acc = 100 * correct / len(test_labels)
    
    results['model1_custom'] = custom_test_acc
    print(f'Model 1 accuracy on custom test set: {custom_test_acc:.2f}%')

    # Task 3: Train new model on custom dataset (last layer only)
    print("\nTraining Model 2 (Custom Dataset)...")
    model2 = prepare_model('last_layer').to(device)
    train_data, val_data, test_data = load_data()
    
    # Create dataloaders for custom dataset
    dataloaders = {
        'train': torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            train_data[0], train_data[1]), batch_size=16, shuffle=True),
        'val': torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            val_data[0], val_data[1]), batch_size=16)
    }
    dataset_sizes = {'train': len(train_data[0]), 'val': len(val_data[0])}
    
    optimizer = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    
    # Train model2
    model2 = train_model(model2, criterion, optimizer, None,
                        dataloaders, dataset_sizes, device, num_epochs=40)
    
    # Test model2 on custom test set
    model2.eval()
    with torch.no_grad():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        outputs = model2(test_images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == test_labels).sum().item()
        test_acc = 100 * correct / len(test_labels)
    
    results['model2'] = test_acc
    print(f'Model 2 accuracy on custom test set: {test_acc:.2f}%')

    # Task 4: Train model3 with multiple unfrozen layers
    print("\nTraining Model 3 (Custom Dataset, Multiple Layers)...")
    model3 = prepare_model('multi_layers').to(device)
    optimizer = optim.SGD(model3.parameters(), lr=0.001, momentum=0.9)
    
    # Train model3
    model3 = train_model(model3, criterion, optimizer, None,
                        dataloaders, dataset_sizes, device, num_epochs=40)
    
    # Test model3 on custom test set
    model3.eval()
    with torch.no_grad():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        outputs = model3(test_images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == test_labels).sum().item()
        test_acc = 100 * correct / len(test_labels)
    
    results['model3'] = test_acc
    print(f'Model 3 accuracy on custom test set: {test_acc:.2f}%')

    # Print the results
    print("Model & Test Set Accuracy Score")
    print("-" * 40)
    print(f"Model 1 on MNIST:\t{results['model1_mnist']:.2f}%")
    print(f"Model 1 on Custom:\t{results['model1_custom']:.2f}%")
    print(f"Model 2 on Custom:\t{results['model2']:.2f}%")
    print(f"Model 3 on Custom:\t{results['model3']:.2f}%")

if __name__=="__main__":
    main()