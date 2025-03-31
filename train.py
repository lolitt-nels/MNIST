# Importing necessary dependencies
from pathlib import Path
import torch  # pytorch
from PIL import Image  # image manipulation (Pillow)
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import mnistModel

def load_and_preprocess_data(batch_size=32):

    train_dataset = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader



def load_model(model_class, model_path, device):

    classifier = model_class().to(device)
    if Path(model_path).exists():
        try:
            classifier.load_state_dict(torch.load(model_path))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No saved model found. Starting from scratch.")
    return classifier


def train_model(classifier, train_loader, optimizer, loss_fun, device, epochs=10):
   
    for epoch in range(epochs):  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = classifier(images)  # Forward pass
            loss = loss_fun(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f"Epoch:{epoch} loss is {loss.item()}")

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, optimizer, and loss initialization
    classifier = load_model(
        mnistModel, "weight/modelState.pth", device
    )  # Load the saved model
    optimizer = Adam(classifier.parameters(), lr=0.001)
    loss_fun = nn.CrossEntropyLoss()

    # Data loading
    train_loader = load_and_preprocess_data()

    # Create weight directory
    weight_dir = Path("weight")
    weight_dir.mkdir(exist_ok=True)

    # Train the model
    train_model(classifier, train_loader, optimizer, loss_fun, device)

    # Save the trained model
    torch.save(classifier.state_dict(), "modelState.pth")
    print("Model training complete and saved.")
