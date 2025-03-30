# Importing necessary dependencies
import torch #pytorch
from PIL import Image #image manipulation (Pillow)
from torch import nn,save,load 
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Create an instance of the image classifier model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier= mnistModel().to(device)
# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()

# Load the saved model
save_path='modelState.pth'
weight_dir = Path("weight")
weight_dir.mkdir(exist_ok=True)
self.model_path = weight_dir / save_path
if self.model_path.exists():
  with open('modelState.pth', 'rb') as f: 
     classifier.load_state_dict(load(f))  
# Train the model
for epoch in range(10):  # Train for 10 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = classifier(images)  # Forward pass
        loss = loss_fun(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")



# Save the trained model
torch.save(classifier.state_dict(), 'modelState.pth')
