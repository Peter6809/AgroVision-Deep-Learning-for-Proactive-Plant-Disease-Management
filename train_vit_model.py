# prompt: build model for plant disease detect using vision transformer and try to make less training for complete process fastly and i want to save the output of model like this model_path = "outputs/plant_disease_model.pth"
# this is my data set path = "/content/drive/MyDrive/Dataset/plants_data". generate code

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the Vision Transformer model (simplified for faster training)
class SimpleViT(nn.Module):
    def __init__(self, num_classes=38):  # Assuming 38 classes initially
        super(SimpleViT, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 224 * 224, num_classes)  # Adjust based on image size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define hyperparameters
batch_size = 32  # Start with a smaller batch size
learning_rate = 0.001
num_epochs = 3  # Reduced number of epochs for faster training


# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data_path = "data/plants_data"  # Update path

try:
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(dataset.classes) # automatically find number of classes
    print(f"Number of classes detected: {num_classes}")


    # Initialize the model, loss function, and optimizer
    model = SimpleViT(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
          images, labels = images.to(device), labels.to(device)

          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          progress_bar.set_postfix({'Loss': loss.item()})

    # Save the trained model
    model_path = "outputs/disease_vit_model.pth"
    # !mkdir -p outputs # create directory if not exists
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")