import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import wandb
import yaml # Added for YAML loading
from model import SimpleNN # <--- IMPORT ADDED

# Load configuration from YAML file
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, 'r') as file:
    yaml_config = yaml.safe_load(file)

# Initialize Weights & Biases, passing the loaded config
wandb.init(project="simple-mnist-training", config=yaml_config)

# Configuration - now accessed from wandb.config which holds the YAML content
config = wandb.config
# The actual values like config.epochs, config.lr, config.batch_size
# will be automatically available if they were in the YAML file.

# Device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

# Simple Neural Network Model - REMOVED FROM HERE
# class SimpleNN(nn.Module):
#    ...

model = SimpleNN().to(device) # <--- NOW USES IMPORTED SimpleNN

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Training loop
print("Starting training...")
for epoch in range(config.epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    epoch_labels = []
    epoch_preds = []

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store predictions and labels for epoch accuracy
        _, predicted = torch.max(outputs.data, 1)
        epoch_preds.extend(predicted.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_step": i+1})

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
    print(f'Epoch [{epoch+1}/{config.epochs}] completed. Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
    wandb.log({
        "epoch_loss": epoch_loss,
        "epoch_accuracy": epoch_accuracy,
        "epoch": epoch + 1
    })

print("Training finished.")

# Evaluation
model.eval() # Set model to evaluation mode
all_preds = []
all_labels = []

# Create W&B Tables
# Table for a sample of test predictions
sample_predictions_table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])
# Table for misclassified examples
misclassified_table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])

MAX_SAMPLES_TO_LOG = 100
samples_logged = 0

with torch.no_grad(): # No need to track gradients for evaluation
    for i, (images, labels) in enumerate(test_loader):
        images_device = images.to(device)
        labels_device = labels.to(device)
        outputs = model(images_device)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Log samples and misclassifications
        for j in range(images.size(0)): # Iterate through images in the batch
            true_label = labels[j].item()
            pred_label = predicted[j].item()

            # Log to sample predictions table
            if samples_logged < MAX_SAMPLES_TO_LOG:
                # Convert image tensor to wandb.Image
                # We need to undo normalization for better visualization if needed, 
                # or ensure it's in a displayable format (e.g., channel first for single channel grayscale)
                # For MNIST, ToTensor() already puts it in (C, H, W) and it's grayscale.
                sample_img = wandb.Image(images[j]) 
                sample_predictions_table.add_data(sample_img, true_label, pred_label)
                samples_logged += 1

            # Log to misclassified table if prediction is wrong
            if pred_label != true_label:
                misclassified_img = wandb.Image(images[j])
                misclassified_table.add_data(misclassified_img, true_label, pred_label)

test_accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy of the model on the {len(test_dataset)} test images: {test_accuracy * 100:.2f}%')
wandb.log({"test_accuracy": test_accuracy})

# Log the tables
wandb.log({"sample_test_predictions": sample_predictions_table})
wandb.log({"misclassified_test_examples": misclassified_table})
print("Logged sample predictions and misclassifications to W&B Tables.")

# Save the model locally
MODEL_PATH = "mnist_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Log the model as a W&B Artifact
model_artifact = wandb.Artifact(
    "mnist-simple-nn", type="model",
    description="Simple Neural Network trained on MNIST",
    metadata=dict(config) # Log a copy of the config with the artifact
)
model_artifact.add_file(MODEL_PATH)
wandb.log_artifact(model_artifact)
print("Model logged as a W&B Artifact.")

wandb.finish()
print("Run completed and logged to Weights & Biases.") 