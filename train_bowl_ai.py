import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Root directory containing the dataset folders
DATASET_DIR = "dataset"

# Number of samples per batch
BATCH_SIZE = 16

# Total number of training epochs
EPOCHS = 10

# Learning rate for the optimizer
LR = 1e-4

# Path where the best model weights will be saved
MODEL_SAVE_PATH = "bowl_classifier.pth"

def main():
    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Data augmentation and preprocessing for training images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to model input size
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color augmentation
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(  # Normalize using ImageNet mean and standard deviation
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Preprocessing for validation images (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to model input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(  # Normalize using ImageNet mean and standard deviation
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load training dataset from folder structure
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_DIR, "train"),
        transform=train_transform
    )

    # Load validation dataset from folder structure
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_DIR, "val"),
        transform=val_transform
    )

    # Print class-to-index mapping determined by ImageFolder
    print("Class mapping from ImageFolder:", train_dataset.class_to_idx)

    # DataLoader for batching and shuffling training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # DataLoader for validation data
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Load pretrained MobileNetV3 Small model
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    # Replace the final classifier layer to match the number of dataset classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(train_dataset.classes))

    # Move model to the selected device
    model = model.to(device)

    # Define classification loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Track best validation accuracy for saving the best model
    best_val_acc = 0.0

    # Main training loop
    for epoch in range(EPOCHS):
        # Set model to training mode
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate over training batches
        for images, labels in train_loader:
            # Move batch data to selected device
            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            # Accumulate total training loss
            train_loss += loss.item() * images.size(0)

            # Get predicted class indices
            preds = outputs.argmax(dim=1)

            # Count correct predictions
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Compute average training loss and accuracy
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Set model to evaluation mode
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            # Iterate over validation batches
            for images, labels in val_loader:
                # Move validation data to selected device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute validation loss
                loss = criterion(outputs, labels)

                # Accumulate validation loss
                val_loss += loss.item() * images.size(0)

                # Get predicted class indices
                preds = outputs.argmax(dim=1)

                # Count correct predictions
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # Compute average validation loss and accuracy
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Print training and validation metrics for the current epoch
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Save the model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")

    # Print final training summary
    print("Training finished.")
    print("Best validation accuracy:", best_val_acc)

# Run the script only if executed directly
if __name__ == "__main__":
    main()