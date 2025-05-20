import datetime
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from DenseNet121 import model, train_function, validate_function, criterion, optimizer, scheduler
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")

# Training definition
n_epochs = 10
batch_size = 32
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Dataset transformations definition
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset local path
train_dir = '/home/adolfo/Documents/TFG/DATASETS/Species - Mediterráneo Aug/Training_Set'
val_dir = '/home/adolfo/Documents/TFG/DATASETS/Species - Mediterráneo Aug/Test_Set'

# Dataset transformation loading
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# Weighted sampler for underrepresented classes
targets = train_dataset.targets
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Dataset loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Learning Rate tracker function
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print(f"\nTraining started at: {datetime.datetime.now().strftime('%H:%M')}\n")

train_loss_array = []
val_loss_array = []
lr_values = []

if __name__ == "__main__":
    # Training loop
    for epoch in range(n_epochs):
        # Train and validation metrics calculation
        train_loss, train_acc = train_function(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_function(model, val_loader, criterion)

        # Metrics tracking
        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        # Scheduler update
        scheduler.step(val_loss)
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Learning Rate tracking
        current_lr = get_lr(optimizer)
        lr_values.append(current_lr)

        print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"\nEpoch {epoch + 1} ended at: {datetime.datetime.now().strftime('%H:%M')}\n")

        # Early Stop conditions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "MFC_focal_loss_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Results plotting
    epochs_range = range(1, len(train_loss_array) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_array, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, val_loss_array, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, lr_values, label='Learning Rate', color='green', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate per Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "MFC_focal_loss_best.pth")
    print(f"\nTraining ended at: {datetime.datetime.now().strftime('%H:%M')}\n")