import torch
from DenseNet121 import model, device
from TRAIN_DenseNet121 import val_loader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Classes names
class_names = ['Anthias anthias', 'Belone belone', 'Boops boops', 'Chlorophthalmus agassizi', 'Coris julis', 'Epinephelus caninus', 'Gobius niger', 'Mugil cephalus', 'Phycis phycis','Polyprion americanus', 'Pseudocaranx dentex', 'Rhinobatos cemiculus', 'Scomber scombrus', 'Solea solea', 'Squalus acanthias', 'Tetrapturus belone', 'Trachinus draco', 'Trigloporus lastoviza']


if __name__ == "__main__":
    # Validation process
    model.load_state_dict(torch.load('MFC_focal_loss_last.pth', map_location='cpu'))

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()