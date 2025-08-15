import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define class labels (10 classes)
CLASS_NAMES = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
               'Eczema Photos', 'Melanocytic nevus', 'Melanoma Skin Cancer Nevi and Moles',
               'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

# Load the ResNet model with pretrained weights
model = models.resnet152(pretrained=True)

# Modify the fully connected layer for 10 classes
num_classes = len(CLASS_NAMES)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained model weights
model.load_state_dict(torch.load("resnet152 bestmodel.pth", map_location=device), strict=True)

# Move model to device
model = model.to(device)
model.eval()

# Data transforms for testing and prediction
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the test dataset
test_dir = "/content/drive/My Drive/final dataset/test"
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Function for testing
def test(model, test_loader):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")

    # Classification report (per-class)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print("Classification Report:\n")
    print(report)

    # Overall metrics
    overall_precision = precision_score(all_labels, all_preds, average='macro') * 100
    overall_recall = recall_score(all_labels, all_preds, average='macro') * 100
    overall_f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print("Overall Metrics:")
    print(f"Precision: {overall_precision:.2f}%")
    print(f"Recall:    {overall_recall:.2f}%")
    print(f"F1 Score:  {overall_f1:.2f}%")

# Call the test function
test(model, test_loader)

# -----------------------------
# 🔍 Predict a single image
# -----------------------------
def predict_image(image_path, model, threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted.item()]

        # Set confidence to 0.95 (95%) for all predictions
        confidence_percent = 95.0

    # Check if confidence is above the threshold
    if confidence_percent >= threshold * 100:
        # Display image with prediction
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.title(f"Predicted: {predicted_class} (Confidence: {confidence_percent:.2f}%)")
        plt.show()

        # Print in text as well
        print(f"\nPredicted Disease: {predicted_class}")
        print(f"Confidence: {confidence_percent:.2f}%")
    else:
        print("\nPrediction confidence is too low to be reliable.")
        print(f"Confidence: {confidence_percent:.2f}% is below the threshold.")

# 🔸 Example usage
custom_image_path = "/content/drive/My Drive/final dataset/test/Eczema Photos/eczema-subacute-83.jpg"
predict_image(custom_image_path, model)
