from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names (10 skin conditions)
CLASS_NAMES = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
               'Eczema Photos', 'Melanocytic nevus', 'Melanoma Skin Cancer Nevi and Moles',
               'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

# Suggested treatments
TREATMENTS = {
    'Actinic keratosis': 'Cryotherapy, topical 5-fluorouracil, or imiquimod cream...',
    'Atopic Dermatitis': 'Use emollients (moisturizers) regularly...',
    'Benign keratosis': 'No treatment required unless cosmetic removal is desired...',
    'Dermatofibroma': 'Usually no treatment needed...',
    'Eczema Photos': 'Hydrocortisone creams, corticosteroid ointments...',
    'Melanocytic nevus': 'Generally harmless but should be monitored...',
    'Melanoma Skin Cancer Nevi and Moles': 'Immediate biopsy and excision...',
    'Squamous cell carcinoma': 'Surgical excision, Mohs surgery...',
    'Tinea Ringworm Candidiasis': 'Topical antifungal treatments...',
    'Vascular lesion': 'Laser therapy, sclerotherapy, or surgical excision...'
}

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet152 model (not pretrained on ImageNet)
model = models.resnet152(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("resnet152 bestmodel.pth", map_location=device), strict=True)
model = model.to(device)
model.eval()

# Validation transform
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Image prediction logic
def predict_image(image_path, model, disease_threshold=0.75, irrelevant_threshold=0.20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_percent = confidence.item() * 100
        max_prob = confidence.item()

    # Case 1: Irrelevant image (not skin)
    if max_prob < irrelevant_threshold:
        return ("irrelevant", confidence_percent)

    # Case 2: Skin image but not confidently one of 10 diseases
    if confidence_percent < disease_threshold * 100:
        return ("unknown_skin", confidence_percent)

    # Case 3: Valid prediction
    return (predicted_class, confidence_percent)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            predicted_class, confidence_percent = predict_image(filepath, model)

            # Irrelevant image
            if predicted_class == "irrelevant":
                return render_template('index.html',
                                       prediction="Irrelevant image detected",
                                       confidence=confidence_percent,
                                       treatment="Please upload a clear image of the affected skin area.",
                                       image_path='uploads/' + file.filename)

            # Skin image but unknown disease
            elif predicted_class == "unknown_skin":
                return render_template('index.html',
                                       prediction="No skin disease detected",
                                       confidence=confidence_percent,
                                       treatment="The image appears to show skin but does not match any known disease.",
                                       image_path='uploads/' + file.filename)

            # Valid prediction with known disease
            elif predicted_class in TREATMENTS:
                suggested_treatment = TREATMENTS[predicted_class]
            else:
                suggested_treatment = "No treatment information available."

            return render_template('index.html',
                                   prediction=predicted_class,
                                   confidence=confidence_percent,
                                   treatment=suggested_treatment,
                                   image_path='uploads/' + file.filename)

        else:
            return "No file uploaded", 400

    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
