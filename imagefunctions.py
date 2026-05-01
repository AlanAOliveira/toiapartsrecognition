import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)
# Modify for car parts (e.g., 20 classes)
model.fc = torch.nn.Linear(model.fc.in_features, 20)

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

image = transform(Image.open("part.jpg")).unsqueeze(0)
output = model(image)
predicted_class = output.argmax(1)