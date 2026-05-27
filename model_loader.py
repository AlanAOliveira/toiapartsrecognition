import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mlflow
import logging

logger = logging.getLogger(__name__)

CAR_PARTS = [
    'headlight', 'taillight', 'front_bumper', 'rear_bumper',
    'hood', 'trunk', 'door', 'fender', 'windshield',
    'side_mirror', 'wheel', 'grille', 'roof', 'license_plate'
]

class CarPartsClassifier:
    def __init__(self, model_path=None, use_mlflow=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = CAR_PARTS
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if use_mlflow and model_path:
            logger.info(f"Loading MLflow model: {model_path}")
            self.model = mlflow.pytorch.load_model(model_path)
        else:
            self.model = self._build_model()
            if model_path:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model ready on {self.device}")

    def _build_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.classes))
        )
        return model

    def predict(self, image, top_k=3):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        return [
            {'part': self.classes[idx.item()],
             'confidence': float(prob.item()) * 100}
            for prob, idx in zip(top_probs, top_indices)
        ]