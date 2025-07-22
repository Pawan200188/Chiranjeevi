from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from database.connection import get_db
from models.scan import ScanResult
from ml_model.model import TumorClassifier

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load("ml_model/Vidyut_4.pth", map_location=device))
model.to(device)
model.eval()

UPLOAD_DIR = "static/uploads/"

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(device)

@router.post("/predict/{filename}")
async def predict(filename: str, db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    image = preprocess_image(file_path)

    with torch.no_grad():
        prediction = model(image)
        prediction = F.softmax(prediction, dim=1)

    prediction = prediction.cpu().numpy()

    print("Raw Model Output:", prediction)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100 

    classification_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    classification_type = classification_labels[predicted_class] if predicted_class < len(classification_labels) else "Unknown"

    db_entry = ScanResult(
        filename=filename, 
        classification_type=classification_type
    )
    db.add(db_entry)
    db.commit()

    return {
        "filename": filename,
        "Classification Type": classification_type,
        "Confidence": f"{confidence:.2f}%",
    }
