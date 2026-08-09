from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from .model import build_model, CLASS_NAMES

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path='models/catdog_cnn.pt'):
    model = build_model()
    path = Path(model_path)
    if path.exists():
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
    model.eval()
    return model

def preprocess_for_inference(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert('RGB')
    return _transform(image).unsqueeze(0)

def predict_image(model, image):
    tensor = preprocess_for_inference(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    label = max(result, key=result.get)
    return {'label': label, 'probabilities': result}
