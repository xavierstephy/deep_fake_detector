<<<<<<< HEAD
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os

def load_trained_model(model_path, num_classes=2):
    """Load the trained model"""
    # Build the same model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_path, img_size=224):
    """Preprocess image for model input"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(model, image_path):
    """Predict if image is real or fake"""
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        fake_prob = probabilities[0][1].item()  # Probability of being fake
        real_prob = probabilities[0][0].item()  # Probability of being real
        
    return {
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
        'confidence': max(fake_prob, real_prob)
    }

def main():
    model_path = "model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run: Copy-Item .\\checkpoints\\best_model.pt .\\model.pth")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path)
    print("Model loaded successfully!")
    
    # Test with sample images if available
    test_images = []
    
    # Check for test images in dataset
    if os.path.exists("dataset2/val/real"):
        real_images = [os.path.join("dataset2/val/real", f) for f in os.listdir("dataset2/val/real")[:3]]
        test_images.extend(real_images)
    
    if os.path.exists("dataset2/val/fake"):
        fake_images = [os.path.join("dataset2/val/fake", f) for f in os.listdir("dataset2/val/fake")[:3]]
        test_images.extend(fake_images)
    
    if test_images:
        print(f"\nTesting with {len(test_images)} sample images:")
        print("-" * 50)
        
        for image_path in test_images:
            try:
                result = predict_image(model, image_path)
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Prediction: {result['prediction']}")
                print(f"Fake Probability: {result['fake_probability']:.3f}")
                print(f"Real Probability: {result['real_probability']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                print("-" * 30)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    else:
        print("\nNo test images found. You can test with:")
        print("python test_model.py")
        print("\nOr use the Streamlit app:")
        print("streamlit run app.py")

if __name__ == "__main__":
    main()
=======
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os

def load_trained_model(model_path, num_classes=2):
    """Load the trained model"""
    # Build the same model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_path, img_size=224):
    """Preprocess image for model input"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(model, image_path):
    """Predict if image is real or fake"""
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        fake_prob = probabilities[0][1].item()  # Probability of being fake
        real_prob = probabilities[0][0].item()  # Probability of being real
        
    return {
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
        'confidence': max(fake_prob, real_prob)
    }

def main():
    model_path = "model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run: Copy-Item .\\checkpoints\\best_model.pt .\\model.pth")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path)
    print("Model loaded successfully!")
    
    # Test with sample images if available
    test_images = []
    
    # Check for test images in dataset
    if os.path.exists("dataset2/val/real"):
        real_images = [os.path.join("dataset2/val/real", f) for f in os.listdir("dataset2/val/real")[:3]]
        test_images.extend(real_images)
    
    if os.path.exists("dataset2/val/fake"):
        fake_images = [os.path.join("dataset2/val/fake", f) for f in os.listdir("dataset2/val/fake")[:3]]
        test_images.extend(fake_images)
    
    if test_images:
        print(f"\nTesting with {len(test_images)} sample images:")
        print("-" * 50)
        
        for image_path in test_images:
            try:
                result = predict_image(model, image_path)
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Prediction: {result['prediction']}")
                print(f"Fake Probability: {result['fake_probability']:.3f}")
                print(f"Real Probability: {result['real_probability']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                print("-" * 30)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    else:
        print("\nNo test images found. You can test with:")
        print("python test_model.py")
        print("\nOr use the Streamlit app:")
        print("streamlit run app.py")

if __name__ == "__main__":
    main()
>>>>>>> 0c57338d10119562221af77a603da511f81f8170
