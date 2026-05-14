import torch
import torchvision
import urllib.request
import os
from PIL import Image
import numpy as np

from utils import get_model, load_image, NormalizedModel
from camopatch import CamoPatch

def download_sample_image(filename="sample.jpg"):
    if not os.path.exists(filename):
        # A sample image from wikimedia
        url = "https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg"
        print(f"Downloading sample image to {filename}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())

def save_image(tensor, filename):
    # tensor shape: (3, H, W) in [0, 1]
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np_img * 255).astype(np.uint8)
    Image.fromarray(np_img).save(filename)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print("Loading ResNet-50...")
    raw_model = get_model('resnet50', device=device)
    model = NormalizedModel(raw_model).to(device)
    model.eval()

    # Prepare Image
    img_path = "sample.jpg"
    download_sample_image(img_path)
    img_tensor, _ = load_image(img_path, device=device)
    
    # Get initial prediction
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        pred_label = logits.argmax(dim=1).item()
        # Golden Retriever in ImageNet is class 207
        print(f"Original Prediction: {pred_label}")

    # Run Attack
    # Paper uses K=10,000, patch_size=40, num_circles=100
    # For quick testing, we can use max_queries=1000
    print("Initializing CamoPatch attack...")
    attacker = CamoPatch(model, patch_size=40, num_circles=100, sigma=0.1, device=device)
    
    print("Running attack (this may take a few minutes)...")
    adv_img, success, l2_dist = attacker.attack(img_tensor, pred_label, max_queries=1000)
    
    # Verify new prediction
    with torch.no_grad():
        new_logits = model(adv_img.unsqueeze(0))
        new_pred = new_logits.argmax(dim=1).item()
    
    print(f"Attack Success: {success}")
    print(f"New Prediction: {new_pred}")
    print(f"L2 Distance of Patch: {l2_dist:.4f}")
    
    # Save adversarial image
    save_image(adv_img, "adv_sample.jpg")
    print("Saved adversarial image to adv_sample.jpg")

if __name__ == "__main__":
    main()
