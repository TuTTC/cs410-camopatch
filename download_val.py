import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

def main():
    parser = argparse.ArgumentParser(description="Download 1000 ImageNet Validation Images")
    parser.add_argument('--output_dir', type=str, default='imagenet_val_1000', help='Output directory')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'vgg16', 'vit_b_16', 'at_resnet50', 'at_wideresnet', 'patchguard'], help='Models to use for filtering')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to download')
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library first: pip install datasets")
        return

    # Load all models
    print(f"Loading {len(args.models)} models on {args.device} for strict filtering: {args.models}")
    import sys
    sys.path.append(os.path.dirname(__file__))
    try:
        from utils import get_model, NormalizedModel
    except ImportError:
        print("Error: Could not import utils.py. Ensure you are running this from the CamoPatch directory.")
        return

    models = []
    for m_name in args.models:
        try:
            print(f"Loading {m_name}...")
            raw_model = get_model(m_name, device=args.device)
            # Standard models and PatchGuard need ImageNet normalization wrapper
            if m_name in ['resnet50', 'vgg16', 'vit_b_16', 'patchguard']:
                model = NormalizedModel(raw_model).to(args.device)
            else:
                model = raw_model # RobustBench models handle normalization internally or expect [0,1]
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Failed to load {m_name}: {e}")
            if m_name == 'patchguard':
                print("Note: To use PatchGuard, you must download 'bagnet17_net.pth' into PatchGuard/checkpoints/.")
            print("Exiting due to model load failure.")
            return

    # Standard ImageNet transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print("Loading official ImageNet-1K validation set from Hugging Face (ILSVRC/imagenet-1k)...")
    try:
        ds = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNOTE: ILSVRC/imagenet-1k is a gated dataset.")
        print("Please ensure you have:")
        print("1. Accepted the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("2. Run 'huggingface-cli login' in your terminal with your access token.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading {args.num_images} images to '{args.output_dir}'...")
    count = 0
    
    # Progress tracking
    if 'tqdm' in globals():
        pbar = tqdm(total=args.num_images)
    else:
        pbar = None

    for sample in ds:
        image = sample['image']
        label = sample['label']
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Transform for inference
        img_tensor = transform(image).unsqueeze(0).to(args.device)
        
        # Check through all models
        all_correct = True
        with torch.no_grad():
            for model in models:
                logits = model(img_tensor)
                pred = logits.argmax(dim=1).item()
                if pred != label:
                    all_correct = False
                    break
        
        if not all_correct:
            continue # Skip this image
            
        # Create class directory
        class_dir = os.path.join(args.output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        # Save image
        file_path = os.path.join(class_dir, f"val_{label}_{count}.jpg")
        image.save(file_path)
        
        count += 1
        if pbar:
            pbar.update(1)
        else:
            if count % 100 == 0:
                print(f"Downloaded {count}/{args.num_images} images...")

        if count >= args.num_images:
            break
            
    if pbar:
        pbar.close()
        
    print(f"Done! Downloaded {args.num_images} images successfully.")
    print(f"You can now run evaluation using: python evaluate.py --data_dir {args.output_dir} --num_images {args.num_images}")

if __name__ == "__main__":
    main()
