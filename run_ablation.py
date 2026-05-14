import torch
import numpy as np
import argparse
import os
import json

from utils import get_model, load_imagenet_subset, NormalizedModel, load_image
from camopatch import CamoPatch

def run_experiment(attacker, image, label, max_queries):
    # Run the attack
    adv_img, success, l2_dist = attacker.attack(image, label, max_queries=max_queries)
    return success, l2_dist

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study for CamoPatch")
    parser.add_argument('--image_path', type=str, default='sample.jpg', help="Path to a single image to use (if data_dir is not provided)")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to ImageNet validation directory (optional)")
    parser.add_argument('--model', type=str, default='resnet50', help="Target model")
    parser.add_argument('--max_queries', type=int, default=2000, help="Query budget K for quick testing")
    parser.add_argument('--num_images', type=int, default=1, help="Number of images to evaluate if data_dir is provided")
    parser.add_argument('--out_file', type=str, default='ablation_results.json', help="Output file for results")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model {args.model}...")
    raw_model = get_model(args.model, device=device)
    model = NormalizedModel(raw_model).to(device)
    model.eval()

    # Load Dataset or Single Image
    images_labels = []
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"Loading dataset from {args.data_dir}...")
        dataset = load_imagenet_subset(args.data_dir, num_images=args.num_images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for img, label in dataloader:
            img = img[0].to(device)
            label = label.item()
            with torch.no_grad():
                orig_logits = model(img.unsqueeze(0))
                orig_pred = orig_logits.argmax(dim=1).item()
            if orig_pred == label:
                images_labels.append((img, label))
    else:
        print(f"Loading single image {args.image_path}...")
        img, _ = load_image(args.image_path, device=device)
        with torch.no_grad():
            orig_logits = model(img.unsqueeze(0))
            label = orig_logits.argmax(dim=1).item() # Use original prediction as label
        images_labels.append((img, label))

    if not images_labels:
        print("No valid images to run ablation study on.")
        return

    results = {}

    def test_param(param_name, param_values, default_kwargs):
        print(f"\n--- Testing {param_name} ---")
        results[param_name] = {}
        for val in param_values:
            print(f"Value: {val}")
            kwargs = default_kwargs.copy()
            kwargs[param_name] = val
            
            successes = 0
            total_l2 = 0
            
            for img, label in images_labels:
                attacker = CamoPatch(model, device=device, **kwargs)
                success, l2 = run_experiment(attacker, img, label, args.max_queries)
                if success:
                    successes += 1
                    total_l2 += l2
                    
            asr = successes / len(images_labels)
            avg_l2 = total_l2 / successes if successes > 0 else 0
            
            results[param_name][val] = {'ASR': asr, 'Avg_L2': avg_l2}
            print(f" -> ASR: {asr:.2%}, Avg L2: {avg_l2:.4f}")

    base_kwargs = {
        'patch_size': 40,
        'num_circles': 100,
        'sigma': 0.1,
        't_init': 300,
        'l_i': 4
    }

    # 4.1.1: Ảnh hưởng của N (num_circles)
    test_param('num_circles', [10, 50, 100, 200], base_kwargs)
    
    # 4.1.2: Ảnh hưởng của Sigma (sigma)
    test_param('sigma', [0.01, 0.05, 0.1, 0.5], base_kwargs)
    
    # 4.1.3: Ảnh hưởng của l_i (Location schedule)
    test_param('l_i', [1, 4, 9, 19], base_kwargs)
    
    # 4.1.4: Ảnh hưởng của t (Temperature)
    test_param('t_init', [10, 100, 300, 1000], base_kwargs)

    # Save results
    with open(args.out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved ablation results to {args.out_file}")

if __name__ == "__main__":
    main()
