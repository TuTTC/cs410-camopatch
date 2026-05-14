import torch
import torchvision
import numpy as np
import argparse
import os

from utils import get_model, load_imagenet_subset, NormalizedModel
from camopatch import CamoPatch

def compute_nnr(original_img, adv_img, patch_size):
    """
    Computes Non-normalised Residual (NNR) as absolute pixel difference.
    """
    diff = torch.abs(original_img - adv_img)
    return diff.sum().item() / (3 * patch_size * patch_size)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CamoPatch")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ImageNet validation directory")
    parser.add_argument('--model', type=str, default='resnet50', help="Target model")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to evaluate")
    parser.add_argument('--max_queries', type=int, default=10000, help="Query budget K")
    parser.add_argument('--patch_size', type=int, default=40, help="Size of the patch")
    parser.add_argument('--num_circles', type=int, default=100, help="Number of circles N")
    parser.add_argument('--sigma', type=float, default=0.1, help="Evolutionary step size sigma")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model {args.model}...")
    raw_model = get_model(args.model, device=device)
    model = NormalizedModel(raw_model).to(device)
    model.eval()

    # Load Dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_imagenet_subset(args.data_dir, num_images=args.num_images)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    attacker = CamoPatch(model, patch_size=args.patch_size, num_circles=args.num_circles, 
                         sigma=args.sigma, device=device)

    total_images = 0
    successful_attacks = 0
    total_l2 = 0.0
    total_nnr = 0.0

    print("Starting evaluation...")
    for i, (img, label) in enumerate(dataloader):
        img = img[0].to(device) # shape (3, H, W)
        label = label.item()
        
        # Verify original prediction
        with torch.no_grad():
            orig_logits = model(img.unsqueeze(0))
            orig_pred = orig_logits.argmax(dim=1).item()
            
        if orig_pred != label:
            print(f"Image {i}: Skipped (Misclassified by original model)")
            continue
            
        total_images += 1
        print(f"Image {i}: True Label={label}, Attacking...")
        
        adv_img, success, l2_dist = attacker.attack(img, label, max_queries=args.max_queries)
        
        if success:
            successful_attacks += 1
            total_l2 += l2_dist
            nnr = compute_nnr(img, adv_img, args.patch_size)
            total_nnr += nnr
            print(f" -> Success! L2={l2_dist:.4f}, NNR={nnr:.4f}")
        else:
            print(" -> Failed.")

    if total_images == 0:
        print("No correctly classified images found.")
        return
        
    asr = successful_attacks / total_images
    avg_l2 = total_l2 / successful_attacks if successful_attacks > 0 else 0
    avg_nnr = total_nnr / successful_attacks if successful_attacks > 0 else 0

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Model: {args.model}")
    print(f"Total Images Evaluated: {total_images}")
    print(f"Attack Success Rate (ASR): {asr:.2%}")
    print(f"Average L2 Distance: {avg_l2:.4f}")
    print(f"Average NNR: {avg_nnr:.4f}")

if __name__ == "__main__":
    main()
