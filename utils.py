import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

def get_model(model_name, device='cpu'):
    if model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == 'vgg16':
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg16(weights=weights)
    elif model_name == 'vit_b_16':
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=weights)
    elif model_name.startswith('at_'):
        from robustbench.utils import load_model
        # E.g., at_resnet50 -> Standard or specific RobustBench model
        if model_name == 'at_resnet50':
            model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf')
        elif model_name == 'at_wideresnet':
            model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')
        else:
            raise ValueError(f"Unknown RobustBench model {model_name}")
    elif model_name == 'patchguard':
        import sys
        patchguard_dir = os.path.join(os.path.dirname(__file__), 'PatchGuard')
        if patchguard_dir not in sys.path:
            sys.path.insert(0, patchguard_dir)
        try:
            import nets.bagnet
            from utils.defense_utils import masking_defense
        except ImportError:
            raise ValueError("PatchGuard repo not found or missing dependencies. Please clone it.")
        
        checkpoint_path = os.path.join(patchguard_dir, 'checkpoints', 'bagnet17_net.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"PatchGuard checkpoint missing at {checkpoint_path}. Please download 'bagnet17_net.pth' from their Google Drive and place it there.")
            
        base_model = nets.bagnet.bagnet17(pretrained=True, clip_range=None, aggregation='none')
        base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['state_dict'])
        
        class PatchGuardWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.window_size = 6 # ceil((32+17-1)/8) assuming patch_size=32
                self.thres = 0.0
                self.masking_defense = masking_defense

            def forward(self, x):
                local_features = self.model(x).detach().cpu().numpy()
                preds = []
                for i in range(len(local_features)):
                    pred = self.masking_defense(local_features[i], thres=self.thres, window_shape=[self.window_size, self.window_size])
                    preds.append(pred)
                batch_size = x.shape[0]
                dummy_logits = torch.zeros((batch_size, 1000), device=x.device)
                for i, p in enumerate(preds):
                    dummy_logits[i, int(p)] = 1.0
                return dummy_logits
                
        model = PatchGuardWrapper(base_model)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model = model.to(device)
    model.eval()
    return model

def load_image(image_path, device='cpu'):
    # Standard ImageNet normalization is NOT applied here 
    # because the patch needs to be generated in [0, 1] pixel space.
    # We will just resize and convert to tensor.
    # Normalization should ideally be done inside the model wrapper if needed.
    # Wait, PyTorch pre-trained models expect normalized inputs.
    # We will handle normalization inside a wrapper model or right before model forward.
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() # scales to [0, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor_img = transform(image).to(device)
    return tensor_img, image

def load_imagenet_subset(data_dir, num_images=1000):
    """
    Loads a subset of ImageNet validation images from a directory.
    Assumes standard ImageNet folder structure or a flat folder of images.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Try to use torchvision ImageFolder
    try:
        dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        # Random subset
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        return subset
    except Exception as e:
        print(f"Failed to load ImageFolder: {e}")
        return None

class NormalizedModel(torch.nn.Module):
    """
    Wraps a model to apply ImageNet normalization on the fly.
    This allows the adversarial attack to work directly in [0, 1] pixel space.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x):
        device = x.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)
