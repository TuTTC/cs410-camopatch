# %% [markdown]
# # Phase 0 — Environment setup
# ## 0.1 Cài đặt dependencies
# Chạy lệnh sau để cài đặt toàn bộ thư viện cần thiết:
# pip install -r requirements.txt
# Các thư viện chính bao gồm: torch, torchvision, numpy, Pillow, scipy, robustbench, datasets, autoattack, tensorflow, opencv-python, huggingface_hub.

# %% [markdown]
# ## 0.1.1 Đăng nhập Hugging Face
# Vì tập dữ liệu ImageNet-1K chính thức (`ILSVRC/imagenet-1k`) là tập dữ liệu đóng (gated), bạn cần:
# 1. Chấp nhận điều khoản tại: https://huggingface.co/datasets/ILSVRC/imagenet-1k
# 2. Chạy ô dưới đây để đăng nhập bằng Access Token (lấy tại https://huggingface.co/settings/tokens)

# %%
try:
    from huggingface_hub import notebook_login
    notebook_login()
except ImportError:
    print("Vui lòng cài đặt huggingface_hub: pip install huggingface_hub")

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import argparse

# %% [markdown]
# ## 0.2 Chuẩn bị dataset
# ImageNet validation set — lấy 1000 ảnh được phân loại đúng, resize về 224x224x3

# %%
def download_imagenet_subset(output_dir='imagenet_val_1000', num_images=1000, 
                             models_to_check=['resnet50', 'vgg16', 'vit_b_16', 'at_resnet50', 'at_wideresnet', 'patchguard']):
    """
    Tải tập ảnh ImageNet validation từ Hugging Face.
    Chỉ giữ lại những ảnh được tất cả các models_to_check đoán đúng.
    Yêu cầu thư viện: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Vui lòng cài đặt thư viện datasets: pip install datasets")
        return False

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_images:
        print(f"Dataset đã tồn tại đủ số lượng ở thư mục: {output_dir}")
        return True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang tải {len(models_to_check)} models lên {device} để lọc ảnh...")
    
    loaded_models = []
    for m_name in models_to_check:
        try:
            raw_model = get_model(m_name, device=device)
            if m_name in ['resnet50', 'vgg16', 'vit_b_16', 'patchguard']:
                model = NormalizedModel(raw_model).to(device)
            else:
                model = raw_model
            model.eval()
            loaded_models.append(model)
        except Exception as e:
            print(f"Không thể tải model {m_name}: {e}")
            if m_name == 'patchguard':
                print("Lưu ý: Bạn cần tải 'bagnet17_net.pth' từ Google Drive của tác giả vào thư mục PatchGuard/checkpoints/")
            return False

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print(f"Đang tải ảnh từ Hugging Face (ILSVRC/imagenet-1k) và lọc qua {len(models_to_check)} models...")
    try:
        ds = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, trust_remote_code=True)
        os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        for sample in ds:
            image = sample['image']
            label = sample['label']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            all_correct = True
            with torch.no_grad():
                for model in loaded_models:
                    logits = model(img_tensor)
                    pred = logits.argmax(dim=1).item()
                    if pred != label:
                        all_correct = False
                        break
                        
            if not all_correct:
                continue
                
            class_dir = os.path.join(output_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            file_path = os.path.join(class_dir, f"val_{label}_{count}.jpg")
            image.save(file_path)
            
            count += 1
            if count % 10 == 0:
                print(f"Đã lọc thành công {count}/{num_images} ảnh...")
            if count >= num_images:
                break
        print("Tải và lọc ảnh hoàn tất!")
        return True
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        print("\nLƯU Ý: ILSVRC/imagenet-1k là tập dữ liệu yêu cầu quyền truy cập.")
        print("Vui lòng đảm bảo bạn đã:")
        print("1. Ấn 'Agree and access repository' tại https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("2. Chạy lệnh 'huggingface-cli login' trong terminal và nhập Access Token.")
        return False


class ImageNetFolder(torchvision.datasets.ImageFolder):
    """
    Custom ImageFolder that uses folder names (0, 1, ..., 999) as the actual integer labels,
    avoiding the alphabetical sorting mismatch.
    """
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort(key=lambda x: int(x) if x.isdigit() else x)
        class_to_idx = {cls_name: int(cls_name) if cls_name.isdigit() else i 
                        for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def load_imagenet_subset(data_dir, num_images=1000):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    try:
        dataset = ImageNetFolder(root=data_dir, transform=transform)
        # Lấy random subset
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        return subset
    except Exception as e:
        print(f"Failed to load ImageNetFolder: {e}")
        return None

# %% [markdown]
# ## 0.3 Load models
# Load các model Conventional, Adversarially trained, và Defended.

# %%
class NormalizedModel(torch.nn.Module):
    """
    Wrapper để chuẩn hóa ảnh theo chuẩn ImageNet ngay bên trong model.
    Giúp cho thuật toán tối ưu trực tiếp trên dải pixel [0, 1].
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
        if model_name == 'at_resnet50':
            model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf')
        elif model_name == 'at_wideresnet':
            model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')
        else:
            raise ValueError(f"Unknown RobustBench model {model_name}")
    elif model_name == 'patchguard':
        import sys
        # Tìm thư mục PatchGuard linh hoạt hơn (hỗ trợ cả Colab và chạy local)
        possible_paths = [
            os.path.join(os.getcwd(), 'PatchGuard'),
            os.path.join(os.path.dirname(os.getcwd()), 'PatchGuard'),
            '/content/PatchGuard'
        ]
        patchguard_dir = None
        for p in possible_paths:
            if os.path.exists(p):
                patchguard_dir = p
                break
        
        if patchguard_dir is None:
            patchguard_dir = os.path.join(os.getcwd(), 'PatchGuard') # Mặc định nếu không thấy

        if patchguard_dir not in sys.path:
            sys.path.insert(0, patchguard_dir)
        import importlib.util
        try:
            # Load nets.bagnet manually to avoid naming conflicts with local utils.py
            bagnet_py = os.path.join(patchguard_dir, 'nets', 'bagnet.py')
            spec_bagnet = importlib.util.spec_from_file_location("pg_nets_bagnet", bagnet_py)
            pg_nets_bagnet = importlib.util.module_from_spec(spec_bagnet)
            sys.modules["nets"] = pg_nets_bagnet 
            sys.modules["nets.bagnet"] = pg_nets_bagnet
            spec_bagnet.loader.exec_module(pg_nets_bagnet)
            nets_bagnet = pg_nets_bagnet
            
            # Load utils.defense_utils manually
            defense_py = os.path.join(patchguard_dir, 'utils', 'defense_utils.py')
            spec_defense = importlib.util.spec_from_file_location("pg_utils_defense", defense_py)
            pg_utils_defense = importlib.util.module_from_spec(spec_defense)
            spec_defense.loader.exec_module(pg_utils_defense)
            masking_defense = pg_utils_defense.masking_defense
        except Exception as e:
            raise ValueError(f"PatchGuard repo loading failed: {e}. Please ensure you cloned it correctly.")
        
        checkpoint_path = os.path.join(patchguard_dir, 'checkpoints', 'bagnet17_net.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"PatchGuard checkpoint missing at {checkpoint_path}. Please download it.")
            
        base_model = nets_bagnet.bagnet17(pretrained=True, clip_range=None, aggregation='none')
        base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['state_dict'])
        
        class PatchGuardWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.window_size = 6
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

# %% [markdown]
# # Phase 1 — Core algorithm (Algorithm 1)
# ## 1.1 Patch initialization (Section 3.2)
# Mỗi patch chồng N=100 circles lên black image. Tọa độ, bán kính, RGB, Transparency $\in [0,1]$

# %%
import torch.nn as nn

class PatchRenderer(nn.Module):
    def __init__(self, patch_size, num_circles=100, device='cpu'):
        super().__init__()
        self.patch_size = patch_size
        self.num_circles = num_circles
        self.device = device
        
        y, x = torch.meshgrid(torch.arange(patch_size, device=device), 
                              torch.arange(patch_size, device=device), indexing='ij')
        self.grid_x = x.float()
        self.grid_y = y.float()

    def forward(self, params):
        patch = torch.zeros((3, self.patch_size, self.patch_size), device=self.device)
        cx = params[:, 0] * self.patch_size
        cy = params[:, 1] * self.patch_size
        r = params[:, 2] * (self.patch_size / 2) 
        
        R = params[:, 3]
        G = params[:, 4]
        B = params[:, 5]
        alpha = params[:, 6]
        
        for i in range(self.num_circles):
            dist_sq = (self.grid_x - cx[i])**2 + (self.grid_y - cy[i])**2
            mask = (dist_sq <= r[i]**2).float()
            
            a = alpha[i] * mask
            color = torch.stack([R[i], G[i], B[i]]).view(3, 1, 1)
            patch = color * a + patch * (1 - a)
            
        return patch

# %% [markdown]
# ## 1.2 Patch optimization, 1.3 Location optimization & 1.4 Loss functions
# Thuật toán (1+1)-ES và Fast Simulated Annealing

# %%
class CamoPatch:
    def __init__(self, model, patch_size=40, num_circles=100, sigma=0.1, t_init=300, l_i=4, device='cpu'):
        self.model = model
        self.patch_size = patch_size
        self.num_circles = num_circles
        self.sigma = sigma
        self.t_init = t_init
        self.l_i = l_i
        self.device = device
        self.renderer = PatchRenderer(patch_size, num_circles, device)

    def _margin_loss(self, logits, y):
        # 1.4 Loss functions: Non-targeted margin loss (âm khi misclassify)
        logits_y = logits[0, y]
        logits_other = logits.clone()
        logits_other[0, y] = -float('inf')
        max_other = logits_other.max()
        return (logits_y - max_other).item()

    def attack(self, image, label, max_queries=10000):
        _, h, w = image.shape
        
        # 1.1 Khởi tạo random
        params = torch.rand((self.num_circles, 7), device=self.device)
        loc_x = np.random.randint(0, w - self.patch_size + 1)
        loc_y = np.random.randint(0, h - self.patch_size + 1)
        
        best_params = params.clone()
        best_loc_x, best_loc_y = loc_x, loc_y
        
        def apply_patch(img, patch, lx, ly):
            adv_img = img.clone()
            adv_img[:, ly:ly+self.patch_size, lx:lx+self.patch_size] = patch
            return adv_img

        def eval_solution(p, lx, ly):
            patch = self.renderer(p)
            adv_img = apply_patch(image, patch, lx, ly)
            with torch.no_grad():
                logits = self.model(adv_img.unsqueeze(0))
            loss = self._margin_loss(logits, label)
            
            orig_region = image[:, ly:ly+self.patch_size, lx:lx+self.patch_size]
            l2_norm = torch.norm(patch - orig_region, p=2).item()
            return loss, l2_norm, adv_img

        current_loss, current_l2, _ = eval_solution(best_params, best_loc_x, best_loc_y)
        
        for k in range(1, max_queries + 1):
            if k % (self.l_i + 1) == 0:
                # 1.3 Location optimization (Fast Simulated Annealing)
                new_lx = np.random.randint(0, w - self.patch_size + 1)
                new_ly = np.random.randint(0, h - self.patch_size + 1)
                
                new_loss, new_l2, _ = eval_solution(best_params, new_lx, new_ly)
                
                accept = False
                if new_loss < 0 and current_loss < 0:
                    if new_l2 < current_l2:
                        accept = True
                    else:
                        t_curr = self.t_init / k
                        d = new_l2 - current_l2
                        if np.random.rand() < np.exp(-d / t_curr):
                            accept = True
                elif new_loss < current_loss:
                    accept = True
                
                if accept:
                    best_loc_x, best_loc_y = new_lx, new_ly
                    current_loss, current_l2 = new_loss, new_l2
            else:
                # 1.2 Patch optimization ((1+1)-ES)
                noise = torch.randn_like(best_params) * self.sigma
                new_params = torch.clamp(best_params + noise, 0, 1)
                
                new_loss, new_l2, _ = eval_solution(new_params, best_loc_x, best_loc_y)
                
                accept = False
                if new_loss < 0 and current_loss < 0:
                    if new_l2 < current_l2:
                        accept = True
                elif new_loss < current_loss:
                    accept = True
                    
                if accept:
                    best_params = new_params.clone()
                    current_loss, current_l2 = new_loss, new_l2
                    
        final_patch = self.renderer(best_params)
        final_adv_img = apply_patch(image, final_patch, best_loc_x, best_loc_y)
        return final_adv_img, current_loss < 0, current_l2

# %% [markdown]
# # Phase 2 — Hyperparameters & experimental setup
# ## 2.1 Hyperparameters & 2.2 Evaluation metrics
# K = 10,000, sigma=0.1, N=100, t=300, li=4. Tracking metrics: Accuracy, L2 distance, NNR.

# %%
def compute_nnr(original_img, adv_img, patch_size):
    """
    Non-normalised Residual (NNR)
    """
    diff = torch.abs(original_img - adv_img)
    return diff.sum().item() / (3 * patch_size * patch_size)

def evaluate_camopatch(data_dir, model_name='resnet50', num_images=10, max_queries=10000, 
                       patch_size=40, num_circles=100, sigma=0.1, t_init=300, l_i=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_name}...")
    raw_model = get_model(model_name, device=device)
    model = NormalizedModel(raw_model).to(device)
    model.eval()

    print(f"Loading dataset from {data_dir}...")
    dataset = load_imagenet_subset(data_dir, num_images=num_images)
    if dataset is None:
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    attacker = CamoPatch(model, patch_size=patch_size, num_circles=num_circles, 
                         sigma=sigma, t_init=t_init, l_i=l_i, device=device)

    total_images = 0
    successful_attacks = 0
    total_l2 = 0.0
    total_nnr = 0.0

    for i, (img, label) in enumerate(dataloader):
        img = img[0].to(device)
        label = label.item()
        
        with torch.no_grad():
            orig_logits = model(img.unsqueeze(0))
            if orig_logits.argmax(dim=1).item() != label:
                continue # Skip misclassified images
            
        total_images += 1
        adv_img, success, l2_dist = attacker.attack(img, label, max_queries=max_queries)
        
        if success:
            successful_attacks += 1
            total_l2 += l2_dist
            total_nnr += compute_nnr(img, adv_img, patch_size)
            print(f"Image {i}: Success. L2={l2_dist:.4f}")
        else:
            print(f"Image {i}: Failed.")

    asr = successful_attacks / total_images if total_images > 0 else 0
    avg_l2 = total_l2 / successful_attacks if successful_attacks > 0 else 0
    avg_nnr = total_nnr / successful_attacks if successful_attacks > 0 else 0

    print("="*30)
    print(f"RESULTS for {model_name}")
    print(f"ASR: {asr:.2%} | Avg L2: {avg_l2:.4f} | Avg NNR: {avg_nnr:.4f}")
    return asr, avg_l2, avg_nnr

# %% [markdown]
# # Phase 3 — Baselines & comparison
# ## 3.1 Implement baseline attacks & 3.2 Statistical testing
# Tích hợp mã nguồn của Patch-RS, TPA, Adv-Watermark, LOAP. So sánh dùng Wilcoxon signed-rank test.

# %%
import scipy.stats as stats

def wilcoxon_test(results_camopatch, results_baseline):
    """
    Thực hiện kiểm định Wilcoxon signed-rank (mức ý nghĩa 5%).
    Input: list of L2 distances (hoặc ASR) của 10 lần runs.
    """
    stat, p_value = stats.wilcoxon(results_camopatch, results_baseline)
    print(f"Wilcoxon test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Sự khác biệt CÓ ý nghĩa thống kê (p < 0.05).")
    else:
        print("Sự khác biệt KHÔNG CÓ ý nghĩa thống kê (p >= 0.05).")
    return p_value

# %% [markdown]
# # Phase 4 — Ablation study
# ## 4.1 Ablation experiments
# Đánh giá ảnh hưởng của các tham số: N, sigma, l_i, t.

# %%
def run_ablation_study(data_dir, model_name='resnet50', base_queries=10000):
    """
    Khung chạy Ablation study để đánh giá sự biến thiên của thuật toán theo từng hyperparam.
    """
    print("--- Phase 4: ABLATION STUDY ---")
    
    # 4.1.1: Ảnh hưởng của N (số circles)
    N_list = [10, 50, 100, 200]
    for n in N_list:
        print(f"\n[Ablation] Testing N = {n}")
        # Bỏ comment để chạy:
        # evaluate_camopatch(data_dir, model_name=model_name, num_circles=n, max_queries=base_queries)
        
    # 4.1.2: Ảnh hưởng của Sigma (step-size)
    sigma_list = [0.01, 0.05, 0.1, 0.5]
    for sig in sigma_list:
        print(f"\n[Ablation] Testing sigma = {sig}")
        # Bỏ comment để chạy:
        # evaluate_camopatch(data_dir, model_name=model_name, sigma=sig, max_queries=base_queries)
        
    # 4.1.3: Ảnh hưởng của l_i (Location schedule)
    li_list = [1, 4, 9, 19]
    for li in li_list:
        print(f"\n[Ablation] Testing l_i = {li}")
        # Bỏ comment để chạy:
        # evaluate_camopatch(data_dir, model_name=model_name, l_i=li, max_queries=base_queries)
        
    # 4.1.4: Ảnh hưởng của t (Temperature)
    t_list = [10, 100, 300, 1000]
    for temp in t_list:
        print(f"\n[Ablation] Testing t = {temp}")
        # Bỏ comment để chạy:
        # evaluate_camopatch(data_dir, model_name=model_name, t_init=temp, max_queries=base_queries)

# %%
# Gọi các hàm chạy ở đây. Ví dụ:
if __name__ == "__main__":
    # Thay đường dẫn sau bằng thư mục ImageNet validation của bạn
    DATA_DIR = "imagenet_val_1000"
    
    # Tải dataset nếu chưa có
    # download_imagenet_subset(output_dir=DATA_DIR, num_images=1000)
    
    # Test chạy đánh giá chuẩn (Phase 2)
    # evaluate_camopatch(data_dir=DATA_DIR, model_name='resnet50', num_images=10)
    
    # Chạy Ablation Study (Phase 4)
    # run_ablation_study(data_dir=DATA_DIR)
