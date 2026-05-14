import torch
import torch.nn as nn
import numpy as np

class PatchRenderer(nn.Module):
    def __init__(self, patch_size, num_circles=100, device='cpu'):
        super().__init__()
        self.patch_size = patch_size
        self.num_circles = num_circles
        self.device = device
        
        # Grid of coordinates for rendering
        y, x = torch.meshgrid(torch.arange(patch_size, device=device), 
                              torch.arange(patch_size, device=device), indexing='ij')
        self.grid_x = x.float()
        self.grid_y = y.float()

    def forward(self, params):
        # params shape: (num_circles, 7)
        # 7 parameters: cx, cy, r, R, G, B, alpha (transparency)
        # all values in [0, 1]
        
        patch = torch.zeros((3, self.patch_size, self.patch_size), device=self.device)
        
        # Scale parameters to patch dimensions
        cx = params[:, 0] * self.patch_size
        cy = params[:, 1] * self.patch_size
        r = params[:, 2] * (self.patch_size / 2) # max radius is half patch size
        
        R = params[:, 3]
        G = params[:, 4]
        B = params[:, 5]
        alpha = params[:, 6]
        
        # We blend circles one by one. In the paper, they overlay circles.
        for i in range(self.num_circles):
            dist_sq = (self.grid_x - cx[i])**2 + (self.grid_y - cy[i])**2
            mask = (dist_sq <= r[i]**2).float()
            
            # Alpha blending: 
            # new_color = circle_color * alpha + old_color * (1 - alpha)
            # Only where mask == 1
            a = alpha[i] * mask
            
            color = torch.stack([R[i], G[i], B[i]]).view(3, 1, 1)
            patch = color * a + patch * (1 - a)
            
        return patch

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
        # Non-targeted: L = f_y - \max_{y_q \neq y} f_{y_q}
        # Negative loss means misclassification
        logits_y = logits[0, y]
        logits_other = logits.clone()
        logits_other[0, y] = -float('inf')
        max_other = logits_other.max()
        return (logits_y - max_other).item()

    def attack(self, image, label, max_queries=10000):
        """
        image: tensor of shape (3, H, W) in [0, 1]
        label: true label
        """
        _, h, w = image.shape
        
        # 1. Initialize patch parameters
        params = torch.rand((self.num_circles, 7), device=self.device)
        
        # 2. Initialize location
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
            
            # L2 norm between patch and original image region
            orig_region = image[:, ly:ly+self.patch_size, lx:lx+self.patch_size]
            l2_norm = torch.norm(patch - orig_region, p=2).item()
            return loss, l2_norm, adv_img

        current_loss, current_l2, _ = eval_solution(best_params, best_loc_x, best_loc_y)
        
        for k in range(1, max_queries + 1):
            if k % (self.l_i + 1) == 0:
                # Location Update (Simulated Annealing)
                new_lx = np.random.randint(0, w - self.patch_size + 1)
                new_ly = np.random.randint(0, h - self.patch_size + 1)
                
                new_loss, new_l2, _ = eval_solution(best_params, new_lx, new_ly)
                
                # Accept if constraint met and L2 better, OR by SA probability
                accept = False
                if new_loss < 0 and current_loss < 0:
                    if new_l2 < current_l2:
                        accept = True
                    else:
                        t_curr = self.t_init / k
                        d = new_l2 - current_l2 # minimize L2
                        if np.random.rand() < np.exp(-d / t_curr):
                            accept = True
                elif new_loss < current_loss:
                    accept = True
                
                if accept:
                    best_loc_x, best_loc_y = new_lx, new_ly
                    current_loss, current_l2 = new_loss, new_l2
            else:
                # Patch Update (1+1)-ES
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

