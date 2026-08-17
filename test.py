
import argparse
from pathlib import Path
import os
import glob
import torch
import torch.nn as nn
import torchvision
from torchdiffeq import odeint_adjoint as odeint
from diffusers.models import AutoencoderKL
from models import get_flow_model
from datasets_prep import get_corruption_dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
                             
from make_corruptions.obs_corruption import gaussian_noise, shot_noise, impulse_noise, defocus_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression,saturate,occlusion,light_out

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class WrapperCondFlow(nn.Module):
    """"""
    def __init__(self, model, cond, action):
        super().__init__()
        self.model = model
        self.cond = cond
        self.action = action

    def forward(self, t, x):
                
        x_cond = torch.cat([x, self.cond], dim=1)                             
        
                         
        if t.dim() == 0:          
            t = t.unsqueeze(0).expand(x.size(0))           
        elif t.size(0) != x.size(0):             
            t = t.expand(x.size(0))

        self.action = self.action.long()
        
        return self.model(t, x_cond, y=self.action)

def corrupt_obs(obs, corruption_type, corruption_severity):
    if corruption_type is None:
        return obs
    if corruption_type == "gaussian_noise":
        obs["rgb"] = gaussian_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "shot_noise":
        obs["rgb"] = shot_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "impulse_noise":
        obs["rgb"] = impulse_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "defocus_blur":
        obs["rgb"] = defocus_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "motion_blur":
        obs["rgb"] = motion_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "zoom_blur":
        obs["rgb"] = zoom_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "snow":
        obs["rgb"] = snow(obs["rgb"], corruption_severity)
    elif corruption_type == "frost":
        obs["rgb"] = frost(obs["rgb"], corruption_severity)
    elif corruption_type == "fog":
        obs["rgb"] = fog(obs["rgb"], corruption_severity)
    elif corruption_type == "brightness":
        obs["rgb"] = brightness(obs["rgb"], corruption_severity)
    elif corruption_type == "contrast":
        obs["rgb"] = contrast(obs["rgb"], corruption_severity)
    elif corruption_type == "elastic_transform":
        obs["rgb"] = elastic_transform(obs["rgb"], corruption_severity)
    elif corruption_type == "pixelate":
        obs["rgb"] = pixelate(obs["rgb"], corruption_severity)
    elif corruption_type == "jpeg_compression":
        obs["rgb"] = jpeg_compression(obs["rgb"], corruption_severity)
    elif corruption_type == "saturate":
        obs["rgb"] = saturate(obs["rgb"], corruption_severity)
    elif corruption_type == "occlusion":
        obs["rgb"] = occlusion(obs["rgb"], corruption_severity)
    elif corruption_type == "light_out":
        obs["rgb"] = light_out(obs["rgb"], corruption_severity)
    obs["rgb"] = obs["rgb"].astype(np.uint8)
    return obs

def sample_from_model(model, z_0, args):
    """"""
                         
    t = torch.tensor([1.0, 0.0], device=args.device)
    
                  
    with torch.no_grad():
        fake_sample = odeint(
            model, 
            z_0, 
            t, 
            method=args.method,
            atol=args.atol,
            rtol=args.rtol
        )
    
    return fake_sample[-1]              

def denoise_from_noisy(model, noisy_latent, condition_input, device, action=None, num_steps=20):
    """latentEulert=0"""
    wrapper_model = WrapperCondFlow(model, condition_input, action=action)
    
    with torch.no_grad():
        x = noisy_latent.clone()
        dt = -1.0 / num_steps                 
        for step in range(num_steps):
            t = torch.tensor([1.0 - (step * abs(dt))], device=device)          
            v_pred = wrapper_model(t, x)
            x = x + dt * v_pred                 
        return x
class GaussianActionMemory:
    def __init__(self, feature_dim, num_actions=7, device='cpu'):
        self.feature_dim = feature_dim
        self.num_action = num_actions
        self.device = device
        self.means = {action_id: torch.zeros(feature_dim, device=self.device) for action_id in [1,2,3,4,5,6]}
        self.vars = {action_id: torch.ones(feature_dim, device=self.device) for action_id in [1,2,3,4,5,6]} 
        self.counts = {action_id: 0 for action_id in [1,2,3,4,5,6]}

    def update(self, features, actions):
        for feature, action in zip(features, actions):
            action_id = int(action)
            if action_id not in self.means: continue 

            feature = feature.to(self.device)

            self.counts[action_id] += 1
            n = self.counts[action_id]
            
                         
            old_mean = self.means[action_id]
            self.means[action_id] = old_mean + (feature - old_mean) / n

                                                                     
            if n > 1:
                old_var = self.vars[action_id]
                delta = feature - old_mean
                delta2 = feature - self.means[action_id]
                self.vars[action_id] = old_var + (delta * delta2) / n
    
    def compute_distance(self, feature, action_id):
        """Mahalanobis distance"""
        action_id = int(action_id)
        if action_id not in self.means or self.counts[action_id] < 2:
            return 0.0                
        
        mean = self.means[action_id]
        var = self.vars[action_id]
        feature = feature.to(self.device)
        
                           
        feature_flat = feature.flatten()
        
                                                 
        diff = feature_flat - mean
              
        var_safe = torch.clamp(var, min=1e-6)
        distance = torch.sqrt(torch.mean(diff**2 / var_safe))
        
        raw_distance = distance.item()
        
        return raw_distance

    def compute_cosine_similarity(self, feature, action_id):
        """"""
        action_id = int(action_id)
        if action_id not in self.means or self.counts[action_id] < 2:
            return 0.0                
        
        mean = self.means[action_id]
        feature = feature.to(self.device)
        
                           
        feature_flat = feature.flatten()
        mean_flat = mean.flatten()
        
                                                           
        inner_product = torch.dot(feature_flat, mean_flat)
        feature_norm = torch.norm(feature_flat)
        mean_norm = torch.norm(mean_flat)
        
              
        if feature_norm == 0 or mean_norm == 0:
            return 0.0
            
        cosine_similarity = inner_product / (feature_norm * mean_norm)
        
        return cosine_similarity.item()

    def get_state(self):
        return {
            'means': self.means,
            'vars': self.vars,  
            'counts': self.counts
        }
    
    def load_state(self, state_dict):
        self.means = state_dict['means']
        self.vars = state_dict['vars']
        self.counts = state_dict['counts']
        for action_id in self.means:
            self.means[action_id] = self.means[action_id].to(self.device)
            self.vars[action_id] = self.vars[action_id].to(self.device)

class ActionRemappingDataset:
    """
    ConcatDataset
    """
    def __init__(self, concat_dataset, dataset_names):
        self.concat_dataset = concat_dataset
        self.dataset_names = dataset_names
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
                
        item = self.concat_dataset[idx]
        
                     
        dataset_idx = 0
        dataset_idx = idx % len(self.dataset_names)
        
                       
        if isinstance(item, dict):
            item = item.copy()
        
               
        dataset_name = self.dataset_names[dataset_idx]
        if 'prev_action' in item:
            item['prev_action'] = remap_actions(item['prev_action'], dataset_name)
        
        return item

def remap_actions(actions, dataset_name):

    if isinstance(actions, torch.Tensor):
        remapped = actions.clone()
    else:
        remapped = torch.tensor(actions, dtype=torch.long)
    
    if 'R2R' in dataset_name:
                                               
        pass         
    elif 'RxR' in dataset_name:
                              
        remapped = torch.where(remapped == 2, torch.tensor(4), remapped)              
        remapped = torch.where(remapped == 3, torch.tensor(5), remapped)              
    
    return remapped

def generate_images(args):
    """"""
    
    autoencoder = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(args.device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    
                      
    print("...")
    content = torch.load(args.content_path, map_location=args.device, weights_only=False)
    
                   
    saved_args = content["args"]
    args.num_classes = saved_args.num_classes
    args.num_in_channels = saved_args.num_in_channels
    args.num_out_channels = saved_args.num_out_channels
    args.nf = saved_args.nf
    args.ch_mult = saved_args.ch_mult
    args.num_res_blocks = saved_args.num_res_blocks
    args.attn_resolutions = saved_args.attn_resolutions
    args.dropout = saved_args.dropout
    args.num_heads = saved_args.num_heads
    args.num_head_channels = saved_args.num_head_channels
    args.num_head_upsample = saved_args.num_head_upsample
    args.use_scale_shift_norm = saved_args.use_scale_shift_norm
    args.resblock_updown = saved_args.resblock_updown
    args.use_new_attention_order = saved_args.use_new_attention_order
    
           
    print("Flow...")
    model = get_flow_model(args).to(args.device)
    
            
    print("...")
    model_state = torch.load(args.model_path, map_location=args.device)
    
                                 
    if any(key.startswith('module.') for key in model_state.keys()):
        new_state = {}
        for key, value in model_state.items():
            new_key = key.replace('module.', '')
            new_state[new_key] = value
        model_state = new_state
    
    model.load_state_dict(model_state)
    model.eval()

    latent_size = args.image_size // 8
    feature_dim = args.num_out_channels * latent_size * latent_size
    memory_bank = GaussianActionMemory(feature_dim=feature_dim, device=args.device)
    memory_bank.load_state(content["memory_bank_state"])
    
           
    dataset = get_corruption_dataset(args)
    dataset_name = "R2R" if "R2R" in args.image_path else "RxR"
    dataset = ActionRemappingDataset(dataset, [dataset_name])

             
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,                  
        shuffle=False,             
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    
            
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"...")
    
                
    prev_latent_corrupted = None
    prev_latent_gt = None
    

    for frame_idx, batch in enumerate(tqdm(data_loader, desc="")):
        print(f"\n {frame_idx + 1} ")
        
                          
        current_real_image = batch['current_image'].to(args.device, non_blocking=True)
        action = batch['prev_action']
        action = action.long().to(args.device)
        
        
                               
        current_real_np = current_real_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        current_real_np = ((current_real_np + 1.0) / 2.0 * 255).astype(np.uint8)
         
                               
        current_obs_dict = {'rgb': current_real_np}
        current_corrupted_obs = corrupt_obs(current_obs_dict, args.corruption_type, args.corruption_severity)
       
                                 
        current_corrupted = current_corrupted_obs['rgb'].astype(np.float32) / 255.0
        current_corrupted = torch.from_numpy(current_corrupted).permute(2, 0, 1).unsqueeze(0).to(args.device)
        current_corrupted = current_corrupted * 2.0 - 1.0               
        
        
                     
        with torch.no_grad():
            current_latent_corrupted = autoencoder.encode(current_corrupted).latent_dist.sample().mul_(args.scale_factor)
            
        if frame_idx == 0:
                                  
            c = torch.zeros_like(current_latent_corrupted)
            condition_input = torch.cat([current_latent_corrupted, c], dim=1)
            
                              
            training_label = torch.full((1,), 0, device=args.device, dtype=torch.long)
            
        else:
                                                               
            c_zeros = torch.zeros_like(prev_latent_corrupted)
            condition_input_0 = torch.cat([current_latent_corrupted, c_zeros], dim=1)
            training_label_0 = torch.full((1,), 0, device=args.device, dtype=torch.long)
            

                                                                            
            pre_diff = prev_latent_gt - prev_latent_corrupted                    
            condition_input_1 = torch.cat([current_latent_corrupted, pre_diff], dim=1)
            
        
                      
        z_0 = torch.randn(1, 4, args.image_size // 8, args.image_size // 8).to(args.device)
        
                         
        # t_z = 0.99
        
        if frame_idx == 0:
                           
                                                      
            noisy_start = (1 - args.t_z) * current_latent_corrupted + args.t_z * z_0
            training_label = torch.full((1,), 0, device=args.device, dtype=torch.long)
            
                                      
            with torch.no_grad():
                fake_latent = denoise_from_noisy(model, noisy_start, condition_input, args.device, training_label, num_steps=20)
                generated_frame = autoencoder.decode(fake_latent / args.scale_factor).sample
                generated_frame = torch.clamp(generated_frame, -1.0, 1.0)
                                    
                                                       
        else:
                              
                           
            noisy_start = (1 - args.t_z) * current_latent_corrupted + args.t_z * z_0
            
                             
            training_label_0 = torch.full((1,), 0, device=args.device, dtype=torch.long)
                                      
            with torch.no_grad():
                fake_latent_0 = denoise_from_noisy(model, noisy_start, condition_input_0, args.device, training_label_0, num_steps=20)
                generated_frame_0 = autoencoder.decode(fake_latent_0 / args.scale_factor).sample
                generated_frame_0 = torch.clamp(generated_frame_0, -1.0, 1.0)
                                    
                                                       
    
                             
            with torch.no_grad():
                fake_latent_1 = denoise_from_noisy(model, noisy_start, condition_input_1, args.device, action, num_steps=20)
                generated_frame_1 = autoencoder.decode(fake_latent_1 / args.scale_factor).sample
                generated_frame_1 = torch.clamp(generated_frame_1, -1.0, 1.0)
            
                                       
            dist_0 = memory_bank.compute_distance(prev_latent_gt-fake_latent_0, action)
            dist_1 = memory_bank.compute_distance(prev_latent_gt-fake_latent_1, action)
            print(f"{frame_idx + 1} - : 0={dist_0:.4f}, 1={dist_1:.4f}")
            # theta = 0.25
            if dist_0 < args.theta or dist_0<=dist_1:
                selected_latent = fake_latent_0
                selected_frame = generated_frame_0
                print(f"  -> 0")
            else:
                weight_0 = dist_1 / (dist_0 + dist_1) +0.1
                weight_1 = 0.9 - weight_0
                selected_latent = weight_0* fake_latent_1 + weight_1* fake_latent_0
                selected_frame = autoencoder.decode(selected_latent / args.scale_factor).sample
                selected_frame = torch.clamp(selected_frame, -1.0, 1.0)
                print(f"  -> 1")
        
                
        if frame_idx == 0:
                                             
            display_images = [
                torch.clamp((current_corrupted + 1.0) / 2.0, 0.0, 1.0).squeeze(0),                 
                                                                                           
                torch.clamp((generated_frame + 1.0) / 2.0, 0.0, 1.0).squeeze(0)       
            ]
            grid_image = torchvision.utils.make_grid(display_images, nrow=2, normalize=True)
        else:
                                                                             
            display_images = [
                torch.clamp((current_corrupted + 1.0) / 2.0, 0.0, 1.0).squeeze(0),                 
                                                                                               
                                                                                                    
                                                                                                    
                torch.clamp((selected_frame + 1.0) / 2.0, 0.0, 1.0).squeeze(0)         
            ]
            grid_image = torchvision.utils.make_grid(display_images, nrow=2, normalize=True)
        
                
        frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.png")
        torchvision.utils.save_image(grid_image, frame_path, normalize=True)
        
                 
        prev_latent_corrupted = current_latent_corrupted
        if frame_idx == 0:
            prev_latent_gt = fake_latent
        else:
                                            
            prev_latent_gt = selected_latent

    print(f"\n! : {args.output_dir}")
    print(f" {len(dataset)} ")


def create_gif_from_frames(input_folder=".", output_filename="animation.gif", duration=100):
                     
    pattern = os.path.join(input_folder, "frame_*.png")
    image_files = glob.glob(pattern)
    image_files.sort()          
    
    if not image_files:
        print("PNG")
        return
    
    print(f" {len(image_files)} ")
    
            
    images = []
    for file_path in image_files:
        img = Image.open(file_path)
        images.append(img)

    if not images:
        print("")
        return
    
            
    try:
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0           
        )
        print(f"GIF: {output_filename}")
        print(f": {len(images)}")
        print(f": {duration}ms")
        
    except Exception as e:
        print(f"GIF: {e}")

def main():
    parser = argparse.ArgumentParser(description="corruption flow")
    
              
    parser.add_argument("--seed", type=int, default=42, help="")
    
            
    parser.add_argument("--corruption_type","--c", type=str, default="jpeg_compression", help="corruption")
    parser.add_argument("--corruption_severity", type=int, default=4, help="corruption")
    parser.add_argument("--layout", action="store_true")
                                                                                                                                                                            
    parser.add_argument("--model_path", type=str, default=str(Path(__file__).parent / "checkpoints" / "checkpoint.pth"))
    
    parser.add_argument("--content_path", type=str, default=str(Path(__file__).parent / "checkpoints" / "content.pth"))
                                                                                                                                                                              
    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse", help="")
    
             
    parser.add_argument("--image_path", type=str, default=str(Path(__file__).parent / "example" / "R2R_VLNCE_v1-3_preprocessed"))
    parser.add_argument("--action_path", type=str, default=str(Path(__file__).parent / "example" / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen_gt.json.gz"))
    parser.add_argument("--keep_aspect_ratio", action="store_true", default=True, help="")
    parser.add_argument("--shuffle_data", action="store_true", default=False, help="")
    
                        
    parser.add_argument("--image_size", type=int, default=224, help="")
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="")
    parser.add_argument("--num_in_channels", type=int, default=12, help=" (4 + 8)")
    parser.add_argument("--num_out_channels", type=int, default=4, help="")
    parser.add_argument("--nf", type=int, default=256, help="")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="")
    parser.add_argument("--attn_resolutions", nargs="+", type=int, default=[16, 8], help="")
    parser.add_argument("--ch_mult", nargs="+", type=int, default=[1, 2, 4], help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
                                
    parser.add_argument("--num_classes", type=int, default=2, help="(0: first_frame condition, 1: prev_gt condition)")
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True, help="scale shift norm")
    parser.add_argument("--resblock_updown", action="store_true", default=False, help="")
    parser.add_argument("--use_new_attention_order", action="store_true", default=False, help="")
    parser.add_argument("--resamp_with_conv", action="store_true", default=True, help="")
    parser.add_argument("--centered", action="store_true", default=True, help="")
    
            
    parser.add_argument("--batch_size", type=int, default=4, help="1")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "result"))
    parser.add_argument("--save_grid", action="store_true", default=False, help="")
    parser.add_argument("--save_comparison", action="store_true", default=True, help="")
    parser.add_argument("--theta",type=float,default=0.25)
    parser.add_argument("--t_z",type=float,default=0.99)
           
    parser.add_argument("--method", type=str, default="dopri5", help="ODE")
    parser.add_argument("--atol", type=float, default=1e-5, help="")
    parser.add_argument("--rtol", type=float, default=1e-5, help="")
    
          
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="")
    
    args = parser.parse_args()
    
            
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
          
    generate_images(args)
    create_gif_from_frames("result")

                             
                                                                               
                                                                                                
                      
        

if __name__ == "__main__":
    main()
