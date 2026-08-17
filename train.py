import argparse
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep import get_corruption_dataset
from models import get_flow_model
from omegaconf import OmegaConf
from torch.multiprocessing import Process
from torchdiffeq import odeint_adjoint as odeint
import wandb
from tqdm import tqdm
import torch.multiprocessing
from glob import glob
from PIL import Image
import copy
from torch.utils.data import ConcatDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ActionRemappingDataset:
    """
    ConcatDataset
    """
    def __init__(self, concat_dataset, dataset_names, dataset_lengths):
        self.concat_dataset = concat_dataset
        self.dataset_names = dataset_names
        self.dataset_lengths = dataset_lengths
        
                            
        self.cumulative_lengths = []
        cumsum = 0
        for length in dataset_lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
                
        item = self.concat_dataset[idx]
        
                     
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_lengths):
            if idx < cumsum:
                dataset_idx = i
                break
        
                       
        if isinstance(item, dict):
            item = item.copy()
        
               
        dataset_name = self.dataset_names[dataset_idx]
        if 'prev_action' in item:
            item['prev_action'] = remap_actions(item['prev_action'], dataset_name)
        
        return item

class WrapperCondFlow(nn.Module):
    def __init__(self, model, cond, action=None) -> None:
        super().__init__()
        self.model = model
        self.cond = cond
        self.action = action

    def forward(self, t, x):
                                   
        x = torch.cat([x, self.cond], dim=1)
        if self.action is not None:
            return self.model(t, x, y=self.action)
        else:
            return self.model(t, x)

def remap_actions(actions, dataset_name):
    """
    ID
    :
    - 0: / ()
    - 1:  ()
    - 2: R2R15
    - 3: R2R15
    - 4: RxR30  
    - 5: RxR30
    """
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

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def sample_from_model(model, z_0, device="cuda"):
    """"""
    t = torch.tensor([1.0, 0.0], device=device)
    fake_image = odeint(model, z_0, t, atol=1e-8, rtol=1e-8)
    return fake_image

def denoise_from_noisy(model, noisy_latent, condition_input, device="cuda", action=None, num_steps=10):
    """cur_aug latentEulert=0"""
    wrapper_model = WrapperCondFlow(model, condition_input, action=action)
    
    with torch.no_grad():
        x = noisy_latent.clone()
        dt = -1.0 / num_steps                 
        for step in range(num_steps):
            t = torch.tensor([1.0 - (step * abs(dt))], device=device)          
            v_pred = wrapper_model(t, x)
            x = x + dt * v_pred                 
        return x

def train(rank, gpu, args):
    """
    
    1. condition[cur_img_latent_aug, 0, action_token]
    2. prev_predictcondition[cur_img_latent_aug, pre_gt-pre_aug, action_token]
    """
                             
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
                        
    if rank == 0:
        wandb.init(
            project="flow_latent_corruption",
            name=f"{args.exp}",
            notes="Flow matching for latent corruption task",
            config=vars(args)
        )
    
    from diffusers.models import AutoencoderKL

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device("cuda:{}".format(gpu))

    batch_size = args.batch_size

                                                                       
    if rank == 0:
        print("Loading PixMix datasets...")
    mixing_set1 = glob(os.path.join(args.pixmix_fractal_path, '*'))
    mixing_set2 = glob(os.path.join(args.pixmix_vis_path, '*'))
    if not mixing_set1 or not mixing_set2:
        raise FileNotFoundError("PixMix datasets not found. Please check --pixmix_fractal_path and --pixmix_vis_path.")
    
                                                                    
    mixing_images1 = [Image.open(p).convert("RGB").resize((224, 224)) for p in mixing_set1]
    mixing_images2 = [Image.open(p).convert("RGB").resize((224, 224)) for p in mixing_set2]
    if rank == 0:
        print(f"Loaded {len(mixing_images1)} images from fractal set and {len(mixing_images2)} from feature vis set.")

   
    datasets, gt_datasets = [], []
    dataset_names = []                  
    for i in range(len(args.image_paths)):
        sub_args = copy.copy(args)
        sub_args.image_path = args.image_paths[i]
        sub_args.action_path = args.action_paths[i]
        
                     
        dataset_name = "R2R" if "R2R" in args.image_paths[i] else "RxR" if "RxR" in args.image_paths[i] else f"Dataset_{i}"
        dataset_names.append(dataset_name)
        
        del sub_args.image_paths
        del sub_args.action_paths
        
        ds = get_corruption_dataset(
            sub_args, 
            augment=True, 
            mixing_images1=mixing_images1, 
            mixing_images2=mixing_images2
        )
        datasets.append(ds)

        gt_ds = get_corruption_dataset(
            sub_args, 
            augment=False,                      
            mixing_images1=mixing_images1, 
            mixing_images2=mixing_images2
        )
        gt_datasets.append(gt_ds)

                     
    if len(datasets) > 1:
                    
        dataset_lengths = [len(ds) for ds in datasets]
        gt_dataset_lengths = [len(ds) for ds in gt_datasets]
        
        concat_dataset = ConcatDataset(datasets)
        concat_gt_dataset = ConcatDataset(gt_datasets)
        
                      
        dataset = ActionRemappingDataset(concat_dataset, dataset_names, dataset_lengths)
        gt_dataset = ActionRemappingDataset(concat_gt_dataset, dataset_names, gt_dataset_lengths)
    elif len(datasets) == 1:
                           
        dataset_lengths = [len(datasets[0])]
        gt_dataset_lengths = [len(gt_datasets[0])]
        
        dataset = ActionRemappingDataset(datasets[0], dataset_names, dataset_lengths)
        gt_dataset = ActionRemappingDataset(gt_datasets[0], dataset_names, gt_dataset_lengths)
    else:
        raise ValueError("No dataset created, please check image_paths and action_paths.")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    gt_sampler = torch.utils.data.distributed.DistributedSampler(gt_dataset, num_replicas=args.world_size, rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    
    gt_loader = torch.utils.data.DataLoader(
        gt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=gt_sampler,
        drop_last=True,
    )

    args.layout = False
    model = get_flow_model(args).to(device)
    
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)                       
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    broadcast_params(model.parameters())

    base_optimizer = optim.AdamW([
        {'params': model.parameters()}
    ], lr=args.lr, weight_decay=0.0)
    optimizer = base_optimizer

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, args.num_epoch, eta_min=1e-5)

         
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    exp = args.exp
    
    parent_dir = "/home/zhangyf/navid_ws/LFM/saved_info/dual_flow/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.load_state_dict(checkpoint["model_dict"])
        base_optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]
        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
           
    val_datasets, gt_val_datasets = [], []
    val_dataset_names = []             
    for i in range(len(args.val_image_paths)):
        sub_args = copy.copy(args)
        sub_args.image_path = args.val_image_paths[i]
        sub_args.action_path = args.val_action_paths[i]
        
                     
        val_dataset_name = "R2R" if "R2R" in args.val_image_paths[i] else "RxR" if "RxR" in args.val_image_paths[i] else f"ValDataset_{i}"
        val_dataset_names.append(val_dataset_name)
        
        if hasattr(sub_args, 'val_image_paths'):
            del sub_args.val_image_paths
        if hasattr(sub_args, 'val_action_paths'):
            del sub_args.val_action_paths
            
        val_ds = get_corruption_dataset(
            sub_args, 
            augment=True,  
            mixing_images1=mixing_images1, 
            mixing_images2=mixing_images2
        )
        val_datasets.append(val_ds)

        gt_val_ds = get_corruption_dataset(
            sub_args, 
            augment=False,  
            mixing_images1=mixing_images1, 
            mixing_images2=mixing_images2
        )
        gt_val_datasets.append(gt_val_ds)

                     
    if len(val_datasets) > 1:
                      
        val_dataset_lengths = [len(ds) for ds in val_datasets]
        gt_val_dataset_lengths = [len(ds) for ds in gt_val_datasets]
        
        concat_val_dataset = ConcatDataset(val_datasets)
        concat_gt_val_dataset = ConcatDataset(gt_val_datasets)
        
                      
        val_dataset = ActionRemappingDataset(concat_val_dataset, val_dataset_names, val_dataset_lengths)
        gt_val_dataset = ActionRemappingDataset(concat_gt_val_dataset, val_dataset_names, gt_val_dataset_lengths)
    elif len(val_datasets) == 1:
                             
        val_dataset_lengths = [len(val_datasets[0])]
        gt_val_dataset_lengths = [len(gt_val_datasets[0])]
        
        val_dataset = ActionRemappingDataset(val_datasets[0], val_dataset_names, val_dataset_lengths)
        gt_val_dataset = ActionRemappingDataset(gt_val_datasets[0], val_dataset_names, gt_val_dataset_lengths)
    else:
        raise ValueError("No validation dataset created.")

                  
    val_subset_size = min(len(val_dataset), 500)
    val_subset = torch.utils.data.Subset(val_dataset, range(val_subset_size))
    gt_val_subset = torch.utils.data.Subset(gt_val_dataset, range(val_subset_size))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=args.world_size, rank=rank)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(gt_val_subset, num_replicas=args.world_size, rank=rank)
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,                 
    )
    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=gt_val_sampler,
        drop_last=True,                 
    )

             
    model.train()
    if rank == 0:
            print("start training")
    for epoch in tqdm(range(init_epoch, args.num_epoch + 1)):
        train_sampler.set_epoch(epoch)
        gt_sampler.set_epoch(epoch)
        for iteration, (batch, gt_batch) in enumerate(zip(data_loader, gt_loader)):
                          
            cur_img_aug = batch['current_image'].to(device, non_blocking=True)
            prev_img_aug = batch['prev_image'].to(device, non_blocking=True)
            prev_action = batch['prev_action'].to(device, non_blocking=True)

                                    
            cur_img_gt = gt_batch['current_image'].to(device, non_blocking=True)
            prev_img_gt = gt_batch['prev_image'].to(device, non_blocking=True)
            model.zero_grad()
                        
            
                                
            pre_img_latent_aug = first_stage_model.encode(prev_img_aug).latent_dist.sample().mul_(args.scale_factor)
            cur_img_latent_aug = first_stage_model.encode(cur_img_aug).latent_dist.sample().mul_(args.scale_factor)
            prev_img_latent_gt = first_stage_model.encode(prev_img_gt).latent_dist.sample().mul_(args.scale_factor)

                                        
            cur_img_latent_gt = first_stage_model.encode(cur_img_gt).latent_dist.sample().mul_(args.scale_factor)
            
            
                      
            t = torch.rand((cur_img_latent_gt.size(0),), device=device)
            t = t.view(-1, 1, 1, 1)
            z_0 = torch.randn_like(cur_img_latent_gt)
            v_t = (1 - t) * cur_img_latent_gt + (1e-5 + (1 - 1e-5) * t) * z_0               
            u = (1 - 1e-5) * z_0 - cur_img_latent_gt            
            

                                      
            use_first_frame_training = random.random() < 0.3
            
            if use_first_frame_training:      
                pre_recon_approx = torch.zeros_like(prev_img_latent_gt)
                prev_action = torch.full((z_0.size(0),), 0, device=device, dtype=torch.long)
                             
             
            else:
                                           
                with torch.no_grad():
                    q = epoch / args.num_epoch * 0.8
                    if random.random() < q:
                        t_z = 0.95
                        noisy_start = (1 - t_z) * cur_img_latent_aug + t_z * z_0  
                        prev_action_tmp = torch.full((z_0.size(0),), 0, device=device, dtype=torch.long)
                                                 
                        temp_condition = torch.cat((pre_img_latent_aug, torch.zeros_like(pre_img_latent_aug)), dim=1)
                        prev_img_latent_denoise = denoise_from_noisy(model, noisy_start, temp_condition, device, prev_action_tmp, num_steps=20)
                        pre_recon_approx = prev_img_latent_denoise - pre_img_latent_aug 
                    else:
                        pre_recon_approx = prev_img_latent_gt - pre_img_latent_aug 
            
                                                                                                                                   
                                                                                                                                                         
            condition_input = torch.cat((cur_img_latent_aug, pre_recon_approx), dim=1)        
            v_t_masked = torch.cat((v_t, condition_input), dim=1)                
            flow_loss = F.mse_loss(model(t.squeeze(), v_t_masked, y=prev_action), u)
                                     
            
            flow_loss.backward()
            optimizer.step()
            global_step += 1

                            
            if iteration % 100 == 0:
                if rank == 0:
                    wandb.log({
                        "train/total_loss": flow_loss.item()
                    })
                    if use_first_frame_training:
                                    
                        wandb.log({
                            "train/flow_loss_first_frame": flow_loss.item()
                        })
                    else:
                                              
                        wandb.log({
                            "train/flow_loss_prev_predict": flow_loss.item(),
                                                                                            
                        })
                        
                                  
        scheduler.step()

        if rank == 0:
                            
            wandb.log({
                "train/epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model.state_dict(),
                        "optimizer": base_optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
              
                    }
                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, "model_{}.pth".format(epoch)),
                )

                      
            if rank == 0:
                model.eval()
                with torch.no_grad():
                                                  
                    val_iter = iter(zip(val_loader, gt_val_loader))
                    try:
                        val_batch, gt_val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(zip(val_loader, gt_val_loader))
                        val_batch, gt_val_batch = next(val_iter)
                    
                              
                    cur_img_aug = val_batch['current_image'][:4].to(device, non_blocking=True)
                    prev_img_aug = val_batch['prev_image'][:4].to(device, non_blocking=True)
                    prev_action = val_batch['prev_action'][:4].to(device, non_blocking=True)
                    cur_img_gt = gt_val_batch['current_image'][:4].to(device, non_blocking=True)
                                                                                                

                                               
                    pre_img_latent_aug = first_stage_model.encode(prev_img_aug).latent_dist.sample().mul_(args.scale_factor)
                    cur_img_latent_aug = first_stage_model.encode(cur_img_aug).latent_dist.sample().mul_(args.scale_factor)
                                                                                                                             

                                      
                    z_0 = torch.randn_like(cur_img_latent_aug)
                    
                                                                       
                    c = torch.zeros_like(pre_img_latent_aug)
                    condition_input_1 = torch.cat((cur_img_latent_aug, c), dim=1)               
                    condition_input_pre = torch.cat((pre_img_latent_aug, c), dim=1)               

                                                                                                
                    t_z = 0.95
                    noisy_start = (1 - t_z) * cur_img_latent_aug + t_z * z_0  
                    prev_action_tmp = torch.full((z_0.size(0),), 0, device=device, dtype=torch.long)
                    prev_img_latent_denoise = denoise_from_noisy(model, noisy_start, condition_input_pre, device, prev_action_tmp, num_steps=20)
                    prev_denoise_image = first_stage_model.decode(prev_img_latent_denoise / args.scale_factor).sample
                    pre_recon_approx = prev_img_latent_denoise - pre_img_latent_aug
                    condition_input_2 = torch.cat((cur_img_latent_aug, pre_recon_approx), dim=1)               
                    
                                                       
                    t_z = 0.95
                    noisy_start = (1 - t_z) * cur_img_latent_aug + t_z * z_0  
                    prev_action_tmp = torch.full((z_0.size(0),), 0, device=device, dtype=torch.long)
                    fake_latent_1 = denoise_from_noisy(model, noisy_start, condition_input_1, device, prev_action_tmp, num_steps=20)
                    fake_image_1 = first_stage_model.decode(fake_latent_1 / args.scale_factor).sample

                                                
                    fake_latent_2 = denoise_from_noisy(model, noisy_start, condition_input_2, device, prev_action, num_steps=20)
                    fake_image_2 = first_stage_model.decode(fake_latent_2 / args.scale_factor).sample
                    
                                   
                               
                    display_images_1 = []
                    for i in range(4):          
                        row_images = [
                            cur_img_aug[i],                 
                            cur_img_gt[i],               
                            fake_image_1[i]                
                        ]
                        display_images_1.extend(row_images)
                    
                                         
                    display_images_2 = []
                    for i in range(4):          
                        row_images = [
                            cur_img_aug[i],                    
                            cur_img_gt[i],                    
                            prev_denoise_image[i],            
                            fake_image_2[i]                               
                        ]
                        display_images_2.extend(row_images)
                  
                                  
                    grid_image_1 = torchvision.utils.make_grid(display_images_1, nrow=3, normalize=True)
                    torchvision.utils.save_image(
                        grid_image_1,
                        os.path.join(exp_path, "validation_first_frame_epoch_{}.png".format(epoch)),
                        normalize=True,
                    )
                    
                                            
                    grid_image_2 = torchvision.utils.make_grid(display_images_2, nrow=4, normalize=True)
                    torchvision.utils.save_image(
                        grid_image_2,
                        os.path.join(exp_path, "validation_prev_predict_epoch_{}.png".format(epoch)),
                        normalize=True,
                    )
                    
                                       
                    wandb.log({
                        "images/validation_first_frame": wandb.Image(grid_image_1, caption=f"First frame condition validation at epoch {epoch}"),
                        "train/epoch": epoch,
                    })
                    
                    wandb.log({
                        "images/validation_prev_predict": wandb.Image(grid_image_2, caption=f"Prev_predict condition validation at epoch {epoch} (cur_aug | prev_gt | cur_gt | generated)"),
                        "train/epoch": epoch
                    })
                
                model.train()


def cleanup():
    """"""
    if dist.is_initialized():
        dist.destroy_process_group()


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    
    gpu_devices = args.gpu_devices
    if rank < len(gpu_devices):
        gpu = gpu_devices[rank]
    else:
        gpu = rank
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier(device_ids=[gpu])
    cleanup()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--image_size", type=int, default=224, help="size of image after padding and resize")
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=12, help="in channel image (4 for v_t + 8 for condition: 4 cur_img + 4 pre_recon)")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of image")
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=3,
        help="number of resnet blocks per scale",
    )
    parser.add_argument("--num_heads", type=int, default=8, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,8),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 2, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--num_classes", type=int, default=7, help="num classes (0-6: stop/first_frame, forward, R2R_left15, R2R_right15, RxR_left30, RxR_right30, other)")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    
    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")
 
             
                                                                                                                                                                                                                                        
                                                                                                                                                                                                                               

    parser.add_argument("--image_paths", nargs='+', type=str, default=["/data/zhangyf/VLN/R2R_VLNCE_v1-3_preprocessed_train","/data/zhangyf/VLN/RxR_VLNCE_v0_train"], help="path to image dataset(s)")
    parser.add_argument("--action_paths", nargs='+', type=str, default=["/home/zhangyf/navid_ws/data/v1/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz","/home/zhangyf/navid_ws/data/v1/RxR_VLNCE_v0/train/train_guide_gt.json.gz"], help="path to action data file(s)")

    parser.add_argument("--keep_aspect_ratio", action="store_true", default=True, help="whether to keep aspect ratio by padding")
                
                                                                                                                                                                            
                                                                                                                                                                                                            

                
                                                                                                                                                                                    
                                                                                                                                                                                      

    parser.add_argument("--pixmix_fractal_path", type=str, default="/home/zhangyf/navid_ws/LFM/fractals_and_fvis/fractals/images", help="Path to PixMix fractal images")
    parser.add_argument("--pixmix_vis_path", type=str, default="/home/zhangyf/navid_ws/LFM/fractals_and_fvis/first_layers_resized256_onevis/images", help="Path to PixMix feature visualization images")

              
    parser.add_argument("--val_image_paths", nargs='+', type=str, default=["/data/zhangyf/VLN/R2R_VLNCE_v1-3_preprocessed_val_seen1"], help="path to validation image dataset(s)")
    parser.add_argument("--val_action_paths", nargs='+', type=str, default=["/home/zhangyf/navid_ws/data/v1/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen_gt.json.gz"], help="path to validation action data file(s)")

                            
    parser.add_argument("--exp", default="recursive_denoise", help="name of experiment")
    parser.add_argument("--dataset", default="corruption", help="name of dataset")
    parser.add_argument("--num_timesteps", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=20)       

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--no_augment", action="store_true", help="disable data augmentation pipeline")
    parser.add_argument("--save_content", action="store_true", default=True)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=4,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=4, help="save ckpt every x epochs")

         
    parser.add_argument(
        "--num_proc_node",
        type=int,
        default=1,
        help="The number of nodes in multi node env.",
    )
    parser.add_argument("--gpu_devices", nargs='+', type=int, default=[4,5,6,7], help="gpu devices")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="6277", help="address for master")

    torch.multiprocessing.set_start_method('spawn', force=True)                      

    args = parser.parse_args()
    args.world_size = args.num_proc_node * len(args.gpu_devices)
    size = len(args.gpu_devices)

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * len(args.gpu_devices)
            global_size = args.num_proc_node * len(args.gpu_devices)
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("starting in debug mode")

        init_processes(0, size, train, args)
