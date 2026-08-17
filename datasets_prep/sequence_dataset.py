                                
import json
import os
import glob
import gzip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional as F
from torchvision import transforms as torchvision
from PIL import ImageOps, ImageEnhance, ImageFilter
from PIL import Image as PILImage
                                                                                
                                                                                  
                                                                                

IMAGE_SIZE = 224
                 
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=float)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def gaussian_noise(x, severity=5):
    c = [.08, .12, 0.18, 0.26, 0.38, 0.45][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


                 
def int_parameter(level, maxval):
  return int(level * maxval / 10)

def float_parameter(level, maxval):
  return float(level) * maxval / 10.

def sample_level(n):
  return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)

def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)

def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BICUBIC)

def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BICUBIC)

def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BICUBIC)

def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BICUBIC)

def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BICUBIC)

def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)

def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
    
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return ((out + 1) / 2)

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
  return (out / 2)

mixings = [add, multiply]

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GaussianNoise(object):
    """"""
    def __init__(self, std_range=[0.05, 2]):
        self.std_range = std_range
    
    def __call__(self, x):
                          
        img_array = np.array(x, dtype=np.float32) / 255.0
        
                
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(0, std, img_array.shape)
        
                         
        noisy_img = np.clip(img_array + noise, 0, 1)
        
                  
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

def Simsiam_transform(image):
    transform = torchvision.transforms.Compose([       
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(0.45, 0.45, 0.45, 0.1)
                ], p=0.8), 
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.RandomApply([GaussianBlur([.1, 6.5])], p=0.5),
                torchvision.transforms.RandomApply([GaussianNoise([0.05, 0.2])], p=0.5),
    ])
    return transform(image)

def augment_input(image, aug_severity=1):
    op = np.random.choice(augmentations_all)
    return op(image.copy(), aug_severity)

def pixmix(orig, mixing_pic, mixing_pic2):
    k, beta = 4, 4
    
    mixed = F.to_tensor(augment_input(orig))

    for _ in range(0,np.random.randint(k)+1):
        if np.random.random() < 0.25:
            aug_image_copy = F.to_tensor(augment_input(orig))
        elif np.random.random() < 0.5 or np.random.random() > 0.25:
            if np.random.random() < 0.5:
                aug_image_copy = F.to_tensor(mixing_pic)
            else:
                aug_image_copy = F.to_tensor(mixing_pic2)
        else: 
            aug_image_copy = F.to_tensor(Simsiam_transform(orig))

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)
    return mixed

class SequenceDataset(Dataset):
    def __init__(self, image_path, action_path, transform=None):
        """
        episodeaction
        
        Args:
            image_path: episode
            action_path: action (train_gt.json)
            transform: 
        """
        self.image_path = image_path
        self.action_path = action_path
        self.transform = transform
        
                    
        self.action_data = {}
        
                      
    
        with gzip.open(action_path, 'rt', encoding='utf-8') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                                            
                for episode_id, episode_info in episode_data.items():
                    self.action_data[episode_id] = episode_info['actions']
     
        
                   
        self.samples = []
        self._collect_samples()
    
    def _collect_samples(self):
        """"""
                        
        episode_dirs = glob.glob(os.path.join(self.image_path, "episode_*"))
        
        for episode_dir in episode_dirs:
            episode_id = os.path.basename(episode_dir).replace('episode_', '')
            
                              
            if episode_id not in self.action_data:
                print(f"Warning: Episode {episode_id} not found in action data!")
                continue
            
                               
            image_files = sorted(glob.glob(os.path.join(episode_dir, "step_*.jpg")))
            actions = self.action_data[episode_id]
            
                               
            if len(image_files) != len(actions) + 1:                      
                print(f"Warning: Episode {episode_id} - image count ({len(image_files)}) "
                      f"doesn't match action count ({len(actions)} + 1)")
                continue
            
                                 
            for i in range(1, len(image_files) - 1):               
                sample = {
                    'episode_id': episode_id,
                    'step': i,
                    'current_image_path': image_files[i],
                    'prev_image_path': image_files[i-1],
                    'prev_action': actions[i-1]                
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
                 
        current_img = np.array(Image.open(sample['current_image_path']).convert("RGB"))
        current_img = np.transpose(current_img, (2, 0, 1))              
        current_img = current_img.astype(np.float32) / 255.0
        current_img = torch.from_numpy(current_img)
        
                 
        prev_img = np.array(Image.open(sample['prev_image_path']).convert("RGB"))
        prev_img = np.transpose(prev_img, (2, 0, 1))              
        prev_img = prev_img.astype(np.float32) / 255.0
        prev_img = torch.from_numpy(prev_img)
        
                      
        prev_action = sample['prev_action']
        
              
        if self.transform is not None:
            current_img = self.transform(current_img)
            prev_img = self.transform(prev_img)
        
                     
        current_img = current_img * 2.0 - 1.0
        prev_img = prev_img * 2.0 - 1.0
        
        return {
            'current_image': current_img,
            'prev_image': prev_img,
            'prev_action': prev_action,
            'episode_id': sample['episode_id'],
            'step': sample['step']
        }


             
class SequenceDatasetWithFirstFrame(Dataset):
    def __init__(self, image_path, action_path, transform=None, keep_aspect_ratio=True,
                 augment=False, mixing_images1=None, mixing_images2=None):
        """
        prev_imageprev_actionNone
        """
        self.image_path = image_path
        self.transform = transform
        self.keep_aspect_ratio = keep_aspect_ratio
        
                    
        self.action_data = {}
        with gzip.open(action_path, 'rt', encoding='utf-8') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                for episode_id, episode_info in episode_data.items():
                    self.action_data[episode_id] = episode_info['actions']

        self.samples = []
        self._collect_samples()
        self.augment = augment
        self.mixing_images1 = mixing_images1
        self.mixing_images2 = mixing_images2

    def _collect_samples(self):
        """"""
        episode_dirs = glob.glob(os.path.join(self.image_path, "episode_*"))
        
        for episode_dir in episode_dirs:
            episode_id = os.path.basename(episode_dir).replace('episode_', '')
            
            if episode_id not in self.action_data:
                continue
            
            image_files = sorted(glob.glob(os.path.join(episode_dir, "step_*.jpg")))
            actions = self.action_data[episode_id]
            
                         
            if image_files:
                sample = {
                    'episode_id': episode_id,
                    'step': 0,
                    'current_image_path': image_files[0],
                    'prev_image_path': None,
                    'prev_action': None
                }
                self.samples.append(sample)
            
                            
            for i in range(1, len(image_files) - 1):
                sample = {
                    'episode_id': episode_id,
                    'step': i,
                    'current_image_path': image_files[i],
                    'prev_image_path': image_files[i-1],
                    'prev_action': actions[i-1]
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
                 
        current_pil = Image.open(sample['current_image_path']).convert("RGB")
        
                 
        if sample['prev_image_path'] is not None:
            prev_pil = Image.open(sample['prev_image_path']).convert("RGB")
        else:
                             
            prev_pil = Image.new('RGB', current_pil.size, (0, 0, 0))
        
                         
        current_pil = current_pil.resize((224, 224), Image.Resampling.LANCZOS)
        prev_pil = prev_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
                      
        prev_action = sample['prev_action']
        if prev_action is None:
            prev_action = 0       
        
                           
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            a = np.random.randint(3,6)
                                                                         
            np.random.seed(seed)
            choice = np.random.randint(8)
            
            if choice == 0:                  
                current_image_tensor = F.to_tensor(current_pil)
                prev_image_tensor = F.to_tensor(prev_pil)
            elif choice == 1:                 
                current_corrupted = gaussian_noise(current_pil, severity=a)
                                                    
                current_corrupted_pil = PILImage.fromarray(current_corrupted.astype(np.uint8))
                current_image_tensor = F.to_tensor(current_corrupted_pil)
                
                prev_corrupted = gaussian_noise(prev_pil, severity=a)
                prev_corrupted_pil = PILImage.fromarray(prev_corrupted.astype(np.uint8))
                prev_image_tensor = F.to_tensor(prev_corrupted_pil)
            elif choice in [2, 3, 4, 5]:         
                                                         
                mix_img1_pil = random.choice(self.mixing_images1 if np.random.random() < 0.5 else self.mixing_images2)
                mix_img2_pil = random.choice(self.mixing_images1 if np.random.random() < 0.5 else self.mixing_images2)
                
                                       
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                current_image_tensor = pixmix(current_pil, mix_img1_pil, mix_img2_pil)
                
                                                                             
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                prev_image_tensor = pixmix(prev_pil, mix_img1_pil, mix_img2_pil)
            else:          
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                current_image_tensor = F.to_tensor(Simsiam_transform(current_pil))
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                prev_image_tensor = F.to_tensor(Simsiam_transform(prev_pil))
            
                                                 
            current_image = (current_image_tensor * 2) - 1
            prev_image = (prev_image_tensor * 2) - 1
        
        else:                           
            if self.transform:
                current_image = self.transform(current_pil)
                prev_image = self.transform(prev_pil)
            else:                                          
                                                   
                current_image = F.to_tensor(current_pil)
                prev_image = F.to_tensor(prev_pil)
                             
                current_image = (current_image * 2) - 1
                prev_image = (prev_image * 2) - 1

        return {
            "current_image": current_image,
            "prev_image": prev_image,
            "prev_action": prev_action
        } 
