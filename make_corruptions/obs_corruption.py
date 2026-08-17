                       

import os
from PIL import Image
import numpy as np

from PIL import Image
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes

                                             


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, candi_images):
    images = []
    dir = os.path.expanduser(dir)
                                          
    for name in sorted(candi_images):
        path = os.path.join(dir, name)
        item = (path, name)
        images.append(item)

    return images


def pil_loader(path):
                                                                                                     
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


                            
                     
          
                                     
                     
                                                                  
                                 


                           
                                               
                                           
                                      
           
                                 


                                         
                                                                                                                       
                                          
                                                 
                            
                                                                                   
                                                                                     

                          
                              
                                  
                          
                                    
                                                  
                              
                            
                                          
                                    

                                   
                                       
                                 
                                        
                                       
                                                   

                                                                                                                                                         
                                           
                  
                                        
                     
                                  

                                                                                         

                                                                                    

                                                             

                        
                               


                                                    

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)




def auc(errs):                                         
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

                                   
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,
                                              ctypes.c_double,
                                              ctypes.c_double,
                                              ctypes.c_double)


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


                                                                                
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


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
                                
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
                               
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_left:trim_left + w]


                                                        


                                             
def none_cor(x):
    return np.array(x)
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38, 0.45][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3, 2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27, 0.35][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6, 8][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5), (12, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))                          

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15), (20, 15)][severity - 1]

    output = BytesIO()
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(x.astype(np.uint8))
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if len(x.shape) == 3 and x.shape[2] == 3:
        return np.clip(x[..., [2, 1, 0]], 0, 255)
    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03),
         np.arange(1, 1.36, 0.04)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4), (3, 1.3)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    h, w = x.shape[:2]
    
                                               
    mapsize = 1
    while mapsize < max(h, w):
        mapsize *= 2
    
    plasma = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])
    
                                              
    if plasma.shape[0] >= h and plasma.shape[1] >= w:
              
        start_h = np.random.randint(0, max(1, plasma.shape[0] - h + 1))
        start_w = np.random.randint(0, max(1, plasma.shape[1] - w + 1))
        plasma = plasma[start_h:start_h + h, start_w:start_w + w]
    else:
              
        plasma = cv2.resize(plasma, (w, h))
    
    x += c[0] * plasma[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75),
         (0.55, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['make_corruptions/frost1.png', 'make_corruptions/frost2.png', 'make_corruptions/frost3.png', 'make_corruptions/frost4.jpg', 'make_corruptions/frost5.jpg', 'make_corruptions/frost6.jpg'][idx]
    frost = cv2.imread(filename)
                                      
    h, w = np.array(x).shape[:2]
    
                                    
    if frost.shape[0] < h or frost.shape[1] < w:
                               
        scale_h = max(1.0, h / frost.shape[0])
        scale_w = max(1.0, w / frost.shape[1])
        scale = max(scale_h, scale_w)
        new_h = int(frost.shape[0] * scale)
        new_w = int(frost.shape[1] * scale)
        frost = cv2.resize(frost, (new_w, new_h))
    
    x_start = np.random.randint(0, max(1, frost.shape[0] - h + 1))
    y_start = np.random.randint(0, max(1, frost.shape[1] - w + 1))
    frost = frost[x_start:x_start + h, y_start:y_start + w][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.50)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])                       

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8),
        mode='L',
    )
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(
        np.frombuffer(snow_layer.make_blob(), np.uint8),
        cv2.IMREAD_UNCHANGED,
    ) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
                                                                              
                                 
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

                                 
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
                                           

                   
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5, .6][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2), (30, 0.3)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7, 5][severity - 1]

    output = BytesIO()
                            
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(x.astype(np.uint8))
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

                          
    return np.array(x)

def occlude_with_color_box(image, box_color, box_coordinates):
    """
    
    
    Args:
        image: numpy
        box_color: (r, g, b)
        box_coordinates: (x, y, w, h)
    
    Returns:
        
    """
    x, y, w, h = box_coordinates
    r, g, b = box_color
    
                
    h_im, w_im = image.shape[:2]
    x = max(0, min(x, w_im))
    y = max(0, min(y, h_im))
    w = min(w, w_im - x)
    h = min(h, h_im - y)
    
                
    image[y:y+h, x:x+w] = [r, g, b]
    
    return image

def occlusion(x, severity=1):
    x = np.array(x)
    h_im, w_im = x.shape[:2]
    
                         
    num_boxes = 2
    base_box_size = [8, 12, 16, 20, 24, 60][severity - 1]

    for _ in range(num_boxes):
                         
        w_box = base_box_size + np.random.randint(-4, 5)
        h_box = base_box_size + np.random.randint(-4, 5)
        
                    
        w_box = min(w_box, w_im)
        h_box = min(h_box, h_im)
        
                  
        r, g, b = 0,0,0
        
                  
        if w_im > w_box and h_im > h_box:
            x_pos = np.random.randint(0, w_im - w_box)
            y_pos = np.random.randint(0, h_im - h_box)
            
                  
            x = occlude_with_color_box(x, box_color=(r, g, b), 
                                     box_coordinates=(x_pos, y_pos, w_box, h_box))
    return x

def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25, 0.1][severity - 1]

                            
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(x.astype(np.uint8))
    h, w = x.size[1], x.size[0]                                  
    x = x.resize((int(w * c), int(h * c)), PILImage.BOX)
    x = x.resize((w, h), PILImage.BOX)

                          
    return np.array(x)


                                                                 
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),                                                                  
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02),
         (244 * 0.15, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

                   
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def light_out(x, severity=1):
    x = np.array(x)
    
                                 
                            
    darkening_factors = [0.8, 0.65, 0.5, 0.35, 0.25, 0.1][severity - 1]
    
              
    x = x * darkening_factors
    
    return np.clip(x, 0, 255).astype(np.uint8)




                                                                                          
                                    
                                      
                             
                                                 
                                                                     
                                  
                                             
                             
                                
                                                                            
                          
                                                                 
                                                                              
                                            
                      
