from skimage.exposure import equalize_adapthist
import skimage.io
import deeptangle as dt # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np
import cv2
import os
    
def load_images_from_folder(folder_path, num_images=11):
    images = []
    for filename in sorted(os.listdir(folder_path))[:num_images]:
        img_path = os.path.join(folder_path, filename)
        img = skimage.io.imread(img_path)
        
        if img.ndim == 3:
            img = img[...,0]
            
        images.append(img)
        
    return images
    

def clip_processing(input_folder, forward_fn, state, score_threshold, overlap_threshold, correction_factor):
    images = load_images_from_folder(input_folder, num_images=11)
    clip = np.stack(images, axis=0)
    # print("stack", clip.shape)
    
    clip = jnp.array(clip)
    # print("array", clip.shape)
    
    clip = 255 - clip
    clip = equalize_adapthist(clip)
    clip = clip * correction_factor
    clip = clip[None, ...]
    # print("clip shape 2 ", clip.shape)
    
    predictions = dt.detect(
        forward_fn,
        state,
        clip,
        threshold=score_threshold,
        overlap_threshold=overlap_threshold,
    )
    
    return clip, predictions

    
        
    
    
    
     
