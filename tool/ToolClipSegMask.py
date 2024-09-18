from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple
import matplotlib.cm as cm

import tool.ToolUtils

class AutoMask:
    def __init__(self,path_to_models):
        self.volume_path = path_to_models
      
    
    def createMasks(self,_inputimage, _inputprompt,_threshold=0.4,_blur=7.0):
        processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=self.volume_path)
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined" ,cache_dir=self.volume_path)
        
        i=_inputimage

        #convert image to numpy array
        image_np = np.array(i)

        input_prc = processor(text=_inputprompt, images=i, padding="max_length", return_tensors="pt")

        # Predict the segemntation mask
        with torch.no_grad():
            outputs = model(**input_prc)
                
        tensor = torch.sigmoid(outputs[0]) # get the mask

        # Apply a threshold to the original tensor to cut off low values
       
        tensor_thresholded = torch.where(tensor > _threshold, tensor, torch.tensor(0, dtype=torch.float))

        # Apply Gaussian blur to the thresholded tensor
        tensor_smoothed = self.blur_mask(tensor_thresholded,_blur)

        # Normalize the smoothed tensor to [0, 1]
        mask_normalized = (tensor_smoothed - tensor_smoothed.min()) / (tensor_smoothed.max() - tensor_smoothed.min())

        # Dilate the normalized mask
        mask_dilated = self.dilate_mask(mask_normalized, 4)

        # Convert the mask to a heatmap and a binary mask
        heatmap = self.apply_colormap(mask_dilated, cm.viridis)
        binary_mask = self.apply_colormap(mask_dilated, cm.Greys_r)

        # Overlay the heatmap and binary mask on the original image
        dimensions = (image_np.shape[1], image_np.shape[0])
        heatmap_resized = tool.ToolUtils.resize_image(heatmap, dimensions)
        binary_mask_resized = tool.ToolUtils.resize_image(binary_mask, dimensions)
        alpha_heatmap, alpha_binary = 0.5, 1

        overlay_heatmap = self.overlay_image(image_np, heatmap_resized, alpha_heatmap)
        overlay_binary = self.overlay_image(image_np, binary_mask_resized, alpha_binary)

        binary_image = Image.fromarray(overlay_binary, mode="RGB")
        heatmap_image = Image.fromarray(overlay_heatmap, mode="RGB")

        return binary_image, heatmap_image

    def apply_colormap(self,mask: torch.Tensor, colormap) -> np.ndarray:
        """Apply a colormap to a tensor and convert it to a numpy array."""
        colored_mask = colormap(mask.numpy())[:, :, :3]
        return (colored_mask * 255).astype(np.uint8)

    def dilate_mask(self,mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
        """Dilate a mask using a square kernel with a given dilation factor."""
        kernel_size = int(dilation_factor * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dilated = cv2.dilate(mask.numpy(), kernel, iterations=1)
        return torch.from_numpy(mask_dilated)

    def blur_mask(self,mask: torch.Tensor, sigma: float) -> np.ndarray:
        """Blur mask."""
        blurred = gaussian_filter(mask.numpy(), sigma=sigma)
        return torch.from_numpy(blurred)

    def overlay_image(self, background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
        """Overlay the foreground image onto the background with a given opacity (alpha)."""
        return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)