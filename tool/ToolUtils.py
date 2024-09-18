import PIL
import numpy as np
import torch
from typing import Tuple
import cv2
import scipy.ndimage

from filmgrainer import filmgrainer as filmgrainer

# Tensor to PIL
def tensor2pil(image):
    return PIL.Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil_to_numpy(_image:PIL.Image) -> np.ndarray:
    image_np = np.array(_image)
    return image_np

def numpy_to_pil(array: np.ndarray) -> PIL.Image:
    im = PIL.Image.fromarray(array)
    return im

def resize_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the given dimensions using linear interpolation."""
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def resize_img_stable_diffusion_base_pixel(input_image, max_side=1280, min_side=1024, size=None, pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = PIL.Image.fromarray(res)
    return input_image

def resize_image_shortest_side_keep_ratio(image: np.ndarray, value: int) -> np.ndarray:
    """Resize an image to the given dimensions using linear interpolation."""
    h,w,c = image.shape
    aspect_ratio = w / h
    if w < h:
        new_width = value
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = value
        new_width = int(new_height * aspect_ratio)
    
    return cv2.resize(image, (new_width,new_height), interpolation=cv2.INTER_LINEAR)

def resize_for_condition_image(input_image: PIL.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
    return img

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array and scale its values to 0-255."""
    array = tensor.numpy().squeeze()
    return (array * 255).astype(np.uint8)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

def image_bounds(image):
    # Ensure we are working with batches
    image = image.unsqueeze(0) if image.dim() == 3 else image

    return([(0, img.shape[0]-1 , 0, img.shape[1]-1) for img in image],)

def inset_image_bounds(image_bounds, inset_left, inset_right, inset_top, inset_bottom):
    inset_bounds = []
    for rmin, rmax, cmin, cmax in image_bounds:
        rmin += inset_top
        rmax -= inset_bottom
        cmin += inset_left
        cmax -= inset_right

        if rmin > rmax or cmin > cmax:
            raise ValueError("Invalid insets provided. Please make sure the insets do not exceed the image bounds.")

        inset_bounds.append((rmin, rmax, cmin, cmax))
    return (inset_bounds,)

def bounded_image_crop(image, image_bounds):
        # Ensure we are working with batches
        image = image.unsqueeze(0) if image.dim() == 3 else image

        # If number of images and bounds don't match, then only the first bounds will be used
        # to crop the images, otherwise, each bounds will be used for each image 1 to 1
        bounds_len = 1 if len(image_bounds) != len(image) else len(image)

        cropped_images = []
        for idx in range(len(image)):
            # If only one bounds object, no need to extract and calculate more than once.
            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                rmin, rmax, cmin, cmax = image_bounds[idx]

                # Check if the provided bounds are valid
                if rmin > rmax or cmin > cmax:
                    raise ValueError("Invalid bounds provided. Please make sure the bounds are within the image dimensions.")

            cropped_images.append(image[idx][rmin:rmax+1, cmin:cmax+1, :])

        return (torch.stack(cropped_images, dim=0),)


def bounded_image_crop_with_mask(image, mask, padding_left, padding_right, padding_top, padding_bottom):
    image = image.unsqueeze(0) if image.dim() == 3 else image
    mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
    # If number of masks and images don't match, then only the first mask will be used on
    # the images, otherwise, each mask will be used for each image 1 to 1
    mask_len = 1 if len(image) != len(mask) else len(image)
    cropped_images = []
    all_bounds = []
    for i in range(len(image)):
        # Single mask or multiple?
        if (mask_len == 1 and i == 0) or mask_len > 0:
            rows = torch.any(mask[i], dim=1)
            cols = torch.any(mask[i], dim=0)
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]

            rmin = max(rmin - padding_top, 0)
            rmax = min(rmax + padding_bottom, mask[i].shape[0] - 1)
            cmin = max(cmin - padding_left, 0)
            cmax = min(cmax + padding_right, mask[i].shape[1] - 1)

        # Even if only a single mask, create a bounds for each cropped image
        all_bounds.append([rmin, rmax, cmin, cmax])
        cropped_images.append(image[i][rmin:rmax+1, cmin:cmax+1, :])

        return torch.stack(cropped_images), all_bounds
    
def bounded_image_blend(target, target_bounds, source, blend_factor=1, feathering=6):
        # Ensure we are working with batches
        target = target.unsqueeze(0) if target.dim() == 3 else target
        source = source.unsqueeze(0) if source.dim() == 3 else source

        # If number of target images and source images don't match then all source images
        # will be applied only to the first target image, otherwise they will be applied
        # 1 to 1
        # If the number of target bounds and source images don't match then all sourcess will
        # use the first target bounds for scaling and placing the source images, otherwise they
        # will be applied 1 to 1
        tgt_len = 1 if len(target) != len(source) else len(source)
        bounds_len = 1 if len(target_bounds) != len(source) else len(source)

        # Convert target PyTorch tensors to PIL images
        tgt_arr = [tensor2pil(tgt) for tgt in target[:tgt_len]]
        src_arr = [tensor2pil(src) for src in source]

        result_tensors = []
        for idx in range(len(src_arr)):
            src = src_arr[idx]
            # If only one target image, then ensure it is the only one used
            if (tgt_len == 1 and idx == 0) or tgt_len > 1:
                tgt = tgt_arr[idx]

            # If only one bounds object, no need to extract and calculate more than once.
            #   Additionally, if only one bounds obuect, then the mask only needs created once
            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                # Extract the target bounds
                rmin, rmax, cmin, cmax = target_bounds[idx]

                # Calculate the dimensions of the target bounds
                height, width = (rmax - rmin + 1, cmax - cmin + 1)

                # Create the feathered mask portion the size of the target bounds
                if feathering > 0:
                    inner_mask = PIL.Image.new('L', (width - (2 * feathering), height - (2 * feathering)), 255)
                    inner_mask = PIL.ImageOps.expand(inner_mask, border=feathering, fill=0)
                    inner_mask = inner_mask.filter(PIL.ImageFilter.GaussianBlur(radius=feathering))
                else:
                    inner_mask = PIL.Image.new('L', (width, height), 255)

                # Create a blend mask using the inner_mask and blend factor
                inner_mask = inner_mask.point(lambda p: p * blend_factor)

                # Create the blend mask with the same size as the target image
                tgt_mask = PIL.Image.new('L', tgt.size, 0)
                # Paste the feathered mask portion into the blend mask at the target bounds position
                tgt_mask.paste(inner_mask, (cmin, rmin))

            # Resize the source image to match the dimensions of the target bounds
            src_resized = src.resize((width, height), PIL.Image.Resampling.LANCZOS)

            # Create a blank image with the same size and mode as the target
            src_positioned = PIL.Image.new(tgt.mode, tgt.size)

            # Paste the source image onto the blank image using the target bounds
            src_positioned.paste(src_resized, (cmin, rmin))

            # Blend the source and target images using the blend mask
            result = PIL.Image.composite(src_positioned, tgt, tgt_mask)

            # Convert the result back to a PyTorch tensor
            result_tensors.append(pil2tensor(result))

        return (torch.cat(result_tensors, dim=0),)
    
def subtract_masks(masks_a, masks_b)-> torch.Tensor:
    subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
    return subtracted_masks

def add_masks(masks_a, masks_b)-> torch.Tensor:
    if masks_a.ndim > 2 and masks_b.ndim > 2:
        added_masks = masks_a + masks_b
    else:
        added_masks = torch.clamp(masks_a.unsqueeze(1) + masks_b.unsqueeze(1), 0, 255)
        added_masks = added_masks.squeeze(1)
    return added_masks

def expand_mask(mask, expand, tapered_corners)-> torch.Tensor:
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return torch.stack(out, dim=0)

def image_blend_mask(image_a, image_b, mask, blend_percentage)-> torch.Tensor:

    # Convert images to PIL
    img_a = tensor2pil(image_a)
    img_b = tensor2pil(image_b)
    mask = PIL.ImageOps.invert(tensor2pil(mask).convert('L'))

    # Mask image
    masked_img = PIL.Image.composite(img_a, img_b, mask.resize(img_a.size))

    # Blend image
    blend_mask = PIL.Image.new(mode="L", size=img_a.size,color=(round(blend_percentage * 255)))
    blend_mask = PIL.ImageOps.invert(blend_mask)
    img_result = PIL.Image.composite(img_a, masked_img, blend_mask)

    del img_a, img_b, blend_mask, mask

    return pil2tensor(img_result)

def filmgrain_image(image:PIL.Image, scale:float, grain_power:float,
                    shadows:float, highs:float, grain_sat:float,
                    sharpen:int=1, grain_type:int=4, src_gamma:float=1.0,
                    gray_scale:bool=False, seed:int=0) -> PIL.Image:
    # image = pil2tensor(image)
    # grain_type, 1=fine, 2=fine simple, 3=coarse, 4=coarser
    grain_type_index = 3

    # Apply grain

    grain_image = filmgrainer.process(image, scale=scale, src_gamma=src_gamma, grain_power=grain_power,
                                      shadows=shadows, highs=highs, grain_type=grain_type_index,
                                      grain_sat=grain_sat, gray_scale=gray_scale, sharpen=sharpen, seed=seed)
    return tensor2pil(torch.from_numpy(grain_image).unsqueeze(0))