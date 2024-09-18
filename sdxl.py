import random
import torch

import sys

import folder_paths

import numpy as np

from tool import ToolUtils

import PIL



from nodes import (

    VAEDecode,

    KSamplerAdvanced,

    KSampler,

    EmptyLatentImage,

    CheckpointLoaderSimple,

    CLIPTextEncode,

    LoraLoader,

    ControlNetLoader,

    LoadImage,
)


MAX_SEED = np.iinfo(np.int32).max


def set_style(prompt, style):

    final_prompt =f""
    negative_prompt = f""

    if style == "3D Model":
        final_prompt= f"professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"
        negative_prompt = f"ugly, deformed, noisy, low poly, blurry, painting"
    elif style == "Analog Film":
        final_prompt= f"analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
        negative_prompt = f"painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    elif style == "Anime":
        final_prompt= f"anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"
        negative_prompt = f"photo, deformed, black and white, realism, disfigured, low contrast"
    elif style == "Cinematic":
        final_prompt= f"cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
        negative_prompt = f"anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    elif style == "Pencil Sketch":
        final_prompt= f"black and white, pencil sketch {prompt}, graphic, rough, lines"
        negative_prompt = f"photo, deformed, realism, disfigured, deformed, glitch, noisy, realistic"
    elif style == "Comic Book":
        final_prompt= f"comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
        negative_prompt = f"photograph, deformed, glitch, noisy, realistic, stock photo"
    elif style == "Digital Art":
        final_prompt= f"concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"
        negative_prompt = f"photo, photorealistic, realism, ugly"
    elif style == "Line Art":
        final_prompt= f"cline art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    elif style == "Lowpoly":
        final_prompt= f"low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition"
        negative_prompt = f"noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    elif style == "Photographic":
        final_prompt= f"cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
        negative_prompt = f"drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
    elif style == "Abstract":
        final_prompt= f"Abstract style {prompt} . Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed"
        negative_prompt = f"realistic, photographic, figurative, concrete"
    elif style == "Watercolor":
        final_prompt= f"Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    elif style == "Monochrome":
        final_prompt= f"Monochrome {prompt} . Black and white, contrast, tone, texture, detailed"
        negative_prompt = f"colorful, vibrant, noisy, blurry, deformed"
    elif style == "Renaissance":
        final_prompt= f"Renaissance style {prompt} . Realistic, perspective, light and shadow, religious or mythological themes, highly detailed"
        negative_prompt = f"ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
    elif style == "Old Masters":
        final_prompt= f"Old masters styled painting {prompt} , highly detailed, matte painting"
        negative_prompt = f"ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract"
    else:
        final_prompt =prompt

    return final_prompt,negative_prompt


def createImage(_outputname):

    with torch.inference_mode():
        
        model_name = "sd_xl_base_1.0.safetensors"
        steps = 20 
        cfg = 7
        sampler = "dpmpp_sde"
        scheduler = "normal"
        style = "Cinematic"
        pos_prompt = "prehistoric tools"
        neg_prompt = "noisy, blurry, low contrast, cheerful, optimistic, vibrant, colorful"
    
        #REGULAR MODELS
        checkpointloadersimple = CheckpointLoaderSimple().load_checkpoint(ckpt_name=model_name)
        model = checkpointloadersimple[0]
        clip = checkpointloadersimple[1]
        vae = checkpointloadersimple[2]

        fprompt, fnegprompt = set_style(pos_prompt,style)

        #CLIP
        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(text=fprompt,clip=clip)
        cliptextencode_7 = cliptextencode.encode(text=fnegprompt , clip=clip)
        clippos = cliptextencode_6[0]
        clipneg = cliptextencode_7[0]

        #OUTPUT NODES
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(width=1024, height=1024, batch_size=1)
        ksampler = KSampler()
        vaedecode = VAEDecode()

        #GENERATION
        seed = random.randint(0, MAX_SEED)
        ksampler_10 = ksampler.sample(model=model,seed=123,steps=steps,cfg=cfg,sampler_name=sampler,scheduler=scheduler,positive=clippos,negative=clipneg,latent_image=emptylatentimage_5[0])
        vaedecode_17 = vaedecode.decode(samples=ksampler_10[0], vae=vae)

        ToolUtils.tensor2pil(vaedecode_17[0]).show()
        ToolUtils.tensor2pil(vaedecode_17[0]).save(_outputname)
        

if __name__ == "__main__":

    createImage("1.png")

    #for x in range(1,10): 
    #    createImage(str(x)+"_.png")