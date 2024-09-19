from beam import Image, endpoint, env, Volume, task_queue

if env.is_remote():
    import torch
    import random
    import torch
    import os
    import sys
    import folder_paths
    import numpy as np
    from tool import ToolUtils
    import base64
    from io import BytesIO
    import boto3
    import PIL
 

    from nodes import (
        VAEDecode,
        KSampler,
        EmptyLatentImage,
        CheckpointLoaderSimple,
        CLIPTextEncode,
    )


image=Image(
        python_version="python3.10",
        python_packages='requirements_cuda121.txt',
        commands=["apt-get update -y && apt-get install --reinstall build-essential -y && apt-get install python3-dev -y && apt-get install ffmpeg -y"])


volume_path = './models'

model_name = "sd_xl_base_1.0.safetensors"
steps = 20 
cfg = 7
sampler = "dpmpp_sde"
scheduler = "normal"


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



def load_models():
    with torch.inference_mode():

        #REGULAR MODEL
        checkpointloadersimple = CheckpointLoaderSimple().load_checkpoint(ckpt_name=model_name)
        model = checkpointloadersimple[0]
        clip = checkpointloadersimple[1]
        vae = checkpointloadersimple[2]

        return model,clip,vae


@task_queue(
    secrets=["TOOL_AWS_KEY","TOOL_AWS_SECRET"],
    on_start=load_models,
    name="SDXL",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    image=image,
    volumes=[Volume(name="models", mount_path=volume_path)],
    keep_warm_seconds=600
)

def handler(context,**inputs):
    with torch.inference_mode():

        MAX_SEED = np.iinfo(np.int32).max

        model,clip,vae = context.on_start_value

        task_id = context.task_id
        
        my_key = os.environ["TOOL_AWS_KEY"]
        my_secret = os.environ["TOOL_AWS_SECRET"]

        prompt = inputs["prompt"]
        style = inputs["style"]
        width = inputs["width"]
        height = inputs["height"]
        num_images = inputs["num_images"]
        callback_url = inputs["callback_url"]
        
        fprompt, fnegprompt = set_style(prompt,style)
    
        s3 = boto3.client("s3",region_name='us-east-1',aws_access_key_id=my_key,aws_secret_access_key=my_secret)
        
        #CREATE CLIP 
        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(text=fprompt,clip=clip)
        cliptextencode_7 = cliptextencode.encode(text=fnegprompt, clip=clip)
        clippos = cliptextencode_6[0]
        clipneg = cliptextencode_7[0]

        #PREPARE OUTPUT
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(width=width, height=height, batch_size=num_images)
        ksampler = KSampler()
        vaedecode = VAEDecode()

        seed = random.randint(0, MAX_SEED)

        ksampler_10 = ksampler.sample(model=model,seed=seed,steps=steps,cfg=cfg,sampler_name=sampler,scheduler=scheduler,positive=clippos,negative=clipneg,latent_image=emptylatentimage_5[0]) 
        vaedecode_17 = vaedecode.decode(samples=ksampler_10[0], vae=vae)

        urls = []

        for i in range(num_images):

            pil_outputimage =   ToolUtils.tensor2pil(vaedecode_17[0][i])
            pil_outputimage.save("/tmp/output_"+str(i+1)+".png")
            s3.upload_file("/tmp/output_"+str(i+1)+".png","imagegenerator.toolofna.com",task_id+'_'+str(i+1)+'.png',ExtraArgs={'ContentType': 'image/png'})
            print("https://d3b3nltctjx3p2.cloudfront.net/"+task_id+'_'+str(i+1)+'.png')
            urls.append("https://d3b3nltctjx3p2.cloudfront.net/"+task_id+'_'+str(i+1)+'.png')
     
        return {"urls":urls}