import random
import torch
import sys
import folder_paths

from tool import ToolUtils

sys.path.append("../")
from nodes import (
    VAEDecode,
    KSamplerAdvanced,
    KSampler,
    EmptyLatentImage,
    CheckpointLoaderSimple,
    CLIPTextEncode,
)

def simpleFlow():

    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(ckpt_name="realvisxlV40_v40Bakedvae.safetensors")

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(width=1024, height=1024, batch_size=1)

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(text="cinematic photo of a zombie, sharp teeth, rotting flesh holes, horror",clip=checkpointloadersimple_4[1],)
        cliptextencode_7 = cliptextencode.encode(text="noisy, blurry, low contrast, cheerful, optimistic, vibrant, colorful", clip=checkpointloadersimple_4[1])

        ksampler = KSampler()
        vaedecode = VAEDecode()
        ksampler_10 = ksampler.sample(model=checkpointloadersimple_4[0],seed=123,steps=25,cfg=3.5,sampler_name="dpmpp_sde",scheduler="normal",positive=cliptextencode_6[0],negative=cliptextencode_7[0],latent_image=emptylatentimage_5[0])

        vaedecode_17 = vaedecode.decode(samples=ksampler_10[0], vae=checkpointloadersimple_4[2])

        ToolUtils.tensor2pil(vaedecode_17[0]).show()
        


if __name__ == "__main__":
    simpleFlow()