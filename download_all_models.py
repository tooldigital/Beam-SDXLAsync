import os, glob, sys
import requests
from tqdm import tqdm
import tarfile
import zipfile

def download_large_file(url, filename):
    exists = os.path.exists(filename)
    if exists:
        pass
    else:
        print("downloading "+url + " to "+filename)
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk)) 
                    f.write(chunk)

def extract_tar(tar_path, extract_path):
    # Open the tar file
    with tarfile.open(tar_path) as tar:
        # Extract all contents into the directory specified by extract_path
        tar.extractall(path=extract_path)

def unzip_file(zip_path, extract_path):
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the directory specified by extract_path
        zip_ref.extractall(extract_path)


root_path = './models'
insightface_path = os.path.join(root_path,'insightface')
facedetection_path = os.path.join(root_path,'facedetection')
facerestorepath_path = os.path.join(root_path,'face_restore')
instantid_path = os.path.join(root_path,'instant_id')

#######################
#insightface
#######################
#Antelopev2
download_large_file('https://weights.replicate.delivery/default/InstantID/models.tar',os.path.join(insightface_path,'models/models.tar'))
extract_tar(os.path.join(insightface_path,'models/models.tar'), os.path.join(insightface_path,'models'))
os.remove(os.path.join(insightface_path,'models/models.tar'))

#Buffalo_l
download_large_file('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',os.path.join(insightface_path,'models/buffalo_l.zip'))
unzip_file(os.path.join(insightface_path,'models/buffalo_l.zip'), os.path.join(insightface_path,'models/buffalo_l'))
os.remove(os.path.join(insightface_path,'models/buffalo_l.zip'))

#Inswapper
download_large_file('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true',os.path.join(insightface_path,'inswapper_128.onnx'))


#######################
#facedetection
#######################
download_large_file('https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth',os.path.join(facedetection_path,'yolov5l-face.pth'))
download_large_file('https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',os.path.join(facedetection_path,'parsing_parsenet.pth'))


#######################
#facerestoration
#######################
download_large_file('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',os.path.join(facerestorepath_path,'GFPGANv1.4.pth'))


#######################
#instantID
#######################
download_large_file('https://weights.replicate.delivery/default/InstantID/checkpoints.tar',os.path.join(instantid_path,'checkpoints.tar'))
extract_tar(os.path.join(instantid_path,'checkpoints.tar'), instantid_path)
os.remove(os.path.join(instantid_path,'checkpoints.tar'))

#YOLOV8
#https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
