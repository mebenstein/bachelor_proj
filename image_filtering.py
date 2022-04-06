import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import shutil
import numpy as np
from functools import partial
import click
from typing import List,Iterator

def get_blur(fname:str) -> float:    
    """Calculate the bluriness of an image using variance of laplacian

    Args:
        fname (str): absolute file name

    Returns:
        float: bluriness value
    """    

    img = cv2.imread(fname)
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    
    return blur

def get_blurs(paths:List[str]) -> List[float]:
    """Calculate blurriness for a list of files in parallel

    Args:
        paths (List[str]): list of absolute file paths 

    Returns:
        List[float]: list of bluriness values
    """    
    with Pool(12) as p: # run in parrallel with progress bar
       return list(tqdm(p.imap(get_blur, paths), total=len(paths), desc="calculating image blur"))

def filter_images(paths:List[str], interval:int, threshold:float) -> Iterator[int]:
    """filter a list of images to yield a sample within each interval and with minimal sharpness 

    Args:
        paths (List[str]): list of image absolute filenames
        interval (int): sample interval
        threshold (float): minimum sharpness

    Yields:
        Iterator[int]: indices of images in the resulting subset
    """    
    blurs = get_blurs(paths) # get blur values

    for i in range(0,len(blurs),interval):
        sub = blurs[i:i+interval] # select subsection of interval size
        amax = np.argmax(sub)
        if sub[amax] > threshold:
            yield i+amax

@click.command()
@click.argument('input',required=True)
@click.argument('target',required=True)
@click.argument('interval',required=True,type=int)
@click.option('--threshold',default=80,type=float,help="minimum sharpness threshold")
def filter_folder(input:str,target:str, interval:int, threshold:float):
    """Filters the images in the input folder and writes the resulting subset to the target folder.

    Args:\n
        input (str): input folder containing numbered images\n
        target (str): output folder\n
        interval (int): frame selection interval, should be chosen to yield 50-100 images\n
        threshold (float): minimum sharpness threshold\n
    """    

    if not os.path.exists(target):
        os.makedirs(target)
    
    # get sorted images files and generate absolute paths
    frames = sorted(filter(lambda x: x.split(".")[-1].lower() in ["png","jpg","jpeg"],os.listdir(input)))
    abs_frames = list(map(partial(os.path.join,input),frames))

    # copy the resulting subset to the target folder
    for i in filter_images(abs_frames, interval, threshold):
        shutil.copyfile(abs_frames[i], os.path.join(target,frames[i]))

if __name__ == "__main__":
    filter_folder()