# Michael Ebenstein's Bachelor project 
This repository contains the code to my bachelors project.

## Requirements
+ [FFMPEG](https://www.ffmpeg.org/download.html)
+ [COLMAP](https://colmap.github.io/install.html)
+ [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn.git)

## Installation

+ Clone this repository using `git clone https://github.com/mebenstein/bachelor_proj.git --recurse-submodules`
+ Download the pre-trained weights for [DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2) from their repository and save them in the `deblurganv2` folder as `best_fpn.h5`


## Data preperation

1. Create a directory for your test sequence
    ```bash
    mkdir data/test
    mv video.mp4 data/test/
    ```
2. Generate a deblurred video
    ```bash
    cd deblurganv2
    python predict.py ../data/test/video.mp4 --video
    mv submit/video_deblur.mp4 ../data/test/
    cd ..
    ```
3. Convert video to frames
    ```bash
    cd data/test
    ffmpeg -i video_deblur.mp4 -vf fps=30 unfiltered_images/%04d.png
    ```
4. Filter images based on bluriness
    ```bash
    python ../../image_filtering.py unfiltered_images images <frame_interval:int>
    ```
5. Estimate poses with COLMAP
    ```bash
    python ../../torch-ngp/colmap2nerf.py --images images --colmap_matcher exhaustive --run_colmap
    cd ../../
    ```

## Usage
1. Train representation. Execute `python learn_scene.py --help` to see the available paramters. The weights will be stored in a workspace folder. (During the first execution PyTorch bindings will be compiled and thus it make take a while to start)
2. View representation. Execute `python view_scene.py --help`

## Misc
+ `misc` contains various Jupyter Notebooks used for benchmarking and other tests for the report
+ `grid_encoding_test.py` contains an experimental implementation of the neural hash encoding that was created in order to help me understand the technology better
+ `kilonerf.py` includes a smaller KiloNeRF benchmark

## License
Please see the `LICENSE` file.