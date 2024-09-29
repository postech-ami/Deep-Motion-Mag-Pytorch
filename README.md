# Author-verified Pytorch Reimplementation of Learning-based Video Motion Magnification (ECCV 2018)
### [Paper]([https://arxiv.org/abs/2312.11360](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Tae-Hyun_Oh_Learning-based_Video_Motion_ECCV_2018_paper.pdf))

## Highlights
**Thank you to Tae-Hyun Oh, a professor at the Postech AMI Lab. and the first author of the Learning-based Video Motion Magnification paper, for validating this PyTorch reimplementation.** 
**Most of this PyTorch reimplementation was written by Kim Sung-Bin, a PhD student at the AMI Lab.**

Most of the source code was referenced in the orignal tensorflow implementation.
1. https://github.com/12dmodel/deep_motion_mag

## Getting started
This code was developed on Ubuntu 18.04 with Python 3.7.6, CUDA 11.1 and PyTorch 1.8.0, using two NVIDIA TITAN RTX (24GB) GPUs. 
Later versions should work, but have not been tested.

### Environment setup

```
conda create -n dmm_pytorch python=3.7.6
conda activate dmm_pytorch

# pytorch installation
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
pip install numpy==1.21.6
pip install pillow tqdm matplotlib scipy tensorboard opencv-python==4.6.0.66
```

## Training
1. Download the dataset at https://github.com/12dmodel/deep_motion_mag.
2. Modify the data_path in the train.sh file and run it, or enter the following command.
    ```
    python main.py --phase="train" --checkpoint_path="Path to the model.tar" --data_path="Path to the directory where the lmdb file are located"
    ```

## Training

    ├── main.py
    ├── Inference
    │   ├── Without a temporal Filter
    │   │   ├── Static
    │   │   ├── Dynamic
    │   ├── With a temporal filter   
    │   │   ├── differenceOfIIR
    │   │   ├── butter
    │   │   ├── fir


## For quick start
1. get the baby video and split into multi frames

        wget -i https://people.csail.mit.edu/mrub/evm/video/baby.mp4
        ffmpeg -i <path_to_input>/baby.mp4 -f image2 <path_to_output>/baby/%06d.png

2. And then run the "Inference with temporal filter

The default mode is "DifferenceOfIIR".
      
       python main.py --phase="play_temporal" --checkpoint_path="Path of the model" --vid_dir="Path to the directory where the video frames are located" --out_dir="path to the output" --amplification_factor=20
     

## Inference
This command is executed in dynamic mode. Delete "--velocity_mag" for static mode.

    python main.py --phase="play" --checkpoint_path="Path to the model.tar" --vid_dir="Path to the directory where the video frames are located" 
    --out_dir="path to the output" --velocity_mag


**Inference with temporal filtered**

This code supports two types of <filter_type>, {"butter" and "differenceOfIIR"}.

    python main.py --phase="play_temporal" --checkpoint_path="Path to the model.tar" --vid_dir="Path to the directory where the video frames are located" --out_dir="path to the output" --amplification_factor=<amplification_factor> --fl=<low_cutoff> --fh=<high_cutoff> --fs=<sampling_rate> --n_filter_tap=<n_filter_tap> --filter_type=<filter_type>
    
## reconstrunct the dataset folder before training
Download the dataset at https://github.com/12dmodel/deep_motion_mag

Then, organize the folder like below.

    ├── main.py
    ├── train
    │   ├── 1
    │   │   ├── amplified
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 2   
    │   │   ├── frameA
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 3   
    │   │   ├── frameB
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 4   
    │   │   ├── frameC
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    ├── README.md
    ├── requirements.txt
    ├── .
    
**PNG Image Dataset to lmdb file**

create the lmdb file for {train set, validation set} in the train dataset dir
        
    python pngtolmdb.py path/to/master/directory


## Train

**Train**

    python main.py --phase="train" --checkpoint_path="Path to the model.tar" --data_path="Path to the directory where the lmdb file are located"

    
## Citation
    @article{oh2018learning,
      title={Learning-based Video Motion Magnification},
      author={Oh, Tae-Hyun and Jaroensri, Ronnachai and Kim, Changil and Elgharib, Mohamed and Durand, Fr{\'e}do and Freeman, William T and Matusik, Wojciech},
      journal={arXiv preprint arXiv:1804.02684},
      year={2018}
    }
