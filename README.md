# Author-verified Pytorch Reimplementation of Learning-based Video Motion Magnification (ECCV 2018)
### [Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Tae-Hyun_Oh_Learning-based_Video_Motion_ECCV_2018_paper.pdf)

## Acknowledgement
**Thank you to Tae-Hyun Oh, a professor at the Postech AMI Lab. and the first author of the Learning-based Video Motion Magnification paper, for validating this PyTorch reimplementation.** 

**Most of this PyTorch reimplementation was written by Kim Sung-Bin, a PhD student at the AMI Lab.**

Most of the source code was referenced in the orignal tensorflow implementation.
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

## Inference
There are various modes for inference in the motion magnification method. Each mode can branch as follows:

    ├── Inference
    │   ├── Without a temporal Filter
    │   │   ├── Static
    │   │   ├── Dynamic
    │   ├── With a temporal filter   
    │   │   ├── differenceOfIIR
    │   │   ├── butter
    │   │   ├── fir

In "Without a temporal filter", the static mode amplifies small motion based on the first frame, while the dynamic mode amplifies small motion by comparing the current frame to the previous frame.

With a temporal filter, amplification is applied by utilizing the temporal filter. This method effectively amplifies small motions of specific frequencies while reducing noise that may arise in the motion magnification results.

**We highly recommend using a temporal filter for real videos, as they are likely to contain the photometric noise.** 

    
    ## For the inference without a temporal filter

    1. get the baby video and split into multi frames. When using a custom video, also split it into multiple frames.
    
            wget -i https://people.csail.mit.edu/mrub/evm/video/baby.mp4
            ffmpeg -i <path_to_input>/baby.mp4 -f image2 <path_to_output>/baby/%06d.png
    
    2. And then run the static mode. Add "--velocity_mag" for dynamic mode.

            python main_dp.py  --checkpoint_path "./model/epoch50.tar" --phase="play" --amplification_factor 20 --vid_dir="Path of the video frames" --is_single_gpu_trained

    **The amplification level can be adjusted by changing the <amplification factor>.** 

    ## For the inference with a temporal filter

    2. And then run the temporal filter mode with differenceOfIIR. This code supports three types of <filter_type>, {"differenceOfIIR", "butter", and "fir"}.
          
           python main_dp.py --checkpoint_path "./model/epoch50.tar" --phase="play_temporal" --vid_dir="Path of the video frames --amplification_factor 20 --fs 30 --freq 0.04 0.4 --filter_type differenceOfIIR --is_single_gpu_trained

    **When applying a temporal filter, it is crucial to accurately specify the frame rate <fs> and the frequency band <freq> to ensure optimal performance and effectiveness.** 

## Citation
    @article{oh2018learning,
      title={Learning-based Video Motion Magnification},
      author={Oh, Tae-Hyun and Jaroensri, Ronnachai and Kim, Changil and Elgharib, Mohamed and Durand, Fr{\'e}do and Freeman, William T and Matusik, Wojciech},
      journal={arXiv preprint arXiv:1804.02684},
      year={2018}
    }
