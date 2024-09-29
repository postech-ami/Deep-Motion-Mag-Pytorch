
CUDA_VISIBLE_DEVICES=0,1 python main_dp.py --checkpoint_path "./model/epoch50.tar" --phase="play_temporal" \
 --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --amplification_factor 20 --fs 30 --freq 0.04 0.4 --filter_type differenceOfIIR --batch_size 1 --is_single_gpu_trained

CUDA_VISIBLE_DEVICES=0,1 python main_dp.py --checkpoint_path "./model/epoch50.tar" --phase="play_temporal" \
 --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --amplification_factor 20 --fs 30 --freq 0.04 0.4 --filter_type butter --batch_size 1 --is_single_gpu_trained

CUDA_VISIBLE_DEVICES=0,1 python main_dp.py --checkpoint_path "./model/epoch50.tar" --phase="play_temporal" \
 --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --amplification_factor 20 --fs 120 --freq 1 2 --filter_type fir --batch_size 1 --is_single_gpu_trained