import os
import argparse
from train import mag
from utils import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train',
                    help='train, play, play_temporal')
parser.add_argument('--checkpoint_path', dest='checkpoint', default="./model/epoch50.tar",
                    help='Path of checkpoint file for load model')
parser.add_argument('--data_path', dest='data_path', default=None,
                    help='Path of dataset directory for train model')

# for inference
parser.add_argument('--vid_dir', dest='vid_dir', default=None,
                    help='Video folder to run the network on.')
parser.add_argument('--frame_ext', dest='frame_ext', default='png',
                    help='Video frame file extension.')
parser.add_argument('--out_dir', dest='out_dir', default="./ohetal_output_videos/",
                    help='Output folder of the video run.')
parser.add_argument('--amplification_factor', dest='amplification_factor',
                    type=float, default=5,
                    help='Magnification factor for inference.')
parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true',
                    help='Whether to do velocity magnification.')
parser.add_argument('--is_single_gpu_trained', dest='is_single_gpu_trained', action='store_true',
                    help='Whether the pretrained model was trained on a single gpu.')


# For temporal operation.
parser.add_argument('--fl', dest='fl', type=float, default=30,
                    help='Low cutoff Frequency.')
parser.add_argument('--fh', dest='fh', type=float, default=50,
                    help='High cutoff Frequency.')
parser.add_argument('--fs', dest='fs', type=float, default=500,
                    help='Sampling rate.')
parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int, default=2,
                    help='Number of filter tap required.')
parser.add_argument('--filter_type', dest='filter_type', type=str, default='differenceOfIIR',
                    help='Type of filter to use, must be Butter or differenceOfIIR.')
parser.add_argument("--freq", type=float, nargs='+', default=[1, 6], help='filter low, high')

# For multi-gpu setting
parser.add_argument('-n', '--nodes', default=1, type=int)
parser.add_argument('-g', '--gpus', default=8, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=4, type=int)

arguments = parser.parse_args()
summary_dir = "./summary"
summary_writer = SummaryWriter(summary_dir)


def main(args):
    # Define a class including the model, training function, and inference function
    model = mag(args)

    if args.phase == "play": # Inference without a temporal filter, 
        os.makedirs("./outputs", exist_ok=True)
        save_vid_name = os.path.basename(args.vid_dir) + "_mag" + str(int(args.amplification_factor))
        if args.velocity_mag:
            save_vid_name = save_vid_name + "_dynamic.mp4" # If the reference frame is the previous frame, it is a dynamic mode
        else:
            save_vid_name = save_vid_name + "_static.mp4" # If the reference frame is the first frame, it is a static mode

        save_vid_name = os.path.join("./outputs", save_vid_name)

        model.play(args.vid_dir,
                   args.frame_ext,
                   save_vid_name,
                   args.amplification_factor,
                   args.velocity_mag)

    elif args.phase == "play_temporal": # Inference with a temporal filter [differenceOfIIR, butterworth, fir]
        os.makedirs("./outputs", exist_ok=True)
    
        save_vid_name = os.path.basename(args.vid_dir) + 'm_{}_fl{}_fh{}_fs{}_n{}_{}.mp4'.format(int(args.amplification_factor), args.freq[0], args.freq[1], args.fs,
                                                      args.n_filter_tap,
                                                      args.filter_type) 
        save_vid_name = os.path.join("./outputs", save_vid_name)

        model.play_temporal(args.vid_dir,
                            save_vid_name,
                            args.amplification_factor,
                            args.freq,
                            args.fs,
                            args.filter_type,
                            args.n_filter_tap)

    elif args.phase == 'train':
        if not os.path.isdir(args.data_path):
            raise ValueError('There is no directory on the target path')
        model.train()


    else:
        raise ValueError('Invalid phase argument. '
                         'Expected ["train", "play", "play_temporal"], '
                         'got ' + args.phase)


if __name__ == '__main__':
    main(arguments)
