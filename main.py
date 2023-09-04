"""
----------------------------------
A Minimal Pytorch version of FLFuseNET
--------------By XWM--------------
"""

import argparse
from utils import del_files


# Args
def parse_args():
    parser = argparse.ArgumentParser()
    # System Parameter
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU_1')

    # Train Parameter
    parser.add_argument('--is_dual', type=int, default=0)
    parser.add_argument('--train_batch', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--C_learning_rate', type=float, default=0.0005)
    parser.add_argument('--patience', type=float, default=3)
    parser.add_argument('--data_per_epoch_train', type=float, default=10000)
    # Paths
    parser.add_argument('--train_path', type=str,
                        default='G:/2D/2D_Train_data/FusionRGB/TrainData/NIR_RGB/',
                        help='Train_Data_Path')

    parser.add_argument('--logs_path', type=str, default='./logs/exp1/')

    return parser.parse_args()


def main():
    args = parse_args()
    from train import train
    del_files(args.logs_path)
    train(args)

if __name__ == '__main__':
    print('Start!!!')
    main()
