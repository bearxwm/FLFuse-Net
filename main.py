"""
----------------------------------
Pytorch version of CrossFusion-NET
----------------------------------
Extractor:  Cut_in Block
St_ branch: Soble branch
Loss:       DUALSSIM + SINGLESLOSS
Data:       DUALNIR
Date:       10-14
P:          15,1200
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
                        default='D:/XWM_Workplace/Compare_Data/RGB_NIR/SR/GRAY/4X/train_share/',
                        help='Train_Data_Path')

    parser.add_argument('--train_path_dual_ir', type=str,
                        default='D:/XWM_Workplace/Compare_Data/RGB_NIR/SR/GRAY/4X/test_ir_sr/',
                        help='Train_Data_Path')
    parser.add_argument('--train_path_dual_vis', type=str,
                        default='D:/XWM_Workplace/Compare_Data/RGB_NIR/SR/GRAY/4X/test_vis_sr/',
                        help='Test_Data_Path')




    parser.add_argument('--logs_path', type=str, default='./logs/exp1/')

    return parser.parse_args()


def main():
    args = parse_args()
    MODE = int(input("Please input MODE !!! \n1 for train OR 2 for test"))
    if MODE is 1:
        from train import train
        del_files(args.logs_path)
        train(args)
    elif MODE is 2:
        from test import test
        test(args)


if __name__ == '__main__':
    # Start Run!
    print('Start!!!')
    main()
