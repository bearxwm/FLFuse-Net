import os
import time
import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from data_loader import testpath, TestData
from model import *
from utils import get_parameter_number
# import cv2


# Args
def parse_args():
    parser = argparse.ArgumentParser()
    # System Parameter
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU_1')

    # Test Parameter
    parser.add_argument('--test_batch', type=int, default=1)
    return parser.parse_args()


def genetrate(path, image_type, out_path, args, batch_num=1):
    # Time
    start = time.time()

    # Data
    data = TestData(path, image_type)

    # Data_Loader

    test_data = DataLoader(dataset=data, batch_size=batch_num, pin_memory=True)

    # BUILD Model
    # if use /model_with_dege_branch
    model = FLFuseNet().to(args.device)
    print('C_NET:', get_parameter_number(model))
    model.load_state_dict(torch.load('./cheackpoint/model_with_dege_branch.ckpt', map_location=args.device))

    # if use /model_without_dege_branch
    # model = FLFuseNet_NoEdgeBranch().to(args.device)
    # print('C_NET:', get_parameter_number(model))
    # model.load_state_dict(torch.load('./cheackpoint/model_without_dege_branch.ckpt', map_location=args.device))

    # model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_data), ncols=100, ascii=True) as t:
            for i, (a_lr, b_lr) in enumerate(test_data):
                t.set_description('|| Image %s' % (i + 1))

                images_a = a_lr.to(args.device)
                images_b = b_lr.to(args.device)

                outputs = model(images_a, images_b)
                outputs = torch.clamp(outputs, 0, 1)

                FUSED_PATH = os.path.join(out_path, 'out%03d.png' % (i + 1))
                outputs = ToPILImage()(torch.squeeze(outputs.data.cpu()))
                outputs.save(FUSED_PATH)

                # outputs = (torch.squeeze(outputs.cpu())).numpy()
                # cv2.imwrite(FUSED_PATH, outputs*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                t.update(len(a_lr))

        tqdm.write('|| TIME: %.4f s' % (time.time() - start))
        tqdm.write('|| Pertime: %.4f ms' % (((time.time() - start) * 1000) / len(test_data)))
        tqdm.write('|| Out_path: %s' % out_path)


def fuse(args):
    fusetype = [
        'TNO',
        # 'NIR'

    ]

    for image_type in fusetype:
        print('------%s fuse start!------' % image_type)
        path, out_path = testpath(image_type)
        genetrate(path, image_type, out_path, args, batch_num=args.test_batch)
        print('-----------end!-----------')

args = parse_args()
fuse(args)

