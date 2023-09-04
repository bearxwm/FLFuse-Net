import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
import cv2


class TrainData(Dataset):
    def __init__(self, directory, FLAGS):
        super(TrainData, self).__init__()
        self.FLAGS = FLAGS
        self.directory = directory
        self.data_list_o = []
        self.data_list = []

        file_dir = self.directory + '/'

        for f in os.listdir(file_dir):

            if '_IR' in f:
                self.data_list_o.append(
                    file_dir + f.split('_')[0] + '_')

        for i in range(self.FLAGS.data_per_epoch_train):
            self.data_list.append(random.choice(self.data_list_o))


    def __getitem__(self, index):

        IR = cv2.imread(str(self.data_list[index]) + 'IR.png', cv2.IMREAD_GRAYSCALE)
        IR = np.float32(IR / 255.)

        VI = cv2.imread(str(self.data_list[index]) + 'VI.png', cv2.IMREAD_GRAYSCALE)
        VI = np.float32(VI / 255.)

        IR, VI = self.random_crop(IR, VI, (128, 128))

        return ToTensor()(IR), ToTensor()(VI)

    @staticmethod
    def random_crop(img_a, img_b, crop_size):
        height, width = img_a.shape[:2]
        crop_height, crop_width = crop_size

        if crop_width > width or crop_height > height:
            top, bottom, left, right = 0, 0, 0, 0
            if crop_height > height:
                top = (crop_height - height) // 2
                bottom = crop_height - height - top
            if crop_width > width:
                left = (crop_width - width) // 2
                right = crop_width - width - left
            color = [0, 0, 0]
            padded_img_a = cv2.copyMakeBorder(img_a, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            padded_img_b = cv2.copyMakeBorder(img_b, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        else:
            padded_img_a = img_a
            padded_img_b = img_b

        x_start = random.randint(0, padded_img_a.shape[1] - crop_width)
        y_start = random.randint(0, padded_img_a.shape[0] - crop_height)

        cropped_img_a = padded_img_a[y_start:y_start + crop_height, x_start:x_start + crop_width]
        cropped_img_b = padded_img_b[y_start:y_start + crop_height, x_start:x_start + crop_width]

        return cropped_img_a, cropped_img_b

    def __len__(self):
        return len(self.data_list)



class TestData(Dataset):
    def __init__(self, directory, DataSetName):
        super(TestData, self).__init__()

        self.directory = directory
        self.data_list = []
        self.DataSetName = DataSetName

        file_dir = self.directory + '/'

        for f in os.listdir(file_dir):

            if '_IR' in f:
                self.data_list.append(
                    file_dir + f.split('_')[0] + '_')

    def __getitem__(self, index):

        if self.DataSetName == 'LLVIP_RGB':
            end_ir = '.jpg'
            end_VI = '.jpg'

        elif self.DataSetName == 'M3FD_RGB':
            end_ir = '.png'
            end_VI = '.png'

        elif self.DataSetName == 'Road_RGB':
            end_ir = '.jpg'
            end_VI = '.jpg'

        elif self.DataSetName == 'TNO':
            end_ir = '.png'
            end_VI = '.png'

        IR = cv2.imread(self.data_list[index] + 'IR' + end_ir, cv2.IMREAD_GRAYSCALE)
        IR = np.float32(IR / 255.)

        VI = cv2.imread(self.data_list[index] + 'VI' + end_VI, cv2.IMREAD_GRAYSCALE)
        VI = np.float32(VI / 255.)

        return ToTensor()(IR), ToTensor()(VI)


    def __len__(self):
        return len(self.data_list)


def testpath(image_type):
    path = None
    out_path = None

    if image_type == 'TNO':
        path = './Dataset/TNO'
        out_path = './fused_images/TNO/'
    elif image_type == 'NIR':
        path = 'D:/Dataset/NIR'
        out_path = './fused_images/NIR/'

    return path, out_path
