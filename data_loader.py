import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, MyRandomCrop, RandomCrop, Resize


class NIRData(Dataset):

    def __init__(self, file_path_a, args):
        super(NIRData, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir(self.file_path_a)

        n = np.random.choice(len(self.image_path_a), args.data_per_epoch_train, replace=True)

        self.mini_batch_a = []
        for index in n:
            self.mini_batch_a.append(self.image_path_a[index])

    def __getitem__(self, indxe):
        img_path_a = self.mini_batch_a[indxe]
        img_a = Image.open(self.file_path_a + img_path_a)

        img_a = RandomCrop((128, 128), pad_if_needed=True)(img_a)

        img_a = ToTensor()(img_a)
        img_b = img_a
        return img_a, img_b

    def __len__(self):
        return len(self.mini_batch_a)


class NIRDataDual(Dataset):

    def __init__(self, file_path_a, file_path_b, args):
        super(NIRDataDual, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir(self.file_path_a)

        self.file_path_b = file_path_b
        self.image_path_b = os.listdir(self.file_path_b)

        n = np.random.choice(len(self.image_path_a), args.data_per_epoch_train, replace=True)

        self.mini_batch_a = []
        self.mini_batch_b = []
        for index in n:
            self.mini_batch_a.append(self.image_path_a[index])
            self.mini_batch_b.append(self.image_path_b[index])

    def __getitem__(self, indxe):
        img_path_a = self.mini_batch_a[indxe]
        img_a = Image.open(self.file_path_a + img_path_a)

        img_path_b = self.mini_batch_b[indxe]
        img_b = Image.open(self.file_path_b + img_path_b)

        img_a, img_b = MyRandomCrop((128, 128), pad_if_needed=True)(img_a, img_b)

        img_a = ToTensor()(img_a)
        img_b = ToTensor()(img_b)
        return img_a, img_b

    def __len__(self):
        return len(self.mini_batch_a)


class TestData(Dataset):

    def __init__(self, file_path_a, file_path_b):
        super(TestData, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir(self.file_path_a)

        self.file_path_b = file_path_b
        self.image_path_b = os.listdir(self.file_path_b)

    def __getitem__(self, indxe):
        img_path_a = self.image_path_a[indxe]
        img_a = Image.open(self.file_path_a + img_path_a)

        img_path_b = self.image_path_b[indxe]
        img_b = Image.open(self.file_path_b + img_path_b)

        img_a = ToTensor()(img_a)
        img_b = ToTensor()(img_b)
        return img_a, img_b

    def __len__(self):
        return len(self.image_path_a)


class TestDLData(Dataset):

    def __init__(self, file_path_a, file_path_b):
        super(TestDLData, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir(self.file_path_a)

        self.file_path_b = file_path_b
        self.image_path_b = os.listdir(self.file_path_b)

    def __getitem__(self, indxe):
        img_path_a = self.image_path_a[indxe]
        img_a = Image.open(self.file_path_a + img_path_a).convert('L')
        h, w = img_a.size
        # print(img_a.size)

        img_path_b = self.image_path_b[indxe]
        img_b = Image.open(self.file_path_b + img_path_b).convert('L')
        # print(img_b.size)
        # img_b = Resize((400, 300))(img_b)

        img_a = ToTensor()(img_a)
        img_b = ToTensor()(img_b)
        return img_a, img_b

    def __len__(self):
        return len(self.image_path_a)


def testpath(image_type):
    a_path = None
    b_path = None
    out_path = None
    rootpath = 'D:/Train_data/Compare_Date/'

    if image_type is 'TNO':
        a_path = 'D:/XWM_Workplace/Compare_Data/TNO/SR/4X/test_ir_sr/'
        b_path = 'D:/XWM_Workplace/Compare_Data/TNO/SR/4X/test_vis_sr/'
        out_path = './fused_images/TNO/'
    elif image_type is 'NIR':
        a_path = 'D:/XWM_Workplace/Compare_Data/RGB_NIR/SR/GRAY/4X/ir_sr_12/'
        b_path = 'D:/XWM_Workplace/Compare_Data/RGB_NIR/SR/GRAY/4X/vis_sr_12/'
        out_path = './fused_images/NIR/'
    elif image_type is 'CT_MRI':
        a_path = rootpath + 'CT_MRI/test_CT/'
        b_path = rootpath + 'CT_MRI/test_MRI/'
        out_path = rootpath + 'CT_MRI/fused_images/'
    elif image_type is 'MEF':
        a_path = rootpath + 'MEF/test_A_gray/'
        b_path = rootpath + 'MEF/test_B_gray/'
        out_path = rootpath + 'MEF/fused_images/'
    elif image_type is 'LAYOUT':
        a_path = rootpath + 'LAYOUT/test_A_grey/'
        b_path = rootpath + 'LAYOUT/test_B_grey/'
        out_path = rootpath + 'LAYOUT/fused_images/'
    elif image_type is '1W':
        a_path = 'D:/CrossFusion-Net/Test/Data/GREY/'
        b_path = 'D:/CrossFusion-Net/Test/Data/GREY/'
        out_path = './1W/'
    elif image_type is 'DL':
        a_path = r'C:\Users\XWM\Desktop\peizhun\IR/'
        b_path = r'C:\Users\XWM\Desktop\peizhun\VIS/'
        out_path = r'C:\Users\XWM\Desktop\DL_COMPARE\FLFuse/'

    return a_path, b_path, out_path
