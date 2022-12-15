from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loder import *
from model import *
from pytorch_ssim import SSIM
import torch.nn as nn

from utils import tensorboard_scalar, get_parameter_number, get_psnr
from torchvision.transforms import ToPILImage

from loss_function import Edgeloss


# Train
def train(args):
    # count = 0
    Val_loss_best = 0

    # Site-Packages
    writer = SummaryWriter(args.logs_path)

    # Data

    # train_data = NIRData(args.train_path, args)
    train_data = NIRDataDual(args.train_path_dual_ir, args.train_path_dual_vis, args)

    TNO_a_sr_path, TNO_b_sr_path, TNO_out_path = testpath('TNO')
    TNO_test_data = TestData(TNO_a_sr_path, TNO_b_sr_path)

    NIR_a_sr_path, NIR_b_sr_path, NIR_out_path = testpath('NIR')
    NIR_test_data = TestData(NIR_a_sr_path, NIR_b_sr_path)

    # Data_Loader
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch,
                              num_workers=4)

    TNO_test_loader = DataLoader(dataset=TNO_test_data, batch_size=1)
    NIR_test_loader = DataLoader(dataset=NIR_test_data, batch_size=1)

    # Build Model
    C_model = FLFuseNet().to(args.device)
    print('C_NET:', get_parameter_number(C_model))

    # Loss
    ssimloss = SSIM().to(args.device)
    Sloss = Edgeloss().to(args.device)

    # Optimizer
    C_opt = torch.optim.Adam([paras for paras in C_model.parameters() if paras.requires_grad == True],
                             lr=args.C_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(C_opt, step_size=5, gamma=0.75)
    # Training
    for epoch in range(args.epochs):

        with tqdm(total=args.data_per_epoch_train, ncols=100, ascii=True) as t:
            # Parameter
            Iteartionnum = 0
            T_t_C_ssim = 0

            for i, (img_a_sr, img_b_sr) in enumerate(train_loader):
                Iteartionnum += len(img_a_sr)
                t.set_description('Iteartion %s' % (epoch * args.data_per_epoch_train + Iteartionnum))

                images_a_sr = img_a_sr.to(args.device)
                images_b_sr = img_b_sr.to(args.device)

                # Model
                C_outputs = C_model(images_a_sr, images_b_sr)

                # C_Model_Losses
                C_ssim_c_a = 1 - ssimloss(C_outputs, images_a_sr)
                C_ssim_c_b = 1 - ssimloss(C_outputs, images_b_sr)

                C_Eloss_c_a = Sloss(C_outputs, images_a_sr)

                C_Totall_Loss = \
                    1 * C_ssim_c_b \
                    + 1 * C_ssim_c_a \
                    + 5 * C_Eloss_c_a

                # Update Gradient
                C_opt.zero_grad()
                C_Totall_Loss.backward()
                C_opt.step()

                # TensorBoard
                # C_Net
                tensorboard_scalar(C_ssim_c_b, T_t_C_ssim, len(train_data),
                                   'Train/SSIM', writer, i, epoch, args, is_ssim=1)

                if Iteartionnum % args.train_batch * 100 == 0:
                    tensoboard_images_a = images_a_sr[0]
                    tensoboard_images_b = images_b_sr[0]
                    tensoboard_images_C_outputs = C_outputs[0]

                    # Losses Scalar
                    writer.add_image('TrainIMG/A_SR',
                                     tensoboard_images_a, epoch + 1)
                    writer.add_image('TrainIMG/B_SR',
                                     tensoboard_images_b, epoch + 1)
                    writer.add_image('TrainIMG/C_OUT',
                                     tensoboard_images_C_outputs, epoch + 1)

                t.update(args.train_batch)

        # Total Parameter Tensorboard
        scheduler.step()
        writer.add_scalar('Parameter/Learning_Rate_G-NET',
                          C_opt.param_groups[0]['lr'],
                          epoch + 1)
        torch.save(C_model.state_dict(), './cheackpoint/model.ckpt')

        if (epoch + 1) % 5 == 0:
            # TNO-Testing
            C_model.eval()
            with torch.no_grad():

                with tqdm(total=len(TNO_test_data), ncols=60, ascii=True) as t:
                    # Parameter
                    TNO_Test_ssim = 0
                    TNO_Test_psnr = 0

                    # Loop
                    for i, (img_a_sr, img_b_sr) in enumerate(TNO_test_loader):
                        t.set_description('TNO-Testing...')

                        images_a_sr = img_a_sr.to(args.device)
                        images_b_sr = img_b_sr.to(args.device)

                        outputs = C_model(images_a_sr, images_b_sr)

                        tensoboard_images_outputs = torch.squeeze(torch.clamp(outputs, 0, 1).cpu(), 0)

                        # Testing Batch Loss
                        ssim_a = ssimloss(outputs, images_a_sr)
                        ssim_b = ssimloss(outputs, images_b_sr)
                        ssim = ssim_a / 2 + ssim_b / 2

                        psnr_a = get_psnr(images_a_sr.cpu()[0], outputs.cpu()[0])
                        psnr_b = get_psnr(images_b_sr.cpu()[0], outputs.cpu()[0])
                        psnr = psnr_a / 2 + psnr_b / 2

                        # Batch Loss
                        TNO_Test_ssim += ssim.item()
                        TNO_Test_psnr += psnr.item()

                        FUSED_PATH = os.path.join(TNO_out_path, 'out%03d.png' % (i + 1))
                        outputs = torch.clamp(outputs, 0, 1)
                        outputs = ToPILImage()(outputs.data.cpu()[0])
                        outputs.save(FUSED_PATH)
                        t.update(1)

                        # TensorBoard
                        if (i + 1) % 1 == 0:
                            # Losses Scalar
                            writer.add_image('TNO-IMG/C_%d' % (i + 1),
                                             tensoboard_images_outputs, epoch + 1)

                    # Testing Loss
                    TNO_Test_ssim = TNO_Test_ssim / 20
                    TNO_Test_psnr = TNO_Test_psnr / 20
                    writer.add_scalar('Test/TNO-SSIM', TNO_Test_ssim, epoch + 1)
                    writer.add_scalar('Test/TNO-PSNR', TNO_Test_psnr, epoch + 1)

            # NIR-Testing
            C_model.eval()
            with torch.no_grad():

                with tqdm(total=len(NIR_test_data), ncols=60, ascii=True) as t:
                    # Parameter
                    NIR_Test_ssim = 0
                    NIR_Test_psnr = 0

                    # Loop
                    for i, (img_a_sr, img_b_sr) in enumerate(NIR_test_loader):
                        t.set_description('NIR-Testing...')

                        images_a_sr = img_a_sr.to(args.device)
                        images_b_sr = img_b_sr.to(args.device)

                        outputs = C_model(images_a_sr, images_b_sr)

                        tensoboard_images_outputs = torch.squeeze(torch.clamp(outputs, 0, 1).cpu(), 0)

                        # Testing Batch Loss
                        ssim_a = ssimloss(outputs, images_a_sr)
                        ssim_b = ssimloss(outputs, images_b_sr)
                        ssim = ssim_a / 2 + ssim_b / 2

                        psnr_a = get_psnr(images_a_sr.cpu()[0], outputs.cpu()[0])
                        psnr_b = get_psnr(images_b_sr.cpu()[0], outputs.cpu()[0])
                        psnr = psnr_a / 2 + psnr_b / 2

                        # Batch Loss
                        NIR_Test_ssim += ssim.item()
                        NIR_Test_psnr += psnr.item()

                        FUSED_PATH = os.path.join(NIR_out_path, 'out%03d.png' % (i + 1))
                        outputs = torch.clamp(outputs, 0, 1)
                        outputs = ToPILImage()(outputs.data.cpu()[0])
                        outputs.save(FUSED_PATH)
                        t.update(1)

                        # TensorBoard
                        if (i + 1) % 1 == 0:
                            # Losses Scalar
                            writer.add_image('NIR-IMG/C_%d' % (i + 1),
                                             tensoboard_images_outputs, epoch + 1)

                    # Testing Loss
                    NIR_Test_ssim = NIR_Test_ssim / 12
                    NIR_Test_psnr = NIR_Test_psnr / 12
                    writer.add_scalar('Test/NIR_SSIM', NIR_Test_ssim, epoch + 1)
                    writer.add_scalar('Test/NIR_PSNR', NIR_Test_psnr, epoch + 1)

                    if np.greater(NIR_Test_ssim, Val_loss_best):
                        Val_loss_best = NIR_Test_ssim
                        torch.save(C_model.state_dict(), './cheackpoint/C_NET.ckpt')
