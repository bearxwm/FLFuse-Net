from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_loader import TrainData
from model import *
from pytorch_ssim import SSIM
from utils import get_parameter_number
from loss_function import Edgeloss


# Train
def train(args):
    # count = 0
    Val_loss_best = 0

    # Site-Packages
    writer = SummaryWriter(args.logs_path)

    # Data
    train_data = TrainData(args.train_path, args)

    # Data_Loader
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch,
                              num_workers=4)

    # Build Model
    model = FLFuseNet().to(args.device)
    print('C_NET:', get_parameter_number(model))

    # Loss
    ssimloss = SSIM().to(args.device)
    Sloss = Edgeloss().to(args.device)

    # Optimizer
    opt = torch.optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True],
                             lr=args.C_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.75)

    # Training
    for epoch in range(args.epochs):

        with tqdm(total=args.data_per_epoch_train, ncols=100, ascii=True) as t:
            # Parameter
            Iteartionnum = 0

            for i, (IR, VI) in enumerate(train_loader):
                Iteartionnum += len(IR)
                t.set_description('Iteartion %s' % (epoch * args.data_per_epoch_train + Iteartionnum))

                IR = IR.to(args.device)
                VI = VI.to(args.device)

                # Model
                fused = model(IR, VI)

                # Losses
                ssim_c_a = 1 - ssimloss(fused, IR)
                ssim_c_b = 1 - ssimloss(fused, VI)

                Eloss_c_a = Sloss(fused, IR)

                Totall_Loss = 1 * ssim_c_a + 1 * ssim_c_b + 5 * Eloss_c_a

                # Update Gradient
                opt.zero_grad()
                Totall_Loss.backward()
                opt.step()

                if i % 100 == 0:
                    writer.add_images(
                        'Train/IR-VI-Fused',
                        torch.cat([
                            IR.detach()[0].unsqueeze(0),
                            VI.detach()[0].unsqueeze(0),
                            fused.detach()[0].unsqueeze(0)
                        ], dim=0),
                        epoch * args.data_per_epoch_train + i * args.train_batch + 1
                    )

                t.update(args.train_batch)

        # Total Parameter Tensorboard
        scheduler.step()
        writer.add_scalar('Parameter/Learning_Rate_G-NET',
                          opt.param_groups[0]['lr'],
                          epoch + 1)
        torch.save(model.state_dict(), './cheackpoint/model.ckpt')