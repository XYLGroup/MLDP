import argparse
import os
import numpy as np
import itertools
import cv2
import datetime
import time
import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
import warnings

warnings.filterwarnings("ignore")
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


####### parameter config #######
def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--lambda_cyc", type=float, default=5.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=10.0, help="identity loss weight")
    parser.add_argument("--lambda_SD", type=float, default=1.0, help="feature loss weight")
    parser.add_argument("--lambda_CD", type=float, default=1.0, help="classify loss weight:0.01, 0.1, 1, 10")
    parser.add_argument("--sift_features", type=int, default=20, help="the number of sift features")
    parser.add_argument("--meta", type=bool, default=True, help="training strategy")
    parser.add_argument("--meta_interval", type=int, default=7)

    parser.add_argument("--dataset_name", type=str, default="Weixing", help="Weixing, F15-1, F15-2")
    parser.add_argument("--path", type=str, default="./data/", help="the location of training and testing data")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

    parser.add_argument("--lr_g", type=float, default=8 * 10e-5, help="adam: learning rate - generator")
    parser.add_argument("--lr_d", type=float, default=8 * 10e-5, help="adam: learning rate - discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=60, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")

    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
    parser.add_argument("--num", type=int, default=5, help="draw_row_number")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")

    opt = parser.parse_args()

    os.makedirs("images/%s" % opt.dataset_name + '/ISAR', exist_ok=True)
    os.makedirs("images/%s" % opt.dataset_name + '/OPT', exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    return opt


####### feature-consistency loss #######
def feature_loss(opt, real_A_img, recov_A_img, sift, criterion_feature):
    real_A_img = real_A_img.astype("uint8")
    recov_A_img = recov_A_img.astype("uint8")
    (kp_real_A, des_real_A) = sift.detectAndCompute(real_A_img, None)
    (kp_recov_A, des_recov_A) = sift.detectAndCompute(recov_A_img, None)
    if des_real_A is None:
        des_real_A = np.zeros_like(des_recov_A)
    if des_recov_A is None:
        des_recov_A = np.zeros_like(des_real_A)
    des_real_A = np.vstack((des_real_A, np.zeros_like(des_recov_A)))
    des_recov_A = np.vstack((des_recov_A, np.zeros_like(des_real_A)))
    REAL = Variable(torch.tensor(des_real_A[:opt.sift_features]).unsqueeze(0), requires_grad=False).type(Tensor)
    RECOV = Variable(torch.tensor(des_recov_A[:opt.sift_features]).unsqueeze(0), requires_grad=False).type(Tensor)
    return criterion_feature(REAL, RECOV)


####### cross_entropy #######
def cross_entropy(input):
    return -(torch.log(input[:, 0]) + torch.log(input[:, 1]))


####### classification-consistency loss #######
def classification_loss(R, ISAR_image, recov_ISAR_img, criterion_classify):
    ordinary_out = F.softmax(R(ISAR_image), dim=1)
    recov_out = F.softmax(R(recov_ISAR_img), dim=1)
    loss_ordinary = cross_entropy(ordinary_out).detach()
    loss_recov = cross_entropy(recov_out).detach()
    return criterion_classify(loss_ordinary, loss_recov)


####### test result #######
def sample_images(opt, G_IO, G_OI, batches_done, val_dataloader):
    imgs = next(iter(val_dataloader))
    G_IO.eval()
    G_OI.eval()
    ISAR_img = Variable(imgs["A"].type(Tensor))
    optical_img = Variable(imgs["B"].type(Tensor))

    fake_optical = G_IO(ISAR_img)
    reconstruct_ISAR = G_OI(fake_optical)

    fake_ISAR = G_OI(optical_img)
    reconstruct_optical = G_IO(fake_ISAR)

    # x-axis
    ISAR_img = make_grid(ISAR_img, nrow=opt.num, normalize=True)
    optical_img = make_grid(optical_img, nrow=opt.num, normalize=True)
    fake_ISAR = make_grid(fake_ISAR, nrow=opt.num, normalize=True)
    fake_optical = make_grid(fake_optical, nrow=opt.num, normalize=True)
    reconstruct_ISAR = make_grid(reconstruct_ISAR, nrow=opt.num, normalize=True)
    reconstruct_optical = make_grid(reconstruct_optical, nrow=opt.num, normalize=True)
    # y-axis
    image_grid_ISAR = torch.cat((ISAR_img, fake_optical, reconstruct_ISAR), 1)
    image_grid_OPT = torch.cat((optical_img, fake_ISAR, reconstruct_optical), 1)

    # save images
    save_image(image_grid_ISAR, "images/%s/ISAR/%s.png" % (opt.dataset_name, batches_done), normalize=False)
    save_image(image_grid_OPT, "images/%s/OPT/%s.png" % (opt.dataset_name, batches_done), normalize=False)


def data_loading(opt):
    ####### data-loading #######
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(root=opt.path + r'\%s' % opt.dataset_name, transforms_=transforms_,
                                         unaligned=True), batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.n_cpu)
    # testing data
    val_dataloader = DataLoader(ImageDataset(root=opt.path + r'\%s' % opt.dataset_name, transforms_=transforms_,
                                             unaligned=True, mode="test"), batch_size=opt.num, shuffle=True,
                                num_workers=opt.n_cpu)
    return dataloader, val_dataloader


def model_loss_config(opt, input_shape):
    ####### loss function #######
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_feature = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_classify = torch.nn.MSELoss()

    ####### Network #######
    G_IO = GeneratorResNet_T(input_shape, opt.n_residual_blocks)
    G_OI = GeneratorResNet_T(input_shape, opt.n_residual_blocks)
    D_U = Discriminator(input_shape)
    D_W = Discriminator(input_shape)
    R = VGG16(class_num=2, flating_num=int(512 * (opt.img_height // 32) ** 2))
    if cuda:
        G_IO = G_IO.cuda()
        G_OI = G_OI.cuda()
        D_U = D_U.cuda()
        D_W = D_W.cuda()
        R.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
        criterion_classify.cuda()

    ####### loading parameters #######
    R = torch.load("class_consistency/ISAR_classifier.pkl")
    if opt.epoch != 0:
        G_IO.load_state_dict(torch.load("saved_models/%s/G_IO_%d.pth" % (opt.dataset_name, opt.epoch)))
        G_OI.load_state_dict(torch.load("saved_models/%s/G_OI_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_U.load_state_dict(torch.load("saved_models/%s/D_U_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_W.load_state_dict(torch.load("saved_models/%s/D_W_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        G_IO.apply(weights_init_normal)
        G_OI.apply(weights_init_normal)
        D_U.apply(weights_init_normal)
        D_W.apply(weights_init_normal)

    return criterion_GAN, criterion_cycle, criterion_feature, criterion_identity, criterion_classify, G_IO, G_OI, D_U, D_W, R


def optimizer_config(opt, G_IO, G_OI, D_U, D_W):
    ####### optimizer #######
    optimizer_G = torch.optim.Adam(itertools.chain(G_IO.parameters(), G_OI.parameters()), lr=opt.lr_g,
                                      betas=(opt.b1, opt.b2))
    optimizer_D_U = torch.optim.Adam(D_U.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    optimizer_D_W = torch.optim.Adam(D_W.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    ####### update of learning-rate #######
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_U = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_U, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_W = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_W, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    return optimizer_G, optimizer_D_U, optimizer_D_W, lr_scheduler_G, lr_scheduler_D_U, lr_scheduler_D_W


# training data
def image_generation_train(opt, sift, dataloader, val_dataloader, criterion_GAN, criterion_cycle, criterion_feature,
                           criterion_identity, criterion_classify, G_IO, G_OI, D_U, D_W, R,
                           optimizer_G, optimizer_D_U, optimizer_D_W, lr_scheduler_G, lr_scheduler_D_U,
                           lr_scheduler_D_W):
    ####### initialize for buffer #######
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    ####### train #######
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        Batch = 0
        loss_feature = 0
        loss_class = 0
        for step, batch in enumerate(dataloader):
            # input of the HDP-CycleGAN
            ISAR_img = Variable(batch["A"].type(Tensor))
            optical_img = Variable(batch["B"].type(Tensor))
            valid = Variable(Tensor(np.ones((ISAR_img.size(0), *D_U.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((ISAR_img.size(0), *D_U.output_shape))), requires_grad=False)

            # ------------------
            #  training generators
            # ------------------
            G_IO.train()
            G_OI.train()
            optimizer_G.zero_grad()
            # Identity loss
            loss_id_U = criterion_identity(G_OI(ISAR_img), ISAR_img)
            loss_id_W = criterion_identity(G_IO(optical_img), optical_img)
            loss_identity = (loss_id_U + loss_id_W)

            # GAN loss
            fake_optical = G_IO(ISAR_img)
            loss_GAN_IO = criterion_GAN(D_W(fake_optical), valid)
            fake_ISAR = G_OI(optical_img)
            loss_GAN_OI = criterion_GAN(D_U(fake_ISAR), valid)

            loss_GAN = loss_GAN_IO + loss_GAN_OI

            # Cycle loss
            recov_ISAR = G_OI(fake_optical)
            loss_cycle_ISAR = criterion_cycle(recov_ISAR, ISAR_img)

            ISAR_img_np = ISAR_img.squeeze().data.cpu().numpy().T
            recov_ISAR_np = recov_ISAR.squeeze().data.cpu().numpy().T

            recov_optical = G_IO(fake_ISAR)
            loss_cycle_optical = criterion_cycle(recov_optical, optical_img)

            loss_cycle = loss_cycle_ISAR + loss_cycle_optical

            if opt.meta:
                # hierarchical structure
                if (Batch + 1) % opt.meta_interval == 0 and Batch != 0:
                    # feature-consistency loss
                    # classification-consistency loss
                    loss_feature += feature_loss(opt, ISAR_img_np, recov_ISAR_np, sift, criterion_feature)
                    loss_class += classification_loss(R, ISAR_img.detach(), recov_ISAR.detach(), criterion_classify)
                    # total loss
                    loss_total = loss_GAN + opt.lambda_cyc * loss_cycle + \
                                 opt.lambda_cyc * opt.lambda_id * loss_identity + \
                                 opt.lambda_SD * loss_feature / 8 + \
                                 opt.lambda_CD * loss_class / 8
                    loss_total.backward()
                    optimizer_G.step()
                    loss_feature = 0
                    loss_class = 0
                    Batch = 0
                else:
                    loss_feature += feature_loss(opt, ISAR_img_np, recov_ISAR_np, sift, criterion_feature)
                    loss_class += classification_loss(R, ISAR_img.detach(), recov_ISAR.detach(), criterion_classify)
                    Batch += 1
                    # total loss
                    loss_total = loss_GAN + opt.lambda_cyc * loss_cycle + \
                                 opt.lambda_cyc * opt.lambda_id * loss_identity
                    loss_total.backward()
                    optimizer_G.step()
            else:
                loss_feature += feature_loss(opt, ISAR_img_np, recov_ISAR_np, sift, criterion_feature)
                loss_class += classification_loss(R, ISAR_img.detach(), recov_ISAR.detach(), criterion_classify)
                # total loss
                loss_total = loss_GAN + opt.lambda_cyc * loss_cycle + \
                             opt.lambda_cyc * opt.lambda_id * loss_identity + \
                             opt.lambda_SD * loss_feature / 8 + \
                             opt.lambda_CD * loss_class / 8
                loss_total.backward()
                optimizer_G.step()
                loss_feature = 0
                loss_class = 0

            # -----------------------
            #  training DU
            # -----------------------
            optimizer_D_U.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_U(ISAR_img), valid)
            # Fake loss
            fake_ISAR_ = fake_A_buffer.push_and_pop(fake_ISAR)
            loss_fake = criterion_GAN(D_U(fake_ISAR_.detach()), fake)
            # Total loss
            loss_D_U = (loss_real + loss_fake) / 2
            loss_D_U.backward()
            optimizer_D_U.step()
            # -----------------------
            #  training DW
            # -----------------------
            optimizer_D_W.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_W(optical_img), valid)
            # Fake loss
            fake_optical_ = fake_B_buffer.push_and_pop(fake_optical)
            loss_fake = criterion_GAN(D_W(fake_optical_.detach()), fake)
            # Total loss
            loss_D_W = (loss_real + loss_fake) / 2
            loss_D_W.backward()
            optimizer_D_W.step()

            loss_D = (loss_D_U + loss_D_W) / 2

            # --------------
            #  process loading
            # --------------
            # the rest of training time
            batches_done = epoch * len(dataloader) + step
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # print
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s "
                % (
                    epoch, opt.n_epochs, step, len(dataloader), loss_D.item(), loss_total.item(), loss_GAN.item(),
                    loss_cycle.item(), time_left,))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(opt, G_IO, G_OI, batches_done, val_dataloader)
        # update of learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_U.step()
        lr_scheduler_D_W.step()
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            # save parameters
            torch.save(G_IO.state_dict(), "saved_models/%s/G_IO_%d.pth" % (opt.dataset_name, epoch + 1))
            torch.save(G_OI.state_dict(), "saved_models/%s/G_OI_%d.pth" % (opt.dataset_name, epoch + 1))
            torch.save(D_U.state_dict(), "saved_models/%s/D_U_%d.pth" % (opt.dataset_name, epoch + 1))
            torch.save(D_W.state_dict(), "saved_models/%s/D_W_%d.pth" % (opt.dataset_name, epoch + 1))


def main():
    ####### setup for breakpoint and data-saving #######
    opt = config()
    sift = cv2.SIFT_create()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    dataloader, val_dataloader = data_loading(opt)
    criterion_GAN, criterion_cycle, criterion_feature, criterion_identity, criterion_classify, G_IO, G_OI, D_U, D_W, R = \
        model_loss_config(opt, input_shape)
    optimizer_G, optimizer_D_U, optimizer_D_W, lr_scheduler_G, lr_scheduler_D_U, lr_scheduler_D_W = \
        optimizer_config(opt, G_IO, G_OI, D_U, D_W)
    image_generation_train(opt, sift, dataloader, val_dataloader, criterion_GAN, criterion_cycle, criterion_feature,
                           criterion_identity, criterion_classify, G_IO, G_OI, D_U, D_W, R,
                           optimizer_G, optimizer_D_U, optimizer_D_W, lr_scheduler_G, lr_scheduler_D_U,
                           lr_scheduler_D_W)


if __name__ == '__main__':
    main()
