from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from datasets.cityscapes import Cityscapes_multiple
from datasets.voc import VOCSegmentation_multiple
from utils import ext_transforms as et
from metrics import StreamSegMetrics

from torch.nn import functional as F
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Set up model
model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
    'deeplabv3plus_mobilenetSA': network.deeplabv3plus_mobilenetSA,
    'deeplabv3plus_mobilenetSAc': network.deeplabv3plus_mobilenetSAc,
    'deeplabv3plus_mobilenet_M': network.deeplabv3plus_mobilenetM,
    'deeplabv3plus_mobilenetSA_M': network.deeplabv3plus_mobilenetSAM,
    'deeplabv3plus_mobilenetSAc_M': network.deeplabv3plus_mobilenetSAcM
}


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_mobilenetSA', 'deeplabv3plus_mobilenetSAc',
                                 'deeplabv3plus_mobilenet_M', 'deeplabv3plus_mobilenetSAc_M',
                                 'deeplabv3plus_mobilenetSA_M'
                                 ], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--test_divide", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight of multi-model prediction (default: 0.5)")

    parser.add_argument("--sin_model", type=str, default='checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth',
                        help="The path of single model")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--divide_data", type=int, default=0)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='logit',
                        choices=['cross_entropy', 'focal_loss', 'logit'], help="loss type (default: False)")
    parser.add_argument("--ye", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--select_class", nargs='+', default=None,
                        help="Select list")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--start_class", type=int, default=1,
                        help='The start class (default: 1)')
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', divide_data=opts.divide_data, download=opts.download,
                                    transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', divide_data=opts.divide_data, download=False,
                                  transform=val_transform)
        test_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                   image_set='test', divide_data=opts.divide_data, download=False,
                                   transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        test_dst = Cityscapes(root=opts.data_root,divide_data=opts.divide_data,
                              split='test', transform=val_transform)
    return train_dst, val_dst, test_dst


def get_dataset_multiple(opts, idx):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation_multiple(root=opts.data_root, year=opts.year,
                                             image_set='train', divide_data=opts.divide_data, download=opts.download,
                                             transform=train_transform, cur_class=idx)
        val_dst = VOCSegmentation_multiple(root=opts.data_root, year=opts.year,
                                           image_set='val', divide_data=opts.divide_data, download=False,
                                           transform=val_transform, cur_class=idx)
        test_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                   image_set='test', divide_data=opts.divide_data, download=False,
                                   transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes_multiple(root=opts.data_root,
                                        split='train', divide_data=opts.divide_data, transform=train_transform,
                                        cur_class=idx - 1)
        val_dst = Cityscapes_multiple(root=opts.data_root,
                                      split='val', divide_data=opts.divide_data, transform=val_transform,
                                      cur_class=idx - 1)
        test_dst = Cityscapes(root=opts.data_root,
                              split='test', divide_data=opts.divide_data, transform=val_transform)
    return train_dst, val_dst, test_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None, class_num=1):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if opts.dataset.lower() == 'cityscapes':
                labels = labels + 1
            labels[labels != class_num] = 0  # Get the mask for single class
            # labels[labels == class_num] = 1
            # labels = (labels==class_num).float()
            # mask = labels.detach().cpu().numpy()
            outputs = nn.Sigmoid()(model(images))
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            preds = outputs.detach().cpu().numpy()
            preds[(preds >= 0.5)] = class_num
            preds[(preds < 0.5)] = 0
            preds = preds.astype(int)

            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def test_multiple(opts, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            preds = []
            for class_num in range(1, opts.num_classes):
                modelname = 'checkpoints/multiple_model/best_%s_%s_class%d_os%d.pth' % (
                    opts.model, opts.dataset, class_num, opts.output_stride,)
                checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
                model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
                model.load_state_dict(checkpoint["model_state"])
                model = nn.DataParallel(model)
                model.to(device)
                model.eval()

                outputs = nn.Sigmoid()(model(images))
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

            modelname = opts.sin_model
            checkpoint = torch.load(modelname, map_location=torch.device('cpu'))

            if opts.dataset.lower() == 'cityscapes':
                Nclass = 19
            else:
                Nclass = opts.num_classes
            model = model_map[opts.model.split('_')[0] + '_' + opts.model.split('_')[1]](num_classes=Nclass,
                                                                                         output_stride=opts.output_stride)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            model.eval()
            # sin_outputs = model(images).cpu().numpy()
            sin_outputs = F.softmax(model(images), dim=1).cpu().numpy()
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            # 方案一
            # preds = np.concatenate(preds, 1)                            # 在级联
            # preds_max_index = np.argmax(preds, axis=1) + 1              # 取概率最大的像素点的index, 由于index从零开始，需要+1
            # index= np.unravel_index(np.argmax(preds, axis=0), preds.shape)
            # max_score = np.amax(preds, axis=1)                           # 取概率最大的像素点的像素值 到max_score
            # mask = (max_score >= 0.5).astype(int)                       # 像素值 >0.5的预测才保留
            # preds = np.multiply(preds_max_index, mask)                   # 像素值 >0.5的预测才保留，否则变成0

            # 方案二
            # alpha=0.5
            alpha = opts.alpha
            preds = np.concatenate(preds, 1)
            if opts.dataset.lower() != 'cityscapes':
                background_p = np.expand_dims(sin_outputs[:, 0, :, :], axis=1)  # 抽取单类模型预测的背景概率
                preds = np.concatenate((background_p, preds), 1)  # 单类模型与多雷模型分数融合
            final_preds = alpha * preds + sin_outputs
            # final_preds = final_preds.max(dim=1)[1].cpu().numpy()
            preds = np.argmax(final_preds, axis=1)
            # backgroud_probability=sin_outputs[0,:,:,:]
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        score = metrics.get_results()
    return score, ret_samples


def test_multiple_fuse_specific(opts, loader, device, metrics, ret_samples_ids=None):
    specifc_class_list = [int(i) for i in opts.select_class]  # the list of class score to fuse, start index==1
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # 单模型输出
            modelname = opts.sin_model
            checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
            model = model_map[opts.model.split('_')[0] + '_' + opts.model.split('_')[1]](num_classes=opts.num_classes,
                                                                                         output_stride=opts.output_stride)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            model.eval()
            sin_outputs = F.softmax(model(images), dim=1).cpu().numpy()

            # 多模型输出
            preds = []
            for class_num in range(1, opts.num_classes):
                if class_num in specifc_class_list:  # 判断该类是否在指定的要融合的类中
                    modelname = 'checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' % (
                        opts.model, opts.dataset, class_num, opts.output_stride,)
                    checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
                    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
                    model.load_state_dict(checkpoint["model_state"])
                    model = nn.DataParallel(model)
                    model.to(device)
                    model.eval()

                    outputs = nn.Sigmoid()(model(images))
                    pred = outputs.detach().cpu().numpy()
                else:
                    pred = np.zeros(np.expand_dims(sin_outputs[:, 0, :, :], axis=1).shape)  # 不在类中，该张概率图的所有值为0
                    # pred = 0.5*np.ones(np.expand_dims(sin_outputs[:, 0, :, :], axis=1).shape)  # 不在类中，该张概率图的所有值为0.5
                preds.append(pred)

            alpha = opts.alpha
            preds = np.concatenate(preds, 1)
            background_p = np.expand_dims(sin_outputs[:, 0, :, :], axis=1)  # 抽取单类模型预测的背景概率
            preds = np.concatenate((background_p, preds), 1)  # 单类模型与多雷模型分数融合
            final_preds = alpha * preds + sin_outputs
            # final_preds = final_preds.max(dim=1)[1].cpu().numpy()
            preds = np.argmax(final_preds, axis=1)
            # backgroud_probability=sin_outputs[0,:,:,:]
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        score = metrics.get_results()  # score = metrics.get_results().each_get_results()

    return score, ret_samples


def test_multiple_softmax(opts, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            preds = []
            modelname = 'checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth'
            checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
            model = model_map[opts.model.split('_')[0] + '_' + opts.model.split('_')[1]](num_classes=opts.num_classes,
                                                                                         output_stride=opts.output_stride)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            model.eval()

            sin_outputs = F.softmax(model(images), dim=1).cpu().numpy()
            sin_preds = sin_outputs.detach().max(dim=1)[1].cpu()  # get the clss of max probability
            background_p = np.expand_dims(sin_outputs[:, 0, :, :], axis=1)
            # preds.append(background_p)
            pred = np.zeros(background_p.shape)

            for class_num in range(1, opts.num_classes):
                if class_num in sin_preds:
                    # print("detect %d class in one image"%class_num)
                    modelname = 'checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' % (
                        opts.model, opts.dataset, class_num, opts.output_stride,)
                    checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
                    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
                    model.load_state_dict(checkpoint["model_state"])
                    model = nn.DataParallel(model)
                    model.to(device)
                    model.eval()

                    outputs = nn.Sigmoid()(model(images))
                    pred = outputs.detach().cpu().numpy()
                else:
                    pred = np.zeros(background_p.shape)
                preds.append(pred)

            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            # 方案一
            preds = np.concatenate(preds, 1)  # 在级联
            # background_p=np.concatenate([background_p,preds], 1)
            preds = np.argmax(preds, axis=1)  # 取概率最大的像素点的index
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        score = metrics.get_results()
    return score, ret_samples


def test_single(opts, loader, device, metrics, ret_samples_ids=None):
    for class_num in range(opts.start_class, opts.num_classes):
        train_dst, val_dst, test_dst = get_dataset_multiple(opts, class_num)
        train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        test_loader = data.DataLoader(test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)

        metrics.reset()
        ret_samples = []
        if opts.save_val_results:
            if not os.path.exists('results'):
                os.mkdir('results')
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            img_id = 0

        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(val_loader)):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                labels[labels != class_num] = 0
                # labels[labels == class_num] = 1
                # mask = labels.detach().cpu().numpy()
                # labels = (labels == class_num).float()
                modelname = 'checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' % (
                    opts.model, opts.dataset, class_num, opts.output_stride,)
                checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
                model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
                model.load_state_dict(checkpoint["model_state"])
                model = nn.DataParallel(model)
                model.to(device)
                model.eval()

                outputs = model(images)
                outputs = nn.Sigmoid()(outputs)

                preds = outputs.squeeze().detach().cpu().numpy()
                preds[(preds >= 0.5)] = class_num
                preds[(preds < 0.5)] = 0
                preds = preds.astype(int)
                # preds[index]
                # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)
                if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                    ret_samples.append(
                        (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                if opts.save_val_results:
                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)

                        Image.fromarray(image).save('results/%d_image.png' % img_id)
                        Image.fromarray(target).save('results/%d_target.png' % img_id)
                        Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                        fig = plt.figure()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.7)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        img_id += 1
            score = metrics.get_results()
            print('class_num: %d %s ' % (class_num, metrics.to_str(score)))
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21                  # voc20只用训练20个类,还有一个背景类,metric就要测试 1-20
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    if opts.dataset.lower() == 'cityscapes':  # 因为cityscapes没有背景类，实际上要训练19个类,,所以metric就要测试1-19
        opts.num_classes = 19 + 1
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)
    metrics = StreamSegMetrics(opts.num_classes)
    # Set up optimizer
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'logit':
        criterion = nn.BCELoss(reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None:
        print("Error --ckpt, can't read model")
        return

    _, val_dst, test_dst = get_dataset(opts)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    vis_sample_id = np.random.randint(0, len(test_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images



    # ==========   Test Loop   ==========#

    if opts.test_only:
        print("Dataset: %s,  Val set: %d, Test set: %d" %
              (opts.dataset, len(val_dst), len(test_dst)))

        print("val")
        if opts.select_class:
            test_score, ret_samples = test_multiple_fuse_specific(
                opts=opts, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        else:
            test_score, ret_samples = test_multiple(
                opts=opts, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(test_score))
        # test_score, ret_samples = test_single(opts=opts,
        #                                       loader=test_loader, device=device, metrics=metrics,
        #                                       ret_samples_ids=vis_sample_id)
        print("test")
        if opts.select_class:
            test_score, ret_samples = test_multiple_fuse_specific(
                opts=opts, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        else:
            test_score, ret_samples = test_multiple(
                opts=opts, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(test_score))
        return

    # ==========   Train Loop   ==========#
    utils.mkdir('checkpoints/multiple_model')

    for class_num in range(opts.start_class, opts.num_classes):  # 1-19
        # ==========   Dataset   ==========#
        number_works = 2
        train_dst, val_dst, test_dst = get_dataset_multiple(opts, class_num)
        num_train = len(train_dst) // number_works
        mod_train = len(train_dst) % number_works
        if num_train % opts.batch_size == 1 or (num_train + mod_train) % opts.batch_size == 1:  # 预防batch里只有一个样本
            droplast = True
        else:
            droplast = False
        train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=number_works,
                                       drop_last=droplast)
        # if opts.dataset.lower() == 'cityscapes' and class_num==17:  #有个类被划分后的验证集没有数据
        # if len(val_dst) == 0:  # 如果有个类被划分后的验证集没有数据
        #     val_dst = train_dst
        num_val = len(val_dst) // number_works
        mod_val = len(val_dst) % number_works
        if num_val % opts.val_batch_size == 1 or (num_val + mod_val) % opts.val_batch_size == 1:
            droplast = True
        else:
            droplast = False
        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=number_works,
                                     drop_last=droplast)
        test_loader = data.DataLoader(test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        print("Dataset: %s Class %d, Train set: %d, Val set: %d, Test set: %d" % (
            opts.dataset, class_num, len(train_dst), len(val_dst), len(test_dst)))

        # ==========   Model   ==========#
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

        # ==========   Params and learning rate   ==========#
        params_list = [
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': 0.1 * opts.lr}  # opts.lr
        ]
        if 'SA' in opts.model:
            params_list.append({'params': model.attention.parameters(), 'lr': 0.1 * opts.lr})
        optimizer = torch.optim.Adam(params=params_list, lr=opts.lr, weight_decay=opts.weight_decay)

        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

        model = nn.DataParallel(model)
        model.to(device)

        best_score = 0.0
        cur_itrs = 0
        cur_epochs = 0

        interval_loss = 0
        while True:  # cur_itrs < opts.total_itrs:
            # =====  Train  =====
            model.train()

            cur_epochs += 1
            for (images, labels) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                # x=labels.detach().cpu().numpy()
                # labels = labels.to(device, dtype=torch.float322
                if opts.dataset.lower() == 'cityscapes':
                    labels = labels + 1
                labels = (labels == class_num).float()
                # mask=labels.detach().cpu().numpy()
                # labels[labels == class_num] = 1
                # labels[labels != class_num] = 0  # Get the mask for single class
                # a=labels.detach().cpu().numpy()
                # labels = labels.float()
                # b=labels.detach().cpu().numpy()
                optimizer.zero_grad()
                outputs = model(images)
                outputs = nn.Sigmoid()(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    interval_loss = 0.0

                if (cur_itrs) % opts.val_interval == 0:
                    save_ckpt('checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' %
                              (opts.model, opts.dataset, class_num, opts.output_stride,))
                    print("validation...")
                    model.eval()
                    val_score, ret_samples = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id, class_num=class_num)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save_ckpt('checkpoints/multiple_model/best_%s_%s_class%d_os%d.pth' %
                                  (opts.model, opts.dataset, class_num, opts.output_stride))

                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)
                    model.train()

                scheduler.step()

                if cur_itrs >= opts.total_itrs:
                    save_ckpt('checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' %
                              (opts.model, opts.dataset, class_num, opts.output_stride,))
                    print("Saving..")
                    break
            if cur_itrs >= opts.total_itrs:
                cur_itrs = 0
                break

        print("Model of class %d is trained and saved " % (class_num))


if __name__ == '__main__':
    main()
