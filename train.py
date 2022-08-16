#Part of code on this page taken from https://github.com/facebookresearch/detr
import argparse
import datetime
import json
import logging
import os
import sys
import random
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from utils.data.cornell_data import CornellDataset
from models.resnet import resnet56, resnet20
import utils.misc as utils
from utils.visualisation import plot
import matplotlib.pyplot as plt

from models.backbone import build_backbone
from models import build_model
from utils.dataset_processing import grasp, image
from utils.box_ops import get_coordinated_fron_cxcy_theta, final_iou

import torch
import torch.nn.functional as F

torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    """
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    """
    """
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=600,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    """
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--include_depth', type=int, default=600,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--batches-per-epoch', type=int, default=600,
                        help='Batches per Epoch')
    parser.add_argument('--duplication_factor', type=int, default=3,
                        help='For each image how many duplicates we need to create using randomization')
    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    # getting some args from DETR paper for this to work

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    """
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=128, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=32, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    """
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=128, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=1, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #Experiment Setup
    parser.add_argument('--experiment', default=0, type=int,
                        help='which setup will be use for this experiment')
    args = parser.parse_args()

    experiments = []
    experiment_1 = {}
    experiment_1["random_rotate"] = True
    experiment_1["random_zoom"] = False
    experiment_1["include_depth"] = True
    experiments.append(experiment_1)

    experiment_2 = {}
    experiment_2["random_rotate"] = True
    experiment_2["random_zoom"] = True
    experiment_2["include_depth"] = True
    experiments.append(experiment_2)

    experiment_3 = {}
    experiment_2["random_rotate"] = True
    experiment_2["random_zoom"] = False
    experiment_2["include_depth"] = False
    experiments.append(experiment_3)

    if args.experiment == 0:
        exp_to_run = experiments[0]
    elif args.experiment == 1:
        exp_to_run = experiments[1]
    else:
        exp_to_run = experiments[0]
    return args, exp_to_run


def validate(net, device, val_data, iou_threshold, criterion, dataset, experiment):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """

    net.eval()
    criterion.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {
        }
    }

    ld = len(val_data)
    save_img = 0
    cornell_data = dataset

    with torch.no_grad():
        for x, y in val_data:
            yc = [{k: v[0].to(device) for k, v in y.items()}]
            xc = x.to(device)

            # Run through the model and evaluate the loss
            out = net(xc)
            loss_dict, iou, indices = criterion(out, yc)

            # add the losses to results
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            results['loss'] += loss

            # the augmentations used to generate the image
            img_id = y['image_id'][0]
            rot = y['rot'][0]
            zoom = y['zoom'][0]
            shift_x = y['shift_x'][0]
            shift_y = y['shift_y'][0]

            # output from the model
            logits = out['pred_logits']
            grsps = []
            prob = F.softmax( logits, -1)
            scores, labels = prob[..., :-1].max(-1)
            box = out['pred_boxes'][0]
            angles = out['pred_angles'][0]

            # Convert the output grasps into rectangular coordinates, and find the best grasp
            max_score = 0
            best_index = -1
            for i in range(box.shape[0]):
                #print()

                if labels[0][i] == 0:
                    boxes1 = box[i, :]
                    
                    angle = angles[i, :]
                    boxes1 = get_coordinated_fron_cxcy_theta(boxes1, angle)
                    boxes1 = [torch.round(num) for num in boxes1]
                    grsp = [
                                [boxes1[0].item() , boxes1[1].item() ],
                                [boxes1[2].item() , boxes1[3].item() ],
                                [boxes1[4].item() , boxes1[5].item() ],
                                [boxes1[6].item() , boxes1[7].item() ]
                            ]


                    grsps.append(grsp)
                    if scores[0][i] > max_score:
                        max_score = scores[0][i]
                        best_index = i 
                        recommended_graps = [grsp]    
            
            #Print all the grasps into a folder , to save space we are only saving one out of 10 images
            if best_index > -1 and random.randint(1,10) == 5:
                image = cornell_data.get_rgb_img(img_id.item(), rot.item(), zoom.item())
                grasps = grasp.GraspRectangles.load_from_array(np.array(grsps))
                image.offset_img(shift_x.item() , shift_y.item() )
                image.img = image.img.transpose(1, 2, 0)

                fig, ax = plt.subplots()
                image.show(ax)
                grasps.show(ax)
                plt.savefig("output/experiment_{}_image_{}.png".format(experiment, img_id))

                grasps = grasp.GraspRectangles.load_from_array(np.array(recommended_graps))
                
                for gr in grasps:
                    print(gr)
                fig, ax = plt.subplots()
                image.show(ax)
                grasps.show(ax)
                plt.savefig("output_single/experiment_{}_image_{}.png".format(experiment, img_id))

            max_iou, iou_list, final_angle = final_iou(box[ best_index, : ], yc[0]["boxes"], angles[best_index, :] , yc[0]["angles"] )

            if max_iou > 0.3 and  abs( angles[best_index, :]  - final_angle ) < 0.175:
                results["correct"] += 1
            else:
                results["failed"] += 1
    

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, criterion, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()
    criterion.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    for x, y in train_data:
        yc = [{k: v.to(device) for k, v in t.items()} for t in y]
        xc = x.to(device)
        backbone = net(xc)
        loss_dict, iou, indices = criterion(backbone, yc)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if batch_idx % 100 == 0:
            logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

        results['loss'] += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), "checkpoint_test_large_8_heads.pth")
    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args, experiment = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CornellDataset('./cornell-dataset/',
                             ds_rotate=args.ds_rotate,
                             random_rotate=experiment['random_rotate'],
                             random_zoom= experiment['random_zoom'],
                             include_depth=experiment['include_depth'],
                             include_rgb=1 ,
                             duplication_factor=args.duplication_factor )

    print('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]
    print('Training size: {}'.format(len(train_indices)))
    print('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler_train, 
        num_workers=args.num_workers,
        #sampler=train_sampler,
        collate_fn=utils.collate_fn
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')

    net, criterion, postprocessors = build_model(args)
    net = net.to(device)
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # Print model architecture.
    """"
    #summary(net, (input_channels, args.input_size, args.input_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, args.input_size, args.input_size))
    sys.stdout = sys.__stdout__
    f.close()

    """

    best_iou = 0.0
    print("number of epochs", args.epochs)
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, criterion,
                              vis=args.vis)

        # Log training losses to tensorboard
        #tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # or n, l in train_results['losses'].items():
        #    tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold,criterion, dataset, args.experiment )
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        #tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        #tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        #or n, l in test_results['losses'].items():
        #    tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        #iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        #if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        #    torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
        #    best_iou = iou

if __name__ == '__main__':
    run()