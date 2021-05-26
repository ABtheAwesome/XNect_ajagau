from argparse import ArgumentParser
import importlib

import matplotlib
#print(matplotlib.get_backend())
#matplotlib.rcParams["backend"] = "TkAgg"
#print(matplotlib.get_backend())
import matplotlib.pyplot as plt
#plt.switch_backend("TkAgg")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as uti
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

from PIL import Image as im

import models.selecsls
from models.selecsls import SelecSLSBlock as stage1

print("Start detecting body pose on one image")

def imsave(img, i, name):
     img = img / 2 + 0.5
     npimg = img.numpy()
     trpimg = np.transpose(npimg, (1, 2, 0))
     print("Type of Image: " + str(type(trpimg)))
     new_img = im.fromarray(trpimg, 'RGB')
     print("Saving Image " + name + str(i) + ".png .....")
     new_img.save(name + str(i) + ".png")

#Argumente werden festgelegt, u.a. welches Model genutzt wird und welche Datenbank.
def opts_parser():
    usage = 'Configure the dataset using image from ./data'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--model_class', type=str, default='selecsls', metavar='FILE',
        help='Select model type to use (DenseNet, SelecSLS, ResNet etc.)')
    parser.add_argument(
        '--model_config', type=str, default='SelecSLS60', metavar='NET_CONFIG',
        help='Select the model configuration')
    parser.add_argument(
        '--model_weights', type=str, default='./weights/SelecSLS60_statedict.pth', metavar='FILE',
        help='Path to model weights')
    parser.add_argument(
        '--dataset_base_path', type=str, default='../MultiPersonTestSet', metavar='FILE',
        help='Path to dataset')
    parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='Which GPU to use.')
    parser.add_argument(
        '--simulate_pruning', type=bool, default=False,
        help='Whether to zero out features with gamma below a certain threshold')
    parser.add_argument(
        '--pruned_and_fused', type=bool, default=False,
        help='Whether to prune based on gamma below a certain threshold and fuse BN')
    parser.add_argument(
        '--gamma_thresh', type=float, default=1e-4,
        help='gamma threshold to use for simulating pruning')
    return parser


def start_recognizing_body_pose(model_class, model_config, model_weights, dataset_base_path, gpu_id, simulate_pruning, pruned_and_fused, gamma_thresh):
    print("Starting to recognize body pose (detect_body_pose.py line 47)")

    model_module = importlib.import_module('models.'+model_class)
    net = model_module.Net(nClasses=1000, config=model_config)
    net.load_state_dict(torch.load(model_weights, map_location= lambda storage, loc: storage))

    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    if pruned_and_fused:
        print('Fusing BN and pruning channels based on gamma ' + str(gamma_thresh))
        net.prune_and_fuse(gamma_thresh)

    if simulate_pruning:
        print('Simulating pruning by zeroing all features with gamma less than '+str(gamma_thresh))
        with torch.no_grad():
            for n, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight[abs(m.weight) < gamma_thresh] = 0
                    m.bias[abs(m.weight) < gamma_thresh] = 0

    # defines transformation of images (so every image has the same size etc) 
    # also images get transformed to PyTorch tensors
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #norm_transform
    ])


    print("Loading local dataset (detect_body_pose.py line 77) in" + str(dataset_base_path))
    # loading dataset and transforming it
    dataset = dset.ImageFolder
    test_set = dataset(dataset_base_path,transform=transform)
   

    i = 0
    for img, label in test_set:
        print("--------------Image " + str(i) + " in testset----------------")
        #print("Shape: " + str(img.shape) + ", Type: " + str(type(img)) + ", Label: " + str(label))
        imsave(img, i, "test_set_")
        print(img)
        i += 1
        if i>4:
            break


    test_loader = DataLoader(test_set, batch_size=1,
                              shuffle=False, num_workers=2, pin_memory=True)
    print("Loaded dataset as type of: " + str(type(test_loader)))
    i = 0
    with torch.no_grad():
        for model_input, label in test_loader:
            print("--------------Image: " + str(i) + " in testloader-----------------")
            #print("Shape: " + str(model_input.shape) + ", Label: " + str(label) + ", Type: " + str(type(label)))            
            pred = F.log_softmax(net(model_input.to(device)), dim=1)
            print("Shape of prediction is " + str(pred.shape) + "with type" + str(type(pred)))
            #print(pred)
            i += 1
            if i>4:
                break
            #pred1 = stage1.forward(stage1, t)




def main():
    parser = opts_parser()
    args = parser.parse_args()

    start_recognizing_body_pose(**vars(args))

if __name__ == '__main__':
    main()