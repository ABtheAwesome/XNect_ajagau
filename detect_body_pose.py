from argparse import ArgumentParser
import importlib

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

print("Start detecting body pose on one image")

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
        '--dataset_base_path', type=str, default='./data', metavar='FILE',
        help='Path to ImageNet dataset')
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
        norm_transform
    ])

    print(f"Loading local dataset (detect_body_pose.py line 77) in {dataset_base_path}")
    # loading dataset and transforming it
    dataset = dset.ImageFolder(dataset_base_path, transform=transform)
    kwargs = {'num_workers': 8, 'pin_memory': True}
    test_loader = DataLoader(dataset, batch_size=1,
                              shuffle=True, **kwargs)

    print(f"The type of the dataset is: {type(test_loader)}")
    print(np.shape(test_loader))
    plt.imshow(test_loader.numpy()[0], cmap='gray')




def main():
    parser = opts_parser()
    args = parser.parse_args()

    start_recognizing_body_pose(**vars(args))

if __name__ == '__main__':
    main()