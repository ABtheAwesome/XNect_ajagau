from argparse import ArgumentParser

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
        '--imagenet_base_path', type=str, default='./data', metavar='FILE',
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