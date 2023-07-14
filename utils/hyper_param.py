from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    # training hyper parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--ds-path', type=str, default='./data/cityscapes',
                        help='dataset path (default: ./data/cityscapes)')
    # checkpoints
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models (default: ./weights)')
    return parser.parse_args()
