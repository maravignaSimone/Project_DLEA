from argparse import ArgumentParser

# define all the parser to execute with custom parameters and custom paths (useful to test both in local and hpc)
def parse_args():
    parser = ArgumentParser()
    # training hyper parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate (default: 1e-2)')
    # path
    parser.add_argument('--ds-path', type=str, default='/hpc/home/simone.maravigna/Project_DLEA/data/cityscapes', # runnning locally './data/cityscapes' in hpc '/hpc/home/simone.maravigna/Project_DLEA/data/cityscapes'
                        help='dataset path (default: ./data/cityscapes)')
    parser.add_argument('--runs_folder', type=str, default='/hpc/home/simone.maravigna/Project_DLEA/runs', # runnning locally './runs' in hpc '/hpc/home/simone.maravigna/Project_DLEA/runs'
                        help='Tensorboard path (default: ./runs)')
    # checkpoints
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-weights', default='/hpc/home/simone.maravigna/Project_DLEA/weights', # runnning locally './weights' in hpc '/hpc/home/simone.maravigna/Project_DLEA/weights'
                        help='directory for saving checkpoint models (default: ./weights)')
    # eval
    parser.add_argument('--output-folder', type=str, default='/hpc/home/simone.maravigna/Project_DLEA/output_data', # runnning locally './output_data' in hpc '/hpc/home/simone.maravigna/Project_DLEA/output_data'
                        help='put the path to save output (default: ./output_data)')
    return parser.parse_args()
