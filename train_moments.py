'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import json
import socket
import logging
import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_model import train_model
from network.symbol_builder import get_symbol

# Create main parser
parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
parser.add_argument('--dataset', default='Moments',
                    help="path to dataset")
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")
parser.add_argument('--clip-size', default=284,
                    help="define the size of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=[3,4],
                    help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=3,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='r3d_50',
                    help="chose the base network")
'''
Initialisation steps :
- step 1: random initialize
- step 2: load the 3D pretrained model if `pretrained_3d' is defined
- step 3: resume if `resume_epoch' >= 0
'''


parser.add_argument('--pretrained_3d', type=str,  default=None,
                    help="load default 3D pretrained model.")
# optimization
parser.add_argument('--fine-tune', type=bool, default=False,
                    help="resume training and then fine tune the classifier")
parser.add_argument('--resume-epoch', type=int, default=-1,
                    help="resume train")
parser.add_argument('--batch-size', type=int, default=48,
                    help="batch size")
parser.add_argument('--long-cycles', type=bool, default=True,
                    help="Enebling long cycles for batches")
parser.add_argument('--short-cycles', type=bool, default=True,
                    help="Enebling short cycles for batches")
parser.add_argument('--lr-base', type=float, default=0.1,
                    help="learning rate")
parser.add_argument('--lr-steps', type=list, default=[25,50,70],
                    help="number of samples to pass before changing learning rate")
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=1,
                    help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=110,
                    help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')
#distributed
parser.add_argument('--world_size', type=int, default=4,
                    help="number of distributed processes")
parser.add_argument('--rank', type=int, default=1,
                    help="rank of processes based on `world_size` used (0,..,`world_size`-1)")

'''
---  S T A R T  O F  F U N C T I O N  A U T O F I L L  ---
    [About]
        Function for creating log directories based on the parser arguments
    [Args]
        - args: ArgumentParser object containg both the name of task (if empty a default folder is created) and the log file to be created.
    [Returns]
        - args: ArgumentParser object with additionally including the model directory and the model prefix.
'''
def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    if not args.log_file:
        if os.path.exists("./exps/logs"):
            args.log_file = "./exps/logs/{}_at-{}.log".format(args.task_name, socket.gethostname())
        else:
            args.log_file = ".{}_at-{}.log".format(args.task_name, socket.gethostname())
    # fixed
    args.model_dir = os.path.join(args.model_dir,args.network)
    args.model_prefix = os.path.join(args.model_dir, args.network)

    return args
'''
---  E N D  O F  F U N C T I O N  A U T O F I L L  ---
'''

'''
---  S T A R T  O F  F U N C T I O N  S E T _ L O G G E R  ---
    [About]
        Function for creating logger for displaying and storing all events.
    [Args]
        - log_file: String for the log file to store all session info through logging.
        - debug_mode: Boolean for additional information while logging.
    [Returns]
        - None
'''
def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    '''add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file'''
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

'''
---  E N D  O F  F U N C T I O N  S E T _ L O G G E R  ---
'''

'''
---  S T A R T  O F  M A I N  F U N C T I O N  ---
'''
if __name__ == "__main__":

    # set args
    args = parser.parse_args()
    args = autofill(args)

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    print('Can use cuda: '+str(torch.cuda.is_available()))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    # load dataset related configuration<
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model with all parameters initialized
    assert (not args.fine_tune or not args.resume_epoch < 0), \
            "args: `resume_epoch' must be defined for fine tuning"
    net, input_conf = get_symbol(name=args.network,
                     print_net=False, # True if args.distributed else False,
                     **dataset_cfg)

    # training
    kwargs = {}
    kwargs.update(dataset_cfg)
    kwargs.update({'input_conf': input_conf})
    kwargs.update(vars(args))
    train_model(sym_net=net, name='MOMENTS', net_name=args.network, dataset_location=args.dataset, **kwargs)
'''
---  E N D  O F  M A I N  F U N C T I O N  ---
'''
