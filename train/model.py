'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import gc
import csv
import time
import coloredlogs, logging
coloredlogs.install()
import math
import torch
from . import metric
from . import callback

from apex import amp


'''
===  S T A R T  O F  C L A S S  C S V W R I T E R ===

    [About]

        Class for creating a writer for the training and validations scores and saving them to .csv

    [Init Args]

        - filename: String for the name of the file to save to. Will create the file or erase any
        previous versions of it.

    [Methods]

        - __init__ : Class initialiser
        - close : Function for closing the file.
        - write : Function for writing new elements along in a new row.
        - size : Function for getting the size of the file.
        - fname : Function for returning the filename.

'''
class CSVWriter():

    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename
'''
===  E N D  O F  C L A S S  C S V W R I T E R ===
'''


'''
===  S T A R T  O F  C L A S S  S T A T I C M O D E L ===

    [About]

        Class responsible for the main functionality during training. Provides function implementations
        for loading a previous model state, create a checkpoint filepath, loading a checkpoint, saving model
        based on the checkpoint path created as well as preform a full forward pass returning the output
        class probabilities and loss(es).

    [Init Args]

        - net: nn.Module containing the full architecture.
        - criterion : nn.Module that specifies the loss criterion (e.g. CrossEntropyLoss). Could also
        include custom losses.
        - model_prefix : String for the prefix to be used when loading a previous state.

    [Methods]

        - __init__ : Class initialiser
        - load_state :
        - get_checkpoint_path :
        - load_checkpoint :
        - save_checkpoint :
        - forward :

'''
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # parameter initialisation
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion

    def load_state(self, state_dict, strict=False):
        # Strict mode that structure should match exactly
        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        # Less strict parameter loading (e.g. fine tuning or adding/removing layers)
        else:
            # customised partialy load function
            net_state_keys = list(self.net.state_dict().keys())
            for name, param in state_dict.items():
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch):
        # model prefix needs to be defined for creating the checkpoint path
        assert self.model_prefix, "Undefined `model_prefix`"
        # checkpoint creation
        checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
        return checkpoint_path

    def load_checkpoint(self, epoch, optimiser=None):
        # Load path
        load_path = self.get_checkpoint_path(epoch)
        assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)
        # checkpoint loading
        checkpoint = torch.load(load_path)
        # Get parametrs that match based on the less strict `load_state`
        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)
        # Optimiser handling
        if optimiser:
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                optimiser.load_state_dict(checkpoint['optimizer'])
                logging.info("Model & Optimiser states are resumed from: `{}'".format(load_path))
            else:
                logging.warning("Failed to load optimiser state from: `{}'".format(load_path))
        else:
            logging.info("Only model state resumed from: `{}'".format(load_path))

        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning("Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, base_directory, optimiser_state=None):
        # Create save path
        save_path = os.path.join(base_directory,'{}_ep-{:04d}.pth'.format(base_directory.split('/')[-1],epoch))
        # Create directory if path does not exist
        if not os.path.exists(base_directory):
            logging.debug("mkdir {}".format(base_directory))
            os.makedirs(base_directory)
        # Create optimiser state if it does not exist. Use the `epoch` and `state_dict`
        if not optimiser_state:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict()},
                        save_path)
            logging.debug("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict(),
                        'optimizer': optimiser_state},
                        save_path)
            logging.debug("Checkpoint (model & optimiser) saved to: {}".format(save_path))


    def forward(self, data, target):
        # Data conversion
        data = data.float().cuda()
        target = target.cuda()

        # Create autograd Variables
        input_var = torch.autograd.Variable(data)
        target_var = torch.autograd.Variable(target)

        # Forward for training/evaluation
        if self.net.training:
            output = self.net(input_var)
        else:
            with torch.no_grad():
                output = self.net(input_var)

        # Use (loss) criterion if specified
        if hasattr(self, 'criterion') and self.criterion is not None and target is not None:
            loss = self.criterion(output, target_var)
        else:
            loss = None
        return [output], [loss]
'''
===  E N D  O F  C L A S S  S T A T I C M O D E L ===
'''


'''
===  S T A R T  O F  C L A S S  M O D E L ===

    [About]

        Class for performing the main dataloading and weight updates. Train functionality happens here.

    [Init Args]

        - net: nn.Module containing the full architecture.
        - criterion : nn.Module that specifies the loss criterion (e.g. CrossEntropyLoss). Could also
        include custom losses.
        - model_prefix : String for the prefix to be used when loading a previous state.
        - step_callback: CallbackList for including all Callbacks created.
        - step_callback_freq: Frequency based on which the Callbacks are updates (and values are logged).
        - save_checkpoint_freq: Integer for the frequency based on which the model is to be saved.
        - opt_batch_size: Integer defines the original batch size to be used.


    [Methods]

        - __init__ : Class initialiser
        - step_end_callback: Function for updating the Callbacks list at the end of each iteration step. In the case of validation,
        this function is called at the end of evaluating.
        - epoch_end_callback: Function for updating the Callbacks at the end of each epoch.
        - adjust_learning_rate: Function for adjusting the learning rate based on the iteration/epoch. Primarily used for circles.
        - fit: Main training loop. Performs both training and evaluation.

'''
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'read_elapse': None,
                                'forward_elapse': None,
                                'backward_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimiser_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size


    def step_end_callback(self):
        # logging.debug("Step {} finished!".format(self.i_step))
        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self,results_directory):
        if self.callback_kwargs['epoch_elapse'] is not None:
            # Main logging definition
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimiser_state=self.callback_kwargs['optimiser_dict'],
                                 base_directory=results_directory)


    def adjust_learning_rate(self, lr, optimiser):
        # learning rate adjustment based on provided lr rate
        for param_group in optimiser.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult


    def fit(self, train_iter, optimiser, lr_scheduler, long_short_steps_dir,
            no_cycles,
            eval_iter=None,
            batch_shape=(24,16,224,224),
            workers=24,
            metrics=metric.Accuracy(topk=1),
            iter_per_epoch=1000,
            epoch_start=0,
            epoch_end=10000,
            directory=None,
            **kwargs):

        # Check kwargs used
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"


        pause_sec = 0.
        train_loaders = {}
        active_batches = {}

        workers= 8

        # Create files to write results to
        if (directory is not None):
            train_file = os.path.join(directory,'train_results.csv')
        else:
            train_file = './train_results.csv'
        train_writer = CSVWriter(train_file)
        train_writer.write(('Epoch', 'Top1', 'Top5','Loss'))

        if (directory is not None):
            val_file = os.path.join(directory,'val_results.csv')
        else:
            val_file = './val_results.csv'
        val_writer = CSVWriter(val_file)
        #val_writer = csv.writer(f_val, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        val_writer.write(('Epoch', 'Top1', 'Top5','Loss'))

        cycles = True


        for i_epoch in range(epoch_start, epoch_end):

            if (no_cycles):
                logging.info('No cycles selected')
                train_loader = iter(torch.utils.data.DataLoader(train_iter,batch_size=batch_shape[0], shuffle=True,num_workers=workers, pin_memory=False))
                cycles = False

            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()

            # values for writing average topk,loss in epoch
            train_top1_sum = []
            train_top5_sum = []
            train_loss_sum = []
            val_top1_sum = []
            val_top5_sum = []
            val_loss_sum = []

            # Reset all metrics
            metrics.reset()
            # change network `mode` to training to ensure weight updates.
            self.net.train()
            # Time variable definitions
            sum_sample_inst = 0
            sum_read_elapse = 0.
            sum_forward_elapse = 0
            sum_backward_elapse = 0
            epoch_speeds = [0,0,0]
            batch_start_time = time.time()
            logging.debug("Start epoch {:d}:".format(i_epoch))
            for i_batch in range(iter_per_epoch):

                b = batch_shape[0]
                t = batch_shape[1]
                h = batch_shape[2]
                w = batch_shape[3]

                selection = ''

                loader_id = 0
                # Increment long cycle steps (8*B).
                if i_batch in long_short_steps_dir['long_0']:
                    b = 8 * b
                    t = t//4
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment long cycle steps (4*B).
                elif i_batch in long_short_steps_dir['long_1']:
                    b = 4 * b
                    t = t//2
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment long cycle steps (2*B).
                elif i_batch in long_short_steps_dir['long_2']:
                    b = 2 * b
                    t = t//2

                # Increment short cycle steps (2*b).
                if i_batch in long_short_steps_dir['short_1']:
                    loader_id = 1
                    b = 2 * b
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment short cycle steps (4*b).
                elif i_batch in long_short_steps_dir['short_2']:
                    loader_id = 2
                    b = 4 * b
                    h = h//2
                    w = w//2

                if (h%2 != 0): h+=1
                if (w%2 != 0): w+=1

                if cycles:
                    train_iter.size_setter(new_size=(t,h,w))

                    batch_s = (t,h,w)
                    if (batch_s not in active_batches.values()):
                        logging.info('Creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        # Ensure rendomisation
                        train_iter.shuffle(i_epoch+i_batch)
                        # create dataloader corresponding to the created dataset.
                        train_loader = torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=workers, pin_memory=False)
                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        train_loaders[loader_id]=iter(train_loader)
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        active_batches[loader_id]=batch_s
                        gc.collect()


                    try:
                        if data is not None:
                            del data
                        if target is not None:
                            del target
                        gc.collect()
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse
                    except:
                        # Reinitialise if used up
                        logging.warning('Re-creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        # Ensure rendomisation
                        train_iter.shuffle(i_epoch+i_batch)
                        train_loader = torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=workers, pin_memory=False)

                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        train_loaders[loader_id]=iter(train_loader)
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        active_batches[loader_id]=batch_s
                        gc.collect()
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse


                    gc.collect()
                else:
                    sum_read_elapse = time.time()
                    data,target = next(train_loader)
                    sum_read_elapse = time.time() - sum_read_elapse

                self.callback_kwargs['batch'] = i_batch

                # Catch Segmentation fault errors
                while True:
                    try:
                        # [forward] making next step
                        torch.cuda.empty_cache()
                        sum_forward_elapse = time.time()
                        outputs, losses = self.forward(data, target)
                        break
                    except Exception as e:
                        # Create new data loader in the (rare) case of segmentation fault
                        logging.warning('Creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        train_iter.shuffle(i_epoch+i_batch+int(time.time()))
                        train_loader = torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=workers, pin_memory=False)

                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        train_loaders[loader_id]=iter(train_loader)
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        active_batches[loader_id]=batch_s
                        gc.collect()
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse
                        gc.collect()

                sum_forward_elapse = time.time() - sum_forward_elapse

                # [backward]
                optimiser.zero_grad()
                sum_backward_elapse = time.time()
                for loss in losses:
                    with amp.scale_loss(loss, optimiser) as scaled_loss:
                        scaled_loss.backward()

                sum_backward_elapse = time.time() - sum_backward_elapse
                lr = lr_scheduler.update()
                batch_size = tuple(data.size())

                self.adjust_learning_rate(optimiser=optimiser,lr=lr)
                optimiser.step()

                # [evaluation] update train metric
                metrics.update([output.data.cpu() for output in outputs],
                               target.cpu(),
                               [loss.data.cpu() for loss in losses],
                               lr,
                               batch_size)

                # Append matrices to lists
                m = metrics.get_name_value()
                train_top1_sum.append(m[1][0][1])
                train_top5_sum.append(m[2][0][1])
                train_loss_sum.append(m[0][0][1])


                # timing each batch
                epoch_speeds += [sum_read_elapse,sum_forward_elapse,sum_backward_elapse]
                sum_sample_inst += data.shape[0]

                if (i_batch % self.step_callback_freq) == 0:
                    # retrive eval results and reset metic
                    self.callback_kwargs['namevals'] = metrics.get_name_value()
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['read_elapse'] = sum_read_elapse / data.shape[0]
                    self.callback_kwargs['forward_elapse'] = sum_forward_elapse / data.shape[0]
                    self.callback_kwargs['backward_elapse'] = sum_backward_elapse / data.shape[0]
                    sum_read_elapse = 0.
                    sum_forward_elapse = 0.
                    sum_backward_elapse = 0.
                    sum_sample_inst = 0.
                    # callbacks
                    self.step_end_callback()

            # Epoch end
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimiser_dict'] = optimiser.state_dict()
            self.epoch_end_callback(directory)
            train_loaders = {}
            active_batches = {}

            l = len(train_top1_sum)
            train_top1_sum = sum(train_top1_sum)/l
            train_top5_sum = sum(train_top5_sum)/l
            train_loss_sum = sum(train_loss_sum)/l
            train_writer.write((i_epoch, train_top1_sum, train_top5_sum,train_loss_sum))
            logging.info('Epoch [{:d}]  (train)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss: {:.5f}'.format(i_epoch,train_top1_sum,train_top5_sum,train_loss_sum))

            # Evaluation happens here
            if (eval_iter is not None) and ((i_epoch+1) % max(1, int(self.save_checkpoint_freq/2))) == 0:
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))
                metrics.reset()
                self.net.eval()
                sum_read_elapse = time.time()
                sum_forward_elapse = 0.
                sum_sample_inst = 0
                for i_batch, (data, target) in enumerate(eval_iter):
                    sum_read_elapse = time.time()
                    self.callback_kwargs['batch'] = i_batch
                    sum_forward_elapse = time.time()

                    # [forward] making next step
                    torch.cuda.empty_cache()
                    outputs, losses = self.forward(data, target)

                    sum_forward_elapse = time.time() - sum_forward_elapse


                    metrics.update([output.data.cpu() for output in outputs],
                                    target.cpu(),
                                   [loss.data.cpu() for loss in losses])

                    m = metrics.get_name_value()
                    val_top1_sum.append(m[1][0][1])
                    val_top5_sum.append(m[2][0][1])
                    val_loss_sum.append(m[0][0][1])


                    sum_sample_inst += data.shape[0]

                    if (i_batch%50 == 0):
                        val_top1_avg = sum(val_top1_sum)/(i_batch+1)
                        val_top5_avg = sum(val_top5_sum)/(i_batch+1)
                        val_loss_avg = sum(val_loss_sum)/(i_batch+1)
                        logging.info('Epoch [{:d}]: Iteration [{:d}]:  (val)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss {:.5f}'.format(i_epoch,i_batch,val_top1_avg,val_top5_avg,val_loss_avg))

                # evaluation callbacks
                self.callback_kwargs['read_elapse'] = sum_read_elapse / data.shape[0]
                self.callback_kwargs['forward_elapse'] = sum_forward_elapse / data.shape[0]
                self.callback_kwargs['namevals'] = metrics.get_name_value()
                self.step_end_callback()

                l = len(val_top1_sum)
                val_top1_sum = sum(val_top1_sum)/l
                val_top5_sum = sum(val_top5_sum)/l
                val_loss_sum = sum(val_loss_sum)/l
                val_writer.write((i_epoch, val_top1_sum, val_top5_sum,val_loss_sum))
                logging.info('Epoch [{:d}]:  (val)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss {:.5f}'.format(i_epoch,val_top1_sum,val_top5_sum,val_loss_sum))


        logging.info("--- Finished ---")
        train_writer.close()
        val_writer.close()
'''
===  E N D  O F  C L A S S  M O D E L ===
'''
