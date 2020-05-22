'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import datetime
import logging
import coloredlogs, logging
coloredlogs.install()
import math
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data import iterator_factory
from train import metric
from train.model import model
from train.lr_scheduler import MultiFactorScheduler

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

'''
---  S T A R T  O F  T R A I N _ M O D E L  ---
    [About]
        Main function that trains the specified model on a chosen dataset.
    [Args]
        - sym_net:
        - name: Sting for the dataset name.
        - model_prefix:
        - input_conf:
        - clip_length: Integer, for the number of frames to be used. Defaults to 32
        - clip_size: Integer, specifying the spatial dimensionality of the frames. Defaults to 224
        - train_frame_interval: Integer or List for the sampling interval to be used for sampling training set frames.
          Defaults to 2.
        - val_frame_interval: Integer or List for the sampling interval to be used for sampling validation set frames.
          Defaults to 2.
        - resume_epoch: Integer to be used in the case that training is to re-start from a specific epoch. If using -1
          the training will start as normal from epoch 0. Defaults to -1.
        - batch_size: Integer for the batch size to be used .In the case of cycles, the batch size is the base batch
          size to be used by the scheduler based on which the new batches will be calculated. Defaults to 16.
        - save_frequency: Integer for the frequency in which the model is saved. It is recommended to set the `save_frequency`
          to 1 and delete any epochs after training in order to ensure that you obtain the best possible results. Defaults to 1.
        - lr_base: Float for the base learning rate used. This is the lr to be used for the first n steps/epochs. If cycles are
          enabled, the learning rate will degrade by .5 at each step. If `resume_epoch` is not -1, the initial learning rate will change accordingly, based on the `lr_base`. Defaults to 0.1.
        - lr_factor: Float for the learning rate decay factor at each of the `lr_steps`, Usually the best practice is to use a
          0.1 reduction, Defaults to 0.1.
        - lr_steps: List of Integers for the steps to decrease the learning rate. Defaults to [50,100,150]
        - long_cycles: Boolean to specify if long cycles are to be used. Defaults to True
        - short_cycles: Boolean to specify if short cycles are to be used. Defaults to True.
        - end_epoch: Integer for the last epoch while training. Defaults to 300.
        - pretrained_3d: String for path of pre-trained model. If None is used, the model is trained from scratch. Defaults to
          None.
        - fine_tune: Bolean for the case of fine-tuning. This is especially useful for decreasing the learning rate of the
          convolution weights while maintaining the learning rate for the classifier (and thus training mostly the neurones
          that are responible for class predictions). Defaults to False.
        - dataset_location: String for the complete filepath of the dataset. This is the parent path. Along that path an
          additional folder 'labels' should exist. Defaults to 'Kinetics'.
        - net_name: String for the network name in order to be ported. Defaults to 'r3d_50'.
        - gpus: Integer for the number of GPUs to be used. Defaults to 4.
    [Returns]
        None
'''
def train_model(sym_net, name, model_prefix, input_conf,
                clip_length=32, clip_size=224, train_frame_interval=2, val_frame_interval=2,
                resume_epoch=-1, batch_size=16, save_frequency=1,
                lr_base=0.01, lr_factor=0.1, lr_steps=[50,100,150],
                long_cycles=True, short_cycles=True, end_epoch=300,
                pretrained_3d=None, fine_tune=False, dataset_location='Kinetics', net_name='r3d_50', gpus=4,
                **kwargs):

    assert torch.cuda.is_available(), "Only support CUDA devices."

    # Make results directory for .csv files if it does not exist
    results_path = str('./results/'+str(name)+'/'+str(net_name))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # data iterator - randomisation based on date and time values
    iter_seed = torch.initial_seed() + 100 + max(0, resume_epoch) * 100
    now = datetime.datetime.now()
    iter_seed += now.year + now.month + now.day + now.hour + now.minute + now.second

    # Get parent location
    # - `data` folder should include all the dataset examples.
    # - `labels` folder should inclde all labels in .csv format.
    # We use a global label formating - you can have a look at the link in the `README.md` to download the files.
    data_location = dataset_location.split('/data/')[0]

    clip_length = int(clip_length)
    clip_size = int(clip_size)

    train_loaders = {}

    # Create custom loaders for train and validation
    train_data, eval_loader, train_length = iterator_factory.create(
        name=name,
        batch_size=batch_size,
        return_len=True,
        clip_length=clip_length,
        clip_size=clip_size,
        val_clip_length=clip_length,
        val_clip_size=clip_size,
        train_interval=train_frame_interval,
        val_interval=val_frame_interval,
        mean=input_conf['mean'],
        std=input_conf['std'],
        seed=iter_seed,
        data_root=data_location)

    # Create model
    net = model(net=sym_net,
                criterion=torch.nn.CrossEntropyLoss().cuda(),
                model_prefix=model_prefix,
                step_callback_freq=1,
                save_checkpoint_freq=save_frequency,
                opt_batch_size=batch_size, # optional
                )
    net.net.cuda()


    # Parameter LR configuration for optimiser
    # Base layers are based on the layers as loaded to the model
    param_base_layers = []
    base_layers_mult = 1.0

    # New layers are based on fine-tuning
    param_new_layers = []
    new_layers_mult = 1.0

    name_base_layers = []

    param_transpose_layers = []
    transpose_layers_mult = 1.0

    # Iterate over all parameters
    for name, param in net.net.named_parameters():
        if fine_tune:
            if 'transpose' in name.lower():
                param_transpose_layers.append(param)
                transpose_layers_mult = .2
            elif name.lower().startswith('classifier'):
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                base_layers_mult = .6
                name_base_layers.append(name)
        else:
            if 'transpose' in name.lower():
                param_transpose_layers.append(param)
                transpose_layers_mult = .8
            else:
                param_new_layers.append(param)


    # User feedback
    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        logging.info("Optimiser:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

    optimiser = torch.optim.SGD([
        {'params': param_base_layers, 'lr_mult': base_layers_mult},
        {'params': param_new_layers, 'lr_mult': new_layers_mult},
        {'params': param_transpose_layers, 'lr_mult': transpose_layers_mult},],
        lr=lr_base,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
        )

    # Use Apex for Mixed precidion
    net.net, optimiser = amp.initialize(net.net, optimiser, opt_level="O1")

    # Create DataParallel wrapper
    net.net = torch.nn.DataParallel(net.net).cuda()

    # load params from pretrained 3d network
    if pretrained_3d:
        if resume_epoch < 0:
            assert os.path.exists(pretrained_3d), "cannot locate: `{}'".format(pretrained_3d)
            logging.info("Initialiser:: loading model states from: `{}'".format(pretrained_3d))
            checkpoint = torch.load(pretrained_3d)
            net.load_state(checkpoint['state_dict'], strict=False)
        else:
            logging.info("Initialiser:: skip loading model states from: `{}'"
                + ", since it's going to be overwrited by the resumed model".format(pretrained_3d))

    num_steps = train_length // batch_size

    # Long Cycle steps
    if (long_cycles):

        count = 0
        index = 0
        iter_sizes = [8, 4, 2, 1]
        initial_num = num_steps

        # Expected to find the number of batches that fit exactly to the number of iterations.
        # So the sum of the floowing batch sizes should be less or equal to the number of batches left.
        while sum(iter_sizes[index:]) <= num_steps:
            # Case 1: 8 x B
            if iter_sizes[index] == 8:
                count += 1
                index = 1
                num_steps -= 8
            # Case 2: 4 x B
            elif iter_sizes[index] == 4:
                count += 1
                index = 2
                num_steps -= 4
            # Case 3: 2 x B
            elif iter_sizes[index] == 2:
                count += 1
                index = 3
                num_steps -= 2
            # Base case
            elif iter_sizes[index] == 1:
                count += 1
                index = 0
                num_steps -= 1

        print ("New number of batches per epoch is {:d} being equivalent to {:1.3f} of original number of batches with Long cycles".format(count,float(count)/float(initial_num)))
        num_steps = count

    # Short Cycle steps
    if (short_cycles):

        # Iterate for *every* batch
        i = 0

        while i <= num_steps:
            m = i%3
            # Case 1: Base case
            if (m==0):
                num_steps -= 1
            # Case 2: b = 2 x B
            if (m==1):
                num_steps -= 2
            # Case 3: b = 4 x B
            else:
                num_steps -= 4

            i += 1

        # Update new number of batches
        print ("New number of batches per epoch is {:d} being equivalent to {:1.3f} of original number of batches with Short cycles".format(i,float(i)/float(initial_num)))
        num_steps = i

    # Split the batch number to four for every change in the long cycles
    long_steps = None
    if (long_cycles):
        step = num_steps//4
        long_steps = list(range(num_steps))[0::step]
        num_steps = long_steps[-1]

        # Create full list of long steps (for all batches)
        for epoch in range(1,end_epoch):
            end = long_steps[-1]
            long_steps = long_steps + [x.__add__(end) for x in long_steps[-4:]]

        # Fool-proofing
        if (long_steps[0]==0):
            long_steps[0]=1


    # resume training: model and optimiser - (account of various batch sizes)
    if resume_epoch < 0:
        epoch_start = 0
        step_counter = 0
    else:
        try:
            net.load_checkpoint(epoch=resume_epoch, optimizer=optimiser)
        except Exception:
            logging.warning('Initialiser:: No previous checkpoint found in the directory! You can specify the path explicitly with `pretrained_3d` argument.')
        epoch_start = resume_epoch
        step_counter = epoch_start * num_steps

    # Step dictionary creation
    iteration_steps = {'long_0':[],'long_1':[],'long_2':[],'long_3':[],'short_0':[],'short_1':[],'short_2':[]}
    #Populate dictionary
    for batch_i in range(0,num_steps):

        # Long cycle cases
        if batch_i>=0 and batch_i<num_steps//4:
            iteration_steps['long_0'].append(batch_i)
        elif batch_i>=num_steps//4 and batch_i<num_steps//2:
            iteration_steps['long_1'].append(batch_i)
        elif batch_i>=num_steps//2 and batch_i<(3*num_steps)//4:
            iteration_steps['long_2'].append(batch_i)
        else:
            iteration_steps['long_3'].append(batch_i)

        # Short cases
        if (batch_i%3==0):
            iteration_steps['short_0'].append(batch_i)
        elif (batch_i%3==1):
            iteration_steps['short_1'].append(batch_i)
        else:
            iteration_steps['short_2'].append(batch_i)



    # set learning rate scheduler
    lr_scheduler = MultiFactorScheduler(base_lr=lr_base,
                                        steps=[x*num_steps for x in lr_steps],
                                        iterations_per_epoch=num_steps,
                                        iteration_steps=iteration_steps,
                                        factor=lr_factor,
                                        step_counter=step_counter)
    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),
                                metric.BatchSize(name="batch_size"),
                                metric.LearningRate(name="lr"))
    # enable cudnn tune
    cudnn.benchmark = True

    # Main training happens here
    net.fit(train_iter=train_data,
            eval_iter=eval_loader,
            batch_shape=(int(batch_size),int(clip_length),int(clip_size),int(clip_size)),
            workers=8,
            optimiser=optimiser,
            long_short_steps_dir=iteration_steps,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            iter_per_epoch=num_steps,
            epoch_start=epoch_start,
            epoch_end=end_epoch,
            directory=results_path)
'''
---  E N D  O F  T R A I N _ M O D E L  F U N C T I O N  ---
'''
