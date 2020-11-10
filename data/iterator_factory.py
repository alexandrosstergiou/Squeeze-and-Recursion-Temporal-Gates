'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import random
import sys
import coloredlogs, logging
coloredlogs.install()
import math
import torch
import copy
import numpy as np
import imgaug.augmenters as iaa
import torch.multiprocessing as mp
from torch.nn import functional as F
from . import video_transforms as transforms
from .video_iterator import VideoIter
from torch.utils.data.sampler import RandomSampler
from . import video_sampler as sampler


'''
---  S T A R T  O F  F U N C T I O N  G E T _ D I V I N G 4 8  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the Diving48 dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_diving48(data_root=os.path.join('/media','user','disk0'),
               clip_length=8,
               clip_size=256,
               val_clip_length=None,
               val_clip_size=None,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               **kwargs):

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.8, aug)


    train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))

    train = VideoIter(video_prefix = os.path.join(data_root, 'data', 'Diving48_videos','jpg'),
                      csv_filepath = os.path.join(data_root, 'labels', 'diving48_train.csv'),
                      include_timeslices = False,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=224, height=224, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  #sometimes_aug(iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=True)),
                                  #sometimes_aug(iaa.pillike.EnhanceColor(factor=(1.2, 1.6))),
                                  #sometimes_aug(iaa.MotionBlur([3,5,7])),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)

    val   = VideoIter(dataset_location = os.path.join(data_root, 'data',
                      'Diving48_videos','jpg'),
                      csv_filepath = os.path.join(data_root, 'labels', 'diving48_val.csv'),
                      include_timeslices = False,
                      sampler=val_sampler,
                      video_size=(val_clip_length,val_clip_size,val_clip_size),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=val_clip_size, height=val_clip_size, position='center')
                                        ]),
                                        normalise=[mean,std]),
                      name='val')

    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ D I V I N G 4 8  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ H M D B 5 1  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the hmdb51 dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_hmdb51(data_root=os.path.join('/media','user','disk0'),
               clip_length=8,
               clip_size=256,
               val_clip_length=None,
               val_clip_size=None,
               train_interval=4,
               val_interval=4,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               **kwargs):

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.8, aug)


    train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))

    train = VideoIter(dataset_location = os.path.join(data_root, 'data', 'HMDB51_videos','jpg'),
                      csv_filepath = os.path.join(data_root, 'labels', 'hmdb_split1_train_1.csv'),
                      include_timeslices = False,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=224, height=224, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  #sometimes_aug(iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=True)),
                                  #sometimes_aug(iaa.pillike.EnhanceColor(factor=(1.2, 1.6))),
                                  #sometimes_aug(iaa.MotionBlur([3,5,7])),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)

    val   = VideoIter(dataset_location = os.path.join(data_root,'data',
                      'HMDB51_videos','jpg'),
                      csv_filepath = os.path.join(data_root, 'labels', 'hmdb_split1_val.csv'),
                      include_timeslices = False,
                      sampler=val_sampler,
                      video_size=(val_clip_length,val_clip_size,val_clip_size),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=val_clip_size, height=val_clip_size, position='center')
                                        ]),
                                        normalise=[mean,std]),
                      name='val')

    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ H M D B 5 1  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ U C F 1 0 1  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the UCF101 dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 4.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 4.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_ucf101(data_root=os.path.join('/media','user','disk0'),
               clip_length=8,
               clip_size=256,
               val_clip_length=None,
               val_clip_size=None,
               train_interval=4,
               val_interval=4,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               **kwargs):

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.3, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.65, aug)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(dataset_location=os.path.join(data_root, 'data', 'UCF101_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'ucf101_split1_train_1.csv'),
                      include_timeslices = False,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=224, height=224, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  #sometimes_aug(iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=True)),
                                  #sometimes_aug(iaa.pillike.EnhanceColor(factor=(1.2, 1.6))),
                                  #sometimes_aug(iaa.MotionBlur([3,5,7])),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train


    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(dataset_location=os.path.join(data_root, 'data', 'UCF101_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'ucf101_split1_val.csv'),
                      include_timeslices = False,
                      sampler=val_sampler,
                      video_size=(val_clip_length,val_clip_size,val_clip_size),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=val_clip_size, height=val_clip_size, position='center')
                                        ]),
                                        normalise=[mean,std]),
                      name='val')

    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ U C F 1 0 1  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ K I N E T I C S  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the Kinetics dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - name: String for the dataset name. This differs from the rest of the functions are multiple variants of
        kinetics can be used (Mini-Kinetics, K-400, K-600, K-700). Defaults to `KINETICS-700`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 3.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 3.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_kinetics(data_root=os.path.join('/media','user','disk0'),
                 name = 'KINETICS-700',
                 clip_length=8,
                 clip_size=256,
                 val_clip_length=None,
                 val_clip_size=None,
                 train_interval=3,
                 val_interval=3,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.9, aug)


    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))

    if ('200' in name):
        extension = '200'
    elif ('400' in name):
        extension = '400'
    elif ('600' in name):
        extension = '600'
    elif ('700' in name):
        extension = '700'
    else:
        extension = '700_2020'

    train = VideoIter(dataset_location = os.path.join(data_root, 'data',
                      'Kinetics_videos','jpg'),
                      csv_filepath = os.path.join(data_root, 'labels', 'kinetics-'+extension+'_train.csv'),
                      include_timeslices = True,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 324, "longer-side":"keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=clip_size, height=clip_size, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  #sometimes_aug(iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=True)),
                                  #sometimes_aug(iaa.pillike.EnhanceColor(factor=(1.2, 1.6))),
                                  #sometimes_aug(iaa.MotionBlur([3,5,7])),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(dataset_location=os.path.join(data_root, 'data', 'Kinetics_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'kinetics-'+extension+'_val.csv'),
                      include_timeslices = True,
                      sampler=val_sampler,
                      video_size=(val_clip_length,val_clip_size,val_clip_size),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=val_clip_size, height=val_clip_size, position='center')
                                        ]),
                                        normalise=[mean,std]),
                      name='val')
    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ K I N E T I C S  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ H A C S  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the HACS dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_hacs(data_root=os.path.join('/media','user','disk0'),
                 clip_length=8,
                 clip_size=256,
                 val_clip_length=None,
                 val_clip_size=None,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 **kwargs):
    """ data iter for HACS
    """

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.9, aug)


    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))

    train = VideoIter(dataset_location=os.path.join(data_root, 'data' , 'HACS_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'HACS_clips_v1.1_train.csv'),
                      include_timeslices = True,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 384, "longer-side":"keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=384, height=384, position='center'),
                              iaa.CropToFixedSize(width=clip_size, height=clip_size, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  sometimes_aug(iaa.AverageBlur(k=(1,2))),
                                  sometimes_aug(iaa.Multiply((0.8, 1.2))),
                                  sometimes_aug(iaa.GammaContrast((0.85,1.15),per_channel=True)),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(dataset_location=os.path.join(data_root, 'data', 'HACS_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'HACS_clips_v1.1_val.csv'),
                      include_timeslices = True,
                      sampler=val_sampler,
                      video_size=(16,256,256),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=294, height=294, position='center'),
                                            iaa.CropToFixedSize(width=256, height=256, position='center')
                                        ]),
                                        normalise=[mean,std]),
                      name='val')

    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ H A C S  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ M O M E N T S  ---

    [About]
        Function for creating iteratiors for both training and validation sets for the MiT dataset.

    [Args]
        - data_root: String containing the complete path of the dataset. Note that the path should correspond to
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Integer for the width and height of the frames in the video. Defaults to 256.
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_moments(data_root=os.path.join('/media','user','disk0'),
                 clip_length=8,
                 clip_size=256,
                 val_clip_length=None,
                 val_clip_size=None,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 **kwargs):
    """ data iter for Moments in Time
    """

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
    sometimes_seq = lambda aug: iaa.Sometimes(0.9, aug)


    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))

    train = VideoIter(dataset_location=os.path.join(data_root, 'data' ,
                      'Moments_in_Time_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'Moments_in_Time_train_1.csv'),
                      include_timeslices = False,
                      sampler=train_sampler,
                      video_size=(clip_length,clip_size,clip_size),
                      video_transform = transforms.Compose(
                          transforms=iaa.Sequential([
                              iaa.Resize({"shorter-side": 294, "longer-side": "keep-aspect-ratio"}),
                              iaa.CropToFixedSize(width=224, height=224, position='uniform'),
                              sometimes_seq(iaa.Sequential([
                                  sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                  sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                  #sometimes_aug(iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=True)),
                                  #sometimes_aug(iaa.pillike.EnhanceColor(factor=(1.2, 1.6))),
                                  #sometimes_aug(iaa.MotionBlur([3,5,7])),
                                  sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                  sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                  sometimes_aug(
                                      iaa.OneOf([
                                          iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                          iaa.Rotate(rotate=(-10,10)),
                                      ])
                                  )
                              ])),
                              iaa.Fliplr(0.5)
                          ]),
                          normalise=[mean,std]
                      ),
                      name='train',
                      shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(dataset_location=os.path.join(data_root, 'data',
                      'Moments_in_Time_videos','jpg'),
                      csv_filepath=os.path.join(data_root, 'labels', 'Moments_in_Time_val.csv'),
                      include_timeslices = False,
                      sampler=val_sampler,
                      video_size=(val_clip_length,val_clip_size,val_clip_size),
                      video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side": "keep-aspect-ratio"}),
                                            iaa.CenterCropToFixedSize(width=val_clip_size, height=val_clip_size)
                                        ]),
                                        normalise=[mean,std]),
                      name='val')

    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ M O M E N T S  ---
'''



'''
---  S T A R T  O F  F U N C T I O N  C R E A T E  ---

    [About]
        Function for creating iterable datasets.

    [Args]
        - name: String for the name of the dataset. Supported datasets are [`UCF101`,`HMDB51`,`HACS`,`DIVING48`,`KINETICS-X.X`,`MOMENTS`].
        - batch_size: Integer for the size of each batch.
        - return_len: Boolean for returning the length of the dataset. Defaults to False.
        - num_workers: Integer for the number of workers used for loading data from disk. Defaults to 24.

    [Returns]
        - Tuple for training VideoIter object and validation utils.data.DataLoader object.
'''
def create(name, batch_size, return_len=False, num_workers=24, **kwargs):

    if name.upper() == 'UCF101':
        dataset_iter = get_ucf101(**kwargs)
    elif name.upper() == 'HMDB51':
        dataset_iter = get_hmdb51(**kwargs)
    elif 'KINETICS' in name.upper():
        dataset_iter = get_kinetics(name=name,**kwargs)
    elif name.upper() == 'HACS':
        dataset_iter = get_hacs(**kwargs)
    elif name.upper() == 'DIVING48':
        dataset_iter = get_diving48(**kwargs)
    elif name.upper() == 'MOMENTS':
        dataset_iter = get_moments(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))

    train,val = dataset_iter

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    return train,val_loader,train.__len__()

    if(not isinstance(dataset_iter,tuple)):
            train_loader = torch.utils.data.DataLoader(dataset_iter,
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            return train_loader

    train,val = dataset_iter

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    if return_len:
        return(train_loader,val_loader,train.__len__())
    else:
        return (train_loader, val_loader)
'''
---  E N D  O F  F U N C T I O N  C R E A T E  ---
'''
