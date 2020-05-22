'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import cv2
import random
import numpy as np
import torch
import imgaug.augmenters as iaa
import coloredlogs, logging
coloredlogs.install()


'''
===  S T A R T  O F  C L A S S  C O M P O S E ===

    [About]
        Function for composing together differengt transforms. If normalisation is enabled, it will also normalise
        the transformed videos at the end.

    [Init Args]
        - transforms: imgaug.Sequential object that contains all the transforms to be applied. Transforms will be the
        same for each frame in a video to ensure continuity.
        - normalise: Boolean for applying normalisation at the end of transformations. It will normally be the
        ImageNet mean and std. Defaults to None.

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - __call__ : Main class call function for applying transformations and performing normalisation if not None.
'''
class Compose(object):

    def __init__(self, transforms, normalise=None):
        self.transforms = transforms
        self.normalise = normalise


    def __call__(self, data, end_size):

        # imgaug package requires batch size - apply same tranform to all frames
        vid_aug = self.transforms.to_deterministic()
        # Resizing if necessary
        sizing = iaa.Resize({"height":end_size[1],"width":end_size[2]})
        sizing = sizing.to_deterministic()

        data_aug = [vid_aug.augment_image(frame) for frame in data]
        data_aug = [sizing.augment_image(frame) for frame in data_aug]

        data_shape = str(np.asarray(data).shape)
        new_data_shape = str(np.asarray(data_aug).shape)
        #logging.info('Initial shape of frames: {}  and new frames sizes: {}'.format(data_shape,new_data_shape))

        data = np.asarray(data_aug)

        # Convert video to Tensor Object at the end of the transforms
        if isinstance(data, np.ndarray):
            # handle numpy array
            data = torch.from_numpy(data).permute((3, 0, 1, 2))
            # backward compatibility
            data = data.float() / 255.0

        # Check if normalisation is to be applied
        if self.normalise is not None:
            for d, m, s in zip(data, self.normalise[0], self.normalise[1]):
                d.sub_(m).div_(s)
        return data
'''
===  E N D  O F  C L A S S  C O M P O S E ===
'''


'''
===  S T A R T  O F  C L A S S  T R A N S F O R M ===

    [About]
        Base class for all transformations.

    [Methods]
        - set_random_state: Function for randomisation based on seed.
'''
class Transform(object):
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)
'''
===  E N D  O F  C L A S S  T R A N S F O R M ===
'''
