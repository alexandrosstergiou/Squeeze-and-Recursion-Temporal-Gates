'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import math
import numpy as np
import coloredlogs, logging
coloredlogs.install()


'''
===  S T A R T  O F  C L A S S  R A N D O M S A M P L I N G ===

    [About]
        Class for random sampling frames based on a uniform distribution and a sampling interval.

    [Init Args]
        - num: Integer for number of frames.
        - interval: Interger for the sampling interval. Defaults to 1.
        - speed: List or Tuple for the sampling speed based on the uniform distribution.
        - seed: Integer for randomisation.

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - set_num: Setter function for number of frames.
        - sampling: Main function for sampling.
'''
class RandomSampling(object):
    def __init__(self, num, interval=1, speed=[1.0, 1.0], seed=0):
        num = int(num)
        assert num > 0, "at least sampling 1 frame"
        self.frames = num
        self.num = num
        self.interval = interval if type(interval) == list else [interval]
        self.speed = speed
        self.rng = np.random.RandomState(seed)

    #setter for number of frames
    def set_num(self,new_num):
        assert new_num > 0, "at least sampling 1 frame"
        self.num = new_num

    # Setter for new number of sampled frames
    def set_num(self, new_num):
        self.num = new_num

    def sampling(self, range_max, v_id=None, prev_failed=False):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        interval = self.rng.choice(self.interval)
        if self.num == 1:
            return [self.rng.choice(range(0, range_max))]
        # sampling
        speed_min = self.speed[0]
        speed_max = min(self.speed[1], (range_max-1)/((self.num-1)*interval))
        if speed_max < speed_min:
            return [self.rng.choice(range(0, range_max))] * self.num

        random_interval = self.rng.uniform(speed_min, speed_max) * interval

        frame_range = (self.num-1) * random_interval
        clip_start = self.rng.uniform(0, (range_max-1) - frame_range)
        clip_end = clip_start + frame_range
        idxs = np.linspace(clip_start, clip_end, self.num).astype(dtype=np.int).tolist()

        return idxs
'''
===  E N D  O F  C L A S S  R A N D O M S A M P L I N G ===
'''


'''
===  S T A R T  O F  C L A S S  S E Q U E N T I A L S A M P L I N G ===

    [About]
        Class for sequentialsampling frames based on a uniform distribution and a sampling interval.

    [Init Args]
        - num: Integer for number of frames.
        - interval: Interger for the sampling interval. Defaults to 1.
        - shuffle: Boolean for shuffling clips.
        - fix_cursor: Boolean for having a fixed cursor during sampling.
        - seed: Integer for randomisation.

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - set_num: Setter function for number of frames.
        - sampling: Main function for sampling.
'''
class SequentialSampling(object):
    def __init__(self, num, interval=1, shuffle=False, fix_cursor=False, seed=0):
        self.memory = {}
        self.num = int(num)
        self.interval = interval if type(interval) == list else [interval]
        self.shuffle = shuffle
        self.fix_cursor = fix_cursor
        self.rng = np.random.RandomState(seed)

    #setter for number of frames
    def set_num(self,new_num):
        self.num = int(new_num)

    def sampling(self, range_max, v_id, prev_failed=False):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        num = self.num
        interval = self.rng.choice(self.interval)
        frame_range = (num - 1) * interval + 1
        # sampling clips
        if v_id not in self.memory:
            clips = list(range(0, range_max-(frame_range-1), frame_range))
            if self.shuffle:
                self.rng.shuffle(clips)
            self.memory[v_id] = [-1, clips]
        # pickup a clip
        cursor, clips = self.memory[v_id]
        if not clips:
            return [self.rng.choice(range(0, range_max))] * num
        cursor = (cursor + 1) % len(clips)
        if prev_failed or not self.fix_cursor:
            self.memory[v_id][0] = cursor
        # sampling within clip
        idxs = range(clips[cursor], clips[cursor]+frame_range, interval)

        return idxs
'''
===  E N D  O F  C L A S S  S E Q U E N T I A L S A M P L I N G ===
'''

if __name__ == "__main__":

    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    """ test RandomSampling() """

    random_sampler = RandomSampling(num=8, interval=2, speed=[0.5, 2])

    logging.info("RandomSampling(): range_max < num")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=2, v_id=1)))

    logging.info("RandomSampling(): range_max == num")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=8, v_id=1)))

    logging.info("RandomSampling(): range_max > num")
    for i in range(90):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=30, v_id=1)))


    """ test SequentialSampling() """
    sequential_sampler = SequentialSampling(num=3, interval=3, fix_cursor=False)

    logging.info("SequentialSampling():")
    for i in range(10):
        logging.info("{:d}: v_id = {}: {}".format(i, 0, list(sequential_sampler.sampling(range_max=14, v_id=0))))
        # logging.info("{:d}: v_id = {}: {}".format(i, 1, sequential_sampler.sampling(range_max=9, v_id=1)))
        # logging.info("{:d}: v_id = {}: {}".format(i, 2, sequential_sampler.sampling(range_max=2, v_id=2)))
        # logging.info("{:d}: v_id = {}: {}".format(i, 3, sequential_sampler.sampling(range_max=3, v_id=3)))
