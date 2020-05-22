'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import json
import time
from datetime import datetime
import os
import csv
import cv2
import random
import numpy as np
import sqlite3
import torch
import torch.utils.data as data
import coloredlogs, logging
coloredlogs.install()
import linecache
import sys
from torch.nn import functional as F


'''
===  S T A R T  O F  C L A S S  V I D E O  ===

    [About]
        Wrapper class for reading SQL Database files of frames (see jpgs2singlefile.py) and return them as numpy arrays.

    [Init Args]
        - vid_path: String that points to the video filepath

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - __del__ : (Args:None) : Function for closing any open database readers.
        - __enter__ : Function for returning the object.
        - __exit__ : Function for error handling, is used to call __del__
        - reset : Function for resetting the class variables vid_path(string), frame_count(int) and faulty_frame(array)
        - open : Function for establishing a connection to the database file, given that the file actually exists. Will first try and reset all variables (through __reset__) and then create a connection (self.con) and cursor (self.cur) for database iterations. Taks as argument the video path string.
        - count_frames: Function for returning the number of elements/rows in the database (i.e. the number of saved frames).
        - extract_frames: High level function for extracting the frames in the database. The indices of the frames are given as an argument of type `list` which should hold EVERY index of the frames to be extracted.
        - extract_frames_fast: Main function for iterating over database elements/frames and appending them in a numpy array befor returning the created array. Array length is created dynamically at the first iteration. Alongside the indices of frames, the function also takes as an argument if the frames are to be imported with colour or not (boolean variable).
'''
class Video(object):
    """basic Video class"""

    def __init__(self, vid_path, video_transform=None, end_size=(16,224,224)):
        self.path = vid_path
        self.video_path = os.path.join(vid_path, 'frames.db')
        self.frame_path = os.path.join(vid_path, 'n_frames')
        self.video_transform = video_transform
        self.end_size = end_size

    def __enter__(self):
        return self

    def reset(self):
        self.video_path = None
        self.frame_path = None
        self.frame_count = -1
        return self


    def count_frames(self):
        if (os.path.isfile(self.frame_path)):
            self.frame_count = int(open(self.frame_path,'r').read())
        elif (os.path.isfile(self.video_path)):
            con = sqlite3.connect(self.video_path)
            cur = con.cursor()
            sql = "SELECT Objid, frames FROM Images"
            row = cur.execute(sql,frame_names)
            i = 1
            for ObjId in row:
                i += 1
            with open(self.frame_path, 'w') as dst_file:
                dst_file.write(str(i))
        else:
            logging.error('Directory {} is empty!'.format(self.frame_path))
            raise Exception("Empty directory !")
        return self.frame_count

    def extract_frames(self, indices):
        frames = self.extract_frames_fast(indices)
        return frames

    def extract_frames_fast(self, indices):
        frames = []

        con = sqlite3.connect(self.video_path)
        cur = con.cursor()


        # retrieve entire video from database (frames are unordered)
        frame_names = ["{}/{}".format(self.path.split('/')[-1],'frame_%05d'%(index+1)) for index in indices]
        sql = "SELECT Objid, frames FROM Images WHERE ObjId IN ({seq})".format(seq=','.join(['?']*len(frame_names)))
        row = cur.execute(sql,frame_names)

        ids = []
        frames = []
        i = 0

        row = row.fetchall()
        # Video order re-arangement
        for ObjId, item in row:
            #--- Decode blob
            nparr  = np.fromstring(item, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            ids.append(ObjId)
            frames.append(img)
            i+=1

        frames = [frame for _, frame in sorted(zip(ids,frames), key=lambda pair: pair[0])]

        frames = np.asarray(frames)

        # Force cropping if spatial dims are smaller than the expected size
        t, h, w, _ = frames.shape
        indxs = str(indices)
        #if (h<self.end_size[1] or w<self.end_size[2]):
            #logging.info("Video of size ({} {} {}) with indices {} is smaller than the required dims and it will be interpolated.".format(i,h,w,indxs))

        # apply video augmentation
        if self.video_transform is not None:
            frames = self.video_transform(frames, self.end_size)

        # Final resizing - w/ check
        _, t, h, w = list(frames.size())

        # Interpolate temporal dimension if frames are less than the ones required
        if (t!=self.end_size[0]):
            frames = F.interpolate(frames.unsqueeze(0), size=self.end_size, mode='trilinear',align_corners=False).squeeze(0)

        if not('img' in vars()):
            print('Could not load (all) frames from database: ',self.video_path, 'with_indices',indices)

        cur.close()
        con.close()

        #print(self.video_path,frames.size())
        return frames


'''
===  E N D  O F  C L A S S  V I D E O  ===
'''







'''
===  S T A R T  O F  C L A S S  V I D E O I T E R A T O R ===

    [About]

        Iterator class for loading the dataset filelist for a .CSV file to a dictionary and iteratively create Video class objects for frame loading.

    [Init Args]

        - dataset_location: String for the (full) directory path of the dataset.
        - csv_filepath: String for the (full) filepath of the csv file containing datset information.
        - include_timeslices: Boolean, of cases that datasets video directories also include the time-segments in the name (should be mstly used by either Kinetics or HACS).
        - sampler: Any object of the video_sampler file, used for sampling the frames.
        - video_transform: Any object of the video_transforms file, used for applying video transformations (see video_transforms.py file). Defaults to None.
        - name: String for declaring the set that the iterator is made for (e.g. train/test). Defaults to "<NO_NAME>".
        - force_colour: Boolean that is used to determine if the video will be have a single/three channels. Defaults to True.
        - return_video_path: Boolean for returning or not the full video filepath after a video is loaded. Defaults to False.
        - randomise: Boolean for additional randomisation based on date and time. Defaults to True.
        - shuffle_list_seed: Integer, for random shuffling. Defaults to None.

    [Methods]

        - __init__ : Class initialiser
        - getitem_array_from_video: Returns a numpy array containing the frames, an integer for the video label and the full video database path. The input to the function is the index of the video (corresponding to an element in the video_dict)
        - __getitem__ : Wrapper function for the getitem_array_from_video function. Can return either the numpy array of frames with also the curresponding lable in int format as well as the complete filepath of the video (if specified by the user)
        - __len__ : Returning the length/size of the dataset
        - get_video_dict : Main function for creating the video dictionary. Taks as arguments the location of the dataset (directory) the filepath to the .CSV file containing the dataset info and a boolean variable named include_timeslices which is used in the instance that the video folders also include the time segments in their name (defualts to False).

'''
class VideoIter(data.Dataset):

    def __init__(self,
                 dataset_location,
                 csv_filepath,
                 include_timeslices,
                 video_size,
                 sampler,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_video_path=False,
                 randomise = True,
                 shuffle_list_seed=None):

        super(VideoIter, self).__init__()

        # Class parameter initialisation
        self.clip_size = video_size
        self.sampler = sampler
        self.dataset_location = dataset_location
        self.video_transform = video_transform
        self.return_video_path = return_video_path
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        self.randomise = randomise
        # Additional randomisation
        if randomise:
            t = int(time.time())
            self.rng = np.random.RandomState(shuffle_list_seed+t if shuffle_list_seed else t)

        # load video dictionary
        self.video_dict = self.get_video_dict(location=dataset_location,csv_file=csv_filepath,include_timeslices=include_timeslices)

        # Create array to hold the video indices
        self.indices = list(self.video_dict.keys())

        # Shuffling indices array
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.indices)


        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.indices)))


    def size_setter(self,new_size):
        self.clip_size = new_size
        self.sampler.set_num(new_size[0])

    def shuffle(self,seed):
        self.rng = np.random.RandomState(seed)
        if self.randomise:
            t = int(time.time())
            self.rng = np.random.RandomState(seed+t)
        self.rng.shuffle(self.indices)



    def getitem_array_from_video(self, index):
        # get current video info
        v_id, label, vid_path, frame_count = self.video_dict.get(index)

        try:

            # Create Video object
            video = Video(vid_path=vid_path,video_transform=self.video_transform,end_size=self.clip_size)
            if frame_count < 0:
                frame_count = video.count_frames()

            #print('sampling frames...',type(frame_count),type(v_id))
            # dynamic sampling
            sampled_indices = self.sampler.sampling(range_max=frame_count, v_id=v_id)


            # extracting frames
            sampled_frames = video.extract_frames(indices=sampled_indices)


        except IOError as e:
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))


        #print('Processed item w/ index: ',v_id, 'and shape',sampled_frames.shape)

        return sampled_frames, label, vid_path



    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                index = int(index)
                if (index == 0):
                    index += 1
                frames, label, vid_path = self.getitem_array_from_video(index)
                _, t, h, w = frames.size()
                if (t!=self.clip_size[0] and h!=self.clip_size[1] and w!=self.clip_size[0]):
                    raise Exception('Clip size should be ({},{},{}), got clip with: ({},{},{})'.format(*self.clip_size,t,h,w))
                succ = True
            except Exception as e:

                exc_type, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                lineno = tb.tb_lineno
                filename = f.f_code.co_filename
                linecache.checkcache(filename)
                line = linecache.getline(filename, lineno, f.f_globals)
                message = 'Exception in ({}, line {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

                prev_index = index
                #index = self.rng.choice(range(0, self.__len__()))
                d_time = int(round(datetime.now().timestamp() * 1000)) # Ensure randomisation
                index = random.randrange(d_time % self.__len__())
                #logging.warning("VideoIter:: Warning: {}".format(message))
                #logging.warning("VideoIter:: Inital index of {} changed to index:{}".format(prev_index,index))

        if self.return_video_path:
            return frames, label, vid_path
        else:
            return frames, label


    def remove_indices(self,d_indices):
        del self.indices[d_indices]

    def __len__(self):
        return len(self.indices)


    def indices_list(self):
        return self.indices


    def get_video_dict(self, location, csv_file, include_timeslices=True):

        # Esure that both given dataset location and csv filepath do exist
        assert os.path.exists(location), "VideoIter:: failed to locate given dataset location: `{}'".format(location)
        assert os.path.exists(csv_file), "VideoIter:: failed to locate given csv file location: `{}'".format(csv_file)

        # building dataset
        # - videos_dict : Used to store all videos in a dictionary format with video_index(key) : video_info(value)
        # - logging_interval: The number of iterations performed before writting to the logger
        # - found_videos: Integer for counting the videos from the csv file that are indeed part of the dataset in given location
        # - processed_ids: List of all ids_processed in the current logging_interval
        # - labels_dict: Dictionary for associating string labels with ints
        # - labels_last_index: Integer of keeping track of the integers used


        videos_dict = {}
        logging_interval = 10000
        found_videos = 0
        processed_ids = []


        # Store dictionary of labels keys:'str' , values:'int' to a .JSON file (as a common reference between dataset sets)
        if ('train' in csv_file):
            labels_dict_filepath = csv_file.split('train')[0]+'dictionary.json'
        elif ('val' in csv_file):
            labels_dict_filepath = csv_file.split('val')[0]+'dictionary.json'
        else:
            labels_dict_filepath = csv_file.split('test')[0]+'dictionary.json'

        if (os.path.exists(labels_dict_filepath)):
            with open(labels_dict_filepath) as json_dict:
                labels_dict = json.loads(json_dict.read())
            label_last_index =len(labels_dict)
        else:
            labels_dict = {}
            label_last_index = 0

        for i,line in enumerate(csv.DictReader(open(csv_file))):

            if (i%logging_interval == 0):
                logging.info("VideoIter:: Processed {:d}/{:d} videos".format(found_videos,i, ))#str(processed_ids).strip('[]')))
                processed_ids = []

            # Account for folder name of different datasets (e.g. Kinetics & HACS => `youtube_id` , UCF-101 & HMDB-51 => `id`)
            if ('youtube_id' in line):
                id = line.get('youtube_id').strip()
            else:
                id = line.get('id').strip()

            processed_ids.append(id)

            video_path = os.path.join(location, line.get('label'), id)

            # Check if label has already been found and if not add it to the dictionary:
            if not (line.get('label') in labels_dict):
                labels_dict[line.get('label')] = label_last_index
                label_last_index += 1


            # Case that the filename also includes the timeslices
            if (include_timeslices):
                video_path = video_path + ('_%06d'%int(float(line.get('time_start')))+('_%06d'%int(float(line.get('time_end')))))

            # Check if video indeed exists (handler for not downloaded videos)
            if not os.path.exists(video_path):
                # Uncomment line for additional user feedback
                #logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
                continue


            # Increase videos count and read number of frames
            else:
                found_videos += 1
                with open(os.path.join(video_path,'n_frames')) as f:
                    frame_count = int(f.readline())

            # Append video info to dictionary
            info = [found_videos, labels_dict.get(line.get('label')), video_path, frame_count]
            videos_dict[found_videos] = info


        logging.info("VideoIter:: - Found and stored: {:d}/{:d} videos from csv file \n".format(found_videos, i))


        # Save dictionary if it does not already exists
        if not (os.path.exists(labels_dict_filepath)):
            with open(labels_dict_filepath,'w') as json_dict:
                json.dump(labels_dict,json_dict)


        return videos_dict



'''
===  E N D  O F  C L A S S  V I D E O I T E R A T O R  ===
'''
