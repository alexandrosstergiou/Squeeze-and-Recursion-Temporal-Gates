# Squeeze and Recursion Temporal Gates

![supported versions](https://img.shields.io/badge/python-3.5%2C3.6-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue/?style=flat&logo=pytorch&color=informational)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)
![Fork](https://img.shields.io/github/forks/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions?style=social)
![Star](https://img.shields.io/github/stars/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions?style=social)


--------------------------------------------------------------------------------

## Abstract
Temporal motion has been one of the essential components for the effective recognition of actions in videos. Time information and features are primarily extracted hierarchically through small sequences of few frames, in modern models, with the use of 3D convolutions. In this paper, we propose a method that can learn general changes of features across time, making activations not bound to their temporal locality, by also include a general notion of their feature motions. Through this recalibration of temporal feature cues across multiple frames, 3D-CNN models are capable of using features that are prevalent over different time segments, while being less dependent on their temporal receptive fields. We present improvements on both high and low capacity models, with the largest benefits being observed in low-memory models, as most of their current drawbacks rely on their poor generalisation capabilities because of the low number and complexity of their features. We present average improvements, over the corresponding state of the art models, in the range of x\% on Kinetics-700 (K-700), x\% on Moments in Time (MiT), 2.73\% on Human Actions Clips and Segments (HACS), x \% on HMDB-51 and x\% on UCF-101.  

<p align="center">
<img src="./figures/SR.png" width="700" height="150" />
</p>


<p align="center">
<i></i>
<br>
<a href="https://arxiv.org/pdf/1902.01078.pdf" target="blank">[arXiv preprint]</a>
 &nbsp;&nbsp;&nbsp;
<a href="https://www.youtube.com/watch?v=JANUqoMc3es&feature=youtu.be" target="blank">[video presentation]</a>
</p>

## Dependencies

Ensure that the following packages are installed in your machine:

+ `apex`  (version >= 0.1)
+ `coloredlogs`  (version >= 14.0)
+ `ffmpeg-python`  (version >=0.2.0)
+ `imgaug`  (version >= 0.4.0)
+ `opencv-python`  (version >= 4.2.0.32)
+ `torch` (version >= 1.4.0)
+ `youtube-dl` (version >= 2020.3.24)

You can intall all of the packages with the following command:
```
$ pip install apex coloredlogs ffmpeg-python imgaug opencv-python torch torchvision youtube-dl
```

! Disclaimer: This repository is heavily structurally influenced on Yunpeng Chen's MFNet [repo](https://github.com/cypw/PyTorch-MFNet)

## Installation

Please also make sure that `git` is installed in your machine:

```
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/alexandrosstergiou/Squeeze-and-Recursion-Temporal-Gates.git
```

## Datasets

We include training/data loading scripts for six action/video recognition datasets:

- **Human Action Clips and Segments (HACS)** [[link]](http://hacs.csail.mit.edu/) : It includes a total of roughly 500K clip segments sampled over 50K videos. you can download the dataset  from [this link](http://hacs.csail.mit.edu/dataset/HACS_v1.1.1.zip), or alternatively visit the [project's website](http://hacs.csail.mit.edu/).
- **Kinetics** [[link]](https://deepmind.com/research/open-source/kinetics): Composed of approximately 600K clips over 700 classes (with previously 400 \& 600) and each clip having an average duration of 10 seconds. You can download all three sets from [this link](https://storage.googleapis.com/deepmind-media/Datasets/kinetics700.tar.gz).
- **Moments in Time (MiT)** [[link]](http://moments.csail.mit.edu/): Is composed of about 800K videos over 339 classes. Video durations are limited to 3 seconds. The labels can be downloaded from [the website](http://moments.csail.mit.edu/) after competing the form.
- **UCF-101** [[link]](https://www.crcv.ucf.edu/data/UCF101.php): a smaller dataset of 13320 clips of 2~14 seconds. It includes a total of 101 action classes. The dataset can be downloaded in full from their website.
- **HMDB-51** [[link]](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/): The dataset includes a bit less than 10K video of 51 categories sampled from movies and series. The dataset can be downloaded directly from their website.
- **Diving-48** [[link]](http://www.svcl.ucsd.edu/projects/resound/dataset.html): Composed of a total 18K videos of diving actions. The dataset can be downloaded from the project's website.

All three HACS, Kinetics and MiT datasets can be dowloaded through the [ActivityNet's official Crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) by changing the download script. Direct downloads are not (officially) supported due to the vastness of these datasets.

#### Conversion to SQLite3 for speed optimisation

As extracting videos to image format in a per-frame fashion we opt to used SQL databases for each of the videos. This effectively (1) limits the number of random read operations and (2) The number of inodes used as individual image files (`.png`/`.jpeg`/etc.) increase the number of inodes used significantly.

 **In total, the speed-up in reading times is close to ~1000x of that with conventional random-access image files**

*Table for loading speed comparisons between JPG images and SQL databases. Rates are denoted as clips/videos loaded per second (clips/sec.)*

| Batch size | 1536, 4, 80, 80 |384, 8, 112, 112 | 96, 16, 158, 158 |  48, 16, 224, 224 |
| ---------- | --- | --- | --- | --- |
| JPG | 4.06e+3 | 3.56e+2| 4.21e+1| 2.94e+1|
| SQL | **1.05e+6** | **8.07e+5**| **8.53e+4**| **4.69e+4**|


All of the experiments were done with an AMD Threadripper 2950X over 24 workers. Disks used: 2x Samsung 970 Evo Plus (2TB \& 1TB).

*You can use the `dataset2databse` pypi [package](https://pypi.org/project/dataset2database/) or [repo](https://github.com/alexandrosstergiou/dataset2database) to convert videos to SQL*:
```
$ pip install dataset2database
```

#### Directory formatting

We assume a fixed directory formatting for both the data and the labels used. The dataset should be of the following structure at a directory:

```
<dataset>
  |
  └─── jpg
        │
        └──<class_i>
        │     │
        │     │─── <video_id_j>
        │     │         │
        │     │         │─── frames.db
        │     │         └─── n_frames
        │     │    
        │     │─── <video_id_j+1>
        │     │         │
        │     │         │─── frames.db
       ...   ...        └─── n_frames
```

In the structure, any items enclosed in angle brackets should be changed to the correct dataset name, class names and video ids. The three standard elements that should remain the same across any dataset are:
- **jpg**: which is the the container folder for all the classes in the dataset.

- **frames.db**: The SQL database specific for each video that contains all the frames in the format of `ObjId`: which should be a string containing the video filepath alongside the frame number and `frames` that encapsulates all the data. The SQL table should also be called `Images`.

- **n_frames**: A file that should only include the number of frames for the video for quick access.

> You can of course also use your own dataset if you follow the above structure and convert all your videos to SQL databases. The process for doing so should be identical for any of the currently supported datasets.

#### Data loading

The main data loading takes place in `data/video_iterator.py` which you can see for more information, In all line 87~110 handle both connecting to the sql databases and loading. A densely commented version of those lines can be found below for more info:

```python
con = sqlite3.connect('my_video_database.db')# Establishing connection
cur = con.cursor()# Cursor creation
frame_names = ["{}/{}".format(my_path.split('/')[-1],'frame_%05d'%(index+1)) for index in frame_indices]# Frame indices selection
sql = "SELECT Objid, frames FROM Images WHERE ObjId IN ({seq})".format(seq=','.join(['?']*len(frame_names)))# Build SQL request
row = cur.execute(sql,frame_names)# Execute SQL and retrieve frames
ids = []
frames = []
i = 0
row = row.fetchall()# Data fetching
# Video order re-arrangement
for ObjId, item in row:
  #--- Decode blob
  nparr  = np.fromstring(item, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  ids.append(ObjId)
  frames.append(img)
  i+=1
# Ensuring correct order of frames
frames = [frame for _, frame in sorted(zip(ids,frames), key=lambda pair: pair[0])]
# (if required) array conversion [frames x height x width x channels]
frames = np.asarray(frames)
```

#### Augmentations

To increase generalisation capabilities of the models we further include video augmentations inside our training process. These augmentations are primarily divided to **temporal-based** augmentations and **spatial-based** augmentations.

+ Temporal-based augmentations are used to perform changes at the temporal extend of the videos. Therefore, any variations based on time. These include:

  - Temporal sampling interval which is the maximum number of skipped frames between two frames from which the final video will be composed by.
  - Frame sampling which can either be done sequentially (e.g. centre-most frames) or randomly through a uniform/normal distribution.


+ Spatial-based augmentations are performed in frame-level and primarily change either the width-height or the channels/RGB values. We set a probability of 0.8 for performing **any** augmentations for each video, while each augmentation type is assigned a 0.4 probability. We make use of the `imgaug` package for all of our spatial augmentations:

    - Gaussian blur with a sigma { 0.1 , 0.2, 0.3}.
    - Per-channel addition in range of {-5, ..., 15}.
    - Gaussian noise with a scale of 76.5 (i.e 0.3 * 255).
    - Colour enhancement with a factor of (1.2, 1.6).
    - Motion blur with a kernel of { 3, 5, 7}.
    - Hue saturation addition {-16, ..., 16}.
    - Linear contrast with an alpha of {0.85, ..., 1.115}.
    - Perspective transform with scale {0.02,0.05}.
    - Rotation between {-10, ..., 10} degrees.

  Ranges and values were empirically evaluated in order to balance between a reasonable amount of deformation without alleviating the informative features of the video. It's important to note the **all video frames should present exactly the same type of spatial augmentations** to ensure coherence.  


#### Long-Short Circles

We additionally use a Multigrid training schedule for both improving generalisation and training times. Our implementation is based on the [Wu *et al.* paper](https://arxiv.org/abs/1912.00998). For convenience three `Dataloader` objects are used for every long circle that correspond to changes in data for each short circle.
- In case of RAM constraints we suggest to either not use the Multigrid training implementation or decrease the number of workers at `train/model.py`

## Usage

#### Examples

#### Calling arguments

#### Pre-trained weights

#### Switching from half to single point precision

## Monitoring

#### Scores format

We report the loss, top-1 and top-5 accuracy during each logging interval during training. For validation, we report the average loss updated after each batch. Both top-1 and top-5 accuracies are saved in a `.csv` file for both train and validation.

#### Speed monitoring

We also provide a monitor for speeds in terms of video-reading from disk (CPU), forward pass (GPU) and backprop (GPU). Speeds are reported as clips per second. You can have a look at class `SpeedMonitor` in `train/callbacks.py` for more information Overall the output at each logging interval should look like:

<center>
... Speed (r=<span style="color:#a1e2b7">1.53e+5</span> f=<span style="color:#f3e27a">9.84e+1</span> b=<span style="color:#ff7d75">1.78e+1</span>) ...
</center>


Colours are used in order to give a quick understanding if the speed is in general <span style="color:#a1e2b7">fast</span>, <span style="color:#f3e27a">average</span> or <span style="color:#ff7d75">slow</span>. The limits for each of the speeds are defined as:

- reading (r)  <span style="color:#ff7d75">100</span> < <span style="color:#f3e27a">3000</span> < <span style="color:#a1e2b7">inf</span>

- forward (f)  <span style="color:#ff7d75">50</span> < <span style="color:#f3e27a">300</span> < <span style="color:#a1e2b7">inf</span>

- backward (r)  <span style="color:#ff7d75">20</span> < <span style="color:#f3e27a">200</span> < <span style="color:#a1e2b7">inf</span>

**Note that the biggest factor for speeds is the video/clip size rather than the video size**. Reading speeds will fall at average/slow speeds only during `Dataloader` initialisations.


#### Batch size \& lr monitoring

Along scores, speed the batch size and learning rate are also monitored at each logging interval. This is especially useful for using circles.

## Hardware specifications

All experiments were run in a AMD Threadripper 2950X (64GB) system with 4x NVIDA 2080 Ti GPUs.

## Citation

## Licence

MIT

## Contact

Alexandros Stergiou

a dot g dot stergiou at uu dot nl (a.g.stergiou@uu.nl)

Any queries or suggestions are much appreciated!
