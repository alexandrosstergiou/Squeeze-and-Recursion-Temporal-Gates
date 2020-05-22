'''
---  I M P O R T  S T A T E M E N T S  ---
'''
from PIL import Image
import os
import numpy as np
from array import array
import glob
import sqlite3
import cv2
import multiprocessing
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor as Pool


'''
---  S T A R T  O F  F U N C T I O N  F I L E 2 S Q L  ---

    [About]
        Worker function that takes as argument the frames directory and converts all JPEG images inside that directory to SQLite3 BLOBS. For inode preservation after a file has been read it is deleted. The function will iteratively populate the `frames.db` file of the given directory with all the frames.
    [Args]
        - video_i: String that determines the directory which includes all the video frames in a .jpg format.
    [Returns]
        - None
'''

def file2sql(video_i):
    # User feedback
    print('-> processing {}'.format(video_i))
    # Get database path
    filename_db = os.path.join(video_i,'frames.db')
    # Open connection for writing the frames
    con = sqlite3.connect(filename_db, timeout=0.1)
    cur = con.cursor()

    # Frame iteration
    for file_i in glob.glob(video_i+'/*.jpg'):
    	# Read image as a binary blob
        with open(file_i, 'rb') as f:
             image_bytes = f.read()
        f.close()

        # Decode raw bytes to get image size
        nparr  = np.fromstring(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_size = img_np.shape[1]

        # Extract file video and frame # without extension
        objid = os.path.join(video_i.split('/')[-1],file_i.split('/')[-1].split('.')[0])

        # Insert image and data into table
        cur.execute("insert into Images VALUES(?,?,?)", (objid,sqlite3.Binary(image_bytes),image_size))
        # Remove frame file (can be commented out in the case of saving the .jpg frames)
        os.remove(file_i)

    # Commit changes and close connection
    con.commit()
    cur.close()
    con.close()
'''
---  E N D  O F  F U N C T I O N  F I L E 2 S Q L  ---
'''





'''
---  S T A R T  O F  F U N C T I O N  W O R K E R  ---

    [About]
        Main worker function for saving .mp4 videos as .jpg frames (also creates a frames.db SQL database per video).
    [Args]
        - file_i: String for the filepath.
    [Returns]
        - None
'''
def worker(file_i):
    # Feedback message
    print('process #{} is converting file: {}'.format(multiprocessing.current_process().name,file_i))
    sys.stdout.flush()

    # Only consider videos
    if '.mp4' not in file_i:
      return

    # Get file without .mp4 extension
    name, ext = os.path.splitext(file_i)
    name = os.path.join(*(name.split(os.path.sep)[1:]))

    # Create destination directory for video
    dst_directory_path = os.path.join(dst_dir, name)

    try:
        # Case that video path already exists
        if os.path.exists(dst_directory_path):
            # Case that frames have already been extracted
            if not os.path.exists(os.path.join(dst_directory_path, 'frame_00001.jpg')):
                subprocess.call('rm -r {}'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.makedirs(dst_directory_path)
            else:
                return
        else:
            os.makedirs(dst_directory_path)
            filename_db = os.path.join(dst_directory_path,'frames.db')
            create_db(filename_db)
    except:
        print('Exception')
        print(dst_directory_path)
        return

    # FFMPEG frame extraction
    subprocess.call(['ffmpeg', '-i', file_i, '-hide_banner', '-loglevel', 'quiet', '-vf', 'scale=-1:360', '{}/frame_%05d.jpg'.format(dst_directory_path)])

'''
---  E N D  O F  F U N C T I O N  W O R K E R  ---
'''




'''
---  S T A R T  O F  F U N C T I O N  C R E A T E _ D B  ---

    [About]
        Simple SQL database creator.
    [Args]
        - filename: String for the database filename.
    [Returns]
        - None
'''
def create_db(filename):
    db = sqlite3.connect(filename)
    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS Images")
    cursor.execute("CREATE TABLE Images(ObjId STRING, frames BLOB, size INT)")
    db.commit()
    db.close()
'''
---  E N D  O F  F U N C T I O N  C R E A T E _ D B  ---
'''


if __name__ == '__main__':

    start = time.time()
    #SQLITE
    base_dir = 'kinetics/'
    dst_dir = 'kinetics_videos/jpg'

    #--- Extract files from folder following pattern
    files   = glob.glob(base_dir+"*/*.mp4")
    n_files = len(files)
    print('Number of files in folder: ', n_files)


    for c in os.listdir(base_dir):

        # --- FRAME EXTRACTION IS DONE HERE
        base_files = glob.glob(os.path.join(base_dir,c)+"/*.mp4")
        try:
            with Pool() as p1:
                p1.map(worker, base_files)

        except KeyboardInterrupt:
            print ("Caught KeyboardInterrupt, terminating")
            p1.terminate()
            p1.join()

        # --- DATABASE POPULATION IS DONE HERE
        dist_files = glob.glob(os.path.join(dst_dir,c)+"/*")
        try:
            with Pool() as p2:
                p2.map(file2sql, dist_files)

        except KeyboardInterrupt:
            print ("Caught KeyboardInterrupt, terminating")
            p2.terminate()
            p2.join()

    end = time.time()
    print('Conversion to SQLite database was sucessful in %d secs' %(end-start))
