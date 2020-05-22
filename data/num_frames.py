import sqlite3
import glob
import os,shutil


def find_frames(dataset_path):
    for video_file in glob.glob(dataset_path+'/jpg/*/*/*.db'):
        print(video_file)
        # Get number of video frames
        con = sqlite3.connect(video_file)
        cur = con.cursor()
        cur.execute("SELECT COUNT (*) FROM Images")
        result=cur.fetchone()
        n_frames = result[0]
        cur.close()
        con.close()


        # Sanity check
        if n_frames<=0:
            print('{} does not have any frames'.format(video_file))
            shutil.rmtree(video_file.split('/frames')[0])
            continue

        with open(os.path.join(video_file.split('/frames')[0], 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))
