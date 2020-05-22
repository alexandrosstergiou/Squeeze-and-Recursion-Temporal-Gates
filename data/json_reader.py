'''
---  I M P O R T  S T A T E M E N T S  ---
'''

import json
import os
import time
import collections


'''
---  S T A R T  O F  F U N C T I O N  L O A D _ J S O N  ---

    [About]

        Function for returning full video filepaths dictionary.

    [Args]

        - json_path: String for the full json path.
        - parent_path: String of the full path of the dataset.

    [Returns]

        - paths_dir: Dictionary containing the the folderpaths of videos as keys and their labels as values.

'''
def load_json(json_path,parent_path):
    # Dictonary initialisation.
    paths_dir = {}
    class_names = collections.OrderedDict()

    # Loading video addresses.
    with open(json_path) as f:
        videos = json.load(f)


    start_time = time.time()
    # Iterate over json elements
    for i,vid in enumerate(videos):
        l = videos[vid]['annotations']['label']
        if not l in class_names:
            class_names[l] = len(class_names)+1
        path = os.path.join(parent_path,os.path.join(l,vid))
        path = path+'_'+str(int(videos[vid]['annotations']['segment'][0])).zfill(6)+'_'+str(int(videos[vid]['annotations']['segment'][1])).zfill(6)
        paths_dir[path] = class_names[l]

    return paths_dir
'''
---  E N D  O F  F U N C T I O N  L O A D _ J S O N  ---
'''
