'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()

'''
---  S T A R T  O F  F U N C T I O N  G E T _ C O N F I G ---
    [About]
        Function for configuring the number of classes based on the dataset used.
    [Args]
        - name: String for the dataset name to be used for training.
        - variant: Iteger for the Kinetics dataset for selecting the dataset variation (400,600,700). We also included
          Mini-Kinetics dataset with 200 classes.
    [Returns]
        - config: Dictionary that includes a `num_classes` term that specifies how many classes the dataset has.
'''
def get_config(name,variant=None):

    config = {}
    print(name)
    # Case 1: UCF-101
    if 'UCF101' in name.upper():
        config['num_classes'] = 101
    # Case 2: HMDB-51
    elif 'HMDB51' in name.upper():
        config['num_classes'] = 51
    # Case 3: Kinetics - if no variant is used, the
    elif 'KINETICS' in name.upper():
        if variant is None:
            config['num_classes'] = 700
        elif variant == 700:
            config['num_classes'] = 700
        elif variant == 600:
            config['num_classes'] = 600
        elif variant == 400:
            config['num_classes'] = 400
        elif variant == 200:
            config['num_classes'] = 200
        else:
            logging.warning('Kinetics varinat not supported initialising with 700 classes')
            config['num_classes'] = 700
    # Case 4: Diving 48
    elif 'DIVING48' in name.upper():
        config['num_classes'] = 48
    # Case 5: Moments in Time
    elif 'MOMENTS' in name.upper():
        config['num_classes'] = 339
    # Case 6: HACS
    elif 'HACS' in name.upper():
        config['num_classes'] = 200
    else:
        logging.error("Configs for dataset '{}'' not found/supported".format(name))
        raise NotImplemented

    logging.debug("Target dataset: '{}', configs: {}".format(name.upper(), config))

    return config
'''
---  E N D  O F  F U N C T I O N  G E T _ C O N F I G ---
'''
