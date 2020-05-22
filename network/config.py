'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()

'''
---  S T A R T  O F  F U N C T I O N  G E T _ C O N F I G ---
    [About]
        Function for configuring the meand and standard deviation for the model
    [Args]
        - name: String for the network name.
    [Returns]
        - config: Dictionary that includes a `mean` and `std` terms spcifying the mean and standard deviation.
'''
def get_config(name, **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    logging.info("Preprocessing:: using default mean & std.")
    config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    config['std'] = [1 / (.0167 * 255)] * 3


    logging.info("data:: {}".format(config))
    return config
'''
---  E N D  O F  F U N C T I O N  G E T _ C O N F I G ---
'''
