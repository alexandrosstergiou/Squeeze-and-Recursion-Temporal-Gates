'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()
from decimal import Decimal

'''
===  S T A R T  O F  C L A S S  C A L L B A C K ===

    [About]

        Container class for creating claaback methods.

    [Init Args]

        - with_header: Boolean for determining if the epoch number and batch size is also to be printed.

    [Methods]

        - __init__ : Class initialiser
        - __call__: Function that should be implemented based on the sub-class functionality. It will raise by
        default a `NotImplementedError` as the main functionality should be determed by the subclasses.
        - header: Function for appending the header (i.e. epoch # and batch #) to the output string.

'''
class Callback(object):

    def __init__(self, with_header=False):
        self.with_header = with_header

    def __call__(self):
        raise NotImplementedError('Must be implemented in child classes!')

    def header(self, epoch=None, batch=None):
        str_out = ""
        if self.with_header:
            if epoch is not None:
                str_out += "Epoch {:s} ".format(("[%d]"%epoch).ljust(5, ' '))
            if batch is not None:
                str_out += "Batch {:s} ".format(("[%d]"%batch).ljust(6, ' '))
        return str_out
'''
===  E N D  O F  C L A S S  C A L L B A C K ===
'''


'''
===  S T A R T  O F  C L A S S  C A L L B A C K L I S T ===

    [About]

        Callback child class responsible for creating a list of callbacks.

    [Init Args]

        - args: Iterable that contains Callback class items.
        - with_header: Boolean for determining if the epoch number and batch size is also to be printed.

    [Methods]

        - __init__ : Class initialiser
        - __call__: Function that appends all the outputs from the Callback objects of the list into a single string. In the case that the silent variable is False, the string will also be logged.

'''
class CallbackList(Callback):

    def __init__(self, *args, with_header=True):
        super(CallbackList, self).__init__(with_header=with_header)
        assert all([issubclass(type(x), Callback) for x in args]), \
                "Callback inputs illegal: {}".format(args)
        self.callbacks = [callback for callback in args]

    def __call__(self, epoch=None, batch=None, silent=False, **kwargs):
        str_out = self.header(epoch, batch)

        for callback in self.callbacks:
            str_out += callback(**kwargs, silent=True) + " "

        if not silent:
            logging.info(str_out)
        return str_out
'''
===  E N D  O F  C L A S S  C A L L B A C K L I S T ===
'''


'''
===  S T A R T  O F  C L A S S  S P E E D M O N I T O R ===

    [About]

        Callback child class for monitoring batch-wise reading, forward-pass and backward (Gradient Decent)
        speeds. The class can be used for both training and validation as it enables both modes if the
        `backward_elapse` parameter is None.

    [Init Args]

        - with_header: Boolean for determining if the epoch number and batch size is also to be printed.

    [Methods]

        - __init__ : Class initialiser
        - __call__: Function that returns a string with the batch-wise speeds. They are updated in the
        case that the class object is called - e.g. at the end of the batch.

'''
class SpeedMonitor(Callback):

    def __init__(self, with_header=False):
        super(SpeedMonitor, self).__init__(with_header=with_header)

    def __call__(self, read_elapse, forward_elapse, backward_elapse=None, epoch=None, batch=None, silent=False, **kwargs):
        str_out = self.header(epoch, batch)

        if read_elapse is not None:
            # sec./clip -> clip/sec.
            read_freq = 1./read_elapse

            # Colouring
            if read_freq < 100 :
                read = '\033[91m' +'{0:.2e}'.format(Decimal(read_freq)) + '\033[0m'
            elif read_freq < 3000 :
                read = '\033[93m' +'{0:.2e}'.format(Decimal(read_freq)) + '\033[0m'
            else:
                read = '\033[92m' +'{0:.2e}'.format(Decimal(read_freq)) + '\033[0m'

            if forward_elapse is not None:
                # sec./clip -> clip/sec.
                forward_freq = 1./forward_elapse

                # Colouring
                if forward_freq < 50 :
                    forward = '\033[91m' +'{0:.2e}'.format(Decimal(forward_freq)) + '\033[0m'
                elif forward_freq < 300 :
                    forward = '\033[93m' +'{0:.2e}'.format(Decimal(forward_freq)) + '\033[0m'
                else:
                    forward = '\033[92m' +'{0:.2e}'.format(Decimal(forward_freq)) + '\033[0m'

                # Condition detrmines train/val logging/string output
                if backward_elapse is not None:
                    # sec./clip -> clip/sec.
                    backward_freq = 1./backward_elapse

                    # Colouring
                    if backward_freq < 20 :
                        backward = '\033[91m'  +'{0:.2e}'.format(Decimal(backward_freq)) + '\033[0m'
                    elif backward_freq < 200 :
                        backward = '\033[93m' +'{0:.2e}'.format(Decimal(backward_freq)) + '\033[0m'
                    else:
                        backward= '\033[92m' +'{0:.2e}'.format(Decimal(backward_freq)) + '\033[0m'

                    # Print read speed, forward speed and backward speed for a batch (training)
                    str_out += "Speed (r={0} f={1} b={2}) clip/sec ".format(read, forward, backward)
                else:
                    # Print read speed, forward speed for a batch (validation)
                    str_out += "Speed (r={0} f={1}) clip/sec ".format(read, forward)

        if not silent:
            logging.info(str_out)
        return str_out
'''
===  E N D  O F  C L A S S  S P E E D M O N I T O R ===
'''


'''
===  S T A R T  O F  C L A S S  M E T R I C P R I N T E R ===

    [About]

        Callback child class for monitoring batch sizes and learning rates. Primarily useful for
        using cycles while training.

    [Init Args]

        - with_header: Boolean for determining if the epoch number and batch size is also to be printed.

    [Methods]

        - __init__ : Class initialiser
        - __call__: Function that returns a string with the batch size and learning rates.

'''
class MetricPrinter(Callback):

    def __init__(self, with_header=False):
        super(MetricPrinter, self).__init__(with_header=with_header)

    def __call__(self, namevals, epoch=None, batch=None, silent=False, **kwargs):
        str_out = self.header(epoch, batch)
        if namevals is not None:
            for i, nameval in enumerate(namevals):
                name, value = nameval[0]
                # Skip None values (mainly for batch and LR in validations)
                if (value is None):
                    continue
                if (name=='batch_size'):
                    str_out += "{} = ({:5d},{:1d},{:2d},{:3d},{:3d})".format(name,*value)
                elif (name=='lr'):
                    val = Decimal(value)
                    str_out += "{} = {:.2e}".format(name,val)
                else:
                    str_out += "{} = {:.5f}".format(name, value)
                str_out += ", " if i != (len(namevals)-1) else " "

        if not silent:
            logging.info(str_out)
        return str_out
'''
===  E N D  O F  C L A S S  M E T R I C P R I N T E R ===
'''


''' Any tests should be done here...'''
if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    # Test each function
    # [1] Callback
    logging.info("- testing base callback class:")
    c = Callback(with_header=True)
    logging.info(c.header(epoch=1, batch=123))

    # [2] SpeedMonitor
    logging.info("- testing speedmonitor:")
    s = SpeedMonitor(with_header=True)
    s(read_elapse=0.03, forward_elapse=0.01, backward_elapse=0.01, epoch=10, batch=31)
    s = SpeedMonitor(with_header=False)
    s(read_elapse=0.03, forward_elapse=0.01)

    # [5] LR and batch size
    logging.info("- test dict printer")
    d = MetricPrinter(with_header=True)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]], epoch=10, batch=31)
    d = MetricPrinter(with_header=False)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]])

    # [4] DictPrinter
    logging.info("- test dict printer")
    d = MetricPrinter(with_header=True)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]], epoch=10, batch=31)
    d = MetricPrinter(with_header=False)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]])

    # [4] CallbackList
    logging.info("- test callback list")
    c = CallbackList()
    c = CallbackList(SpeedMonitor(), MetricPrinter())
    c(epoch=10, batch=31, sample_elapse=0.3, namevals=[[('acc1',0.123)], [("acc5",0.4453232)]])
