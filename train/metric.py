'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()
import numpy as np


'''
===  S T A R T  O F  C L A S S  E V A L M E T R I C  ===

    [About]

        Object class for calculating average values.

    [Init Args]

        - name: String for the variable name to calculate average value for.

    [Methods]

        - __init__ : Class initialiser
        - update : Function to be implemented by the children sub-classes.
        - reset : Function for resetting the number of instances and the sum of the metric.
        - get : Calculation of the average value based on the number of instances and the provided sum.
        - get_name_value : Function for returning the name(s) and the value(s).
        - check_label_shapes : Function responsible for type and shape checking.

'''
class EvalMetric(object):

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.reset()

    def update(self, preds, labels, losses, lr, batch_size):
        raise NotImplementedError('Must be implemented in child classes!')

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        # case that instances are 0 -> return NaN
        if self.num_inst == 0:
            return (self.name, float('nan'))
        # case that instances are 1 -> return their sum
        if self.num_inst == 1:
            return(self.name, self.sum_metric)
        # case that instances are >1 -> return average
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def check_label_shapes(self, preds, labels):
        # raise if the shape is inconsistent
        if (type(labels) is list) and (type(preds) is list):
            label_shape, pred_shape = len(labels), len(preds)
        else:
            label_shape, pred_shape = labels.shape[0], preds.shape[0]

        if label_shape != pred_shape:
            raise NotImplementedError("")
'''
===  E N D  O F  C L A S S  E V A L M E T R I C  ===
'''


'''
===  S T A R T  O F  C L A S S  M E T R I C L I S T  ===

    [About]

        EvalMetric class for creating a list containing Evalmetric objects.

    [Init Args]

        - name: String for the variable name.

    [Methods]

        - __init__ : Class initialiser
        - update : Function to update the list of EvalMetric objects.
        - reset : Function for resetting the list.
        - get : Function for getting each of the EvalMetric objects in the list.
        - get_name_value : Function for getting the name of the list items.

'''
class MetricList(EvalMetric):
    def __init__(self, *args, name="metric_list"):
        assert all([issubclass(type(x), EvalMetric) for x in args]), \
            "MetricList input is illegal: {}".format(args)
        self.metrics = [metric for metric in args]
        super(MetricList, self).__init__(name=name)

    def update(self, preds, labels, losses=None, lr=None, batch_size=None):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels
        losses = [losses] if type(losses) is not list else losses
        lr = [lr] if type(lr) is not list else lr
        batch_size = [batch_size] if type(batch_size) is not list else batch_size
        for metric in self.metrics:
            metric.update(preds, labels, losses, lr, batch_size)

    def reset(self):
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                metric.reset()
        else:
            logging.warning("No metric defined.")

    def get(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get())
        return ouputs

    def get_name_value(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get_name_value())
        return ouputs
'''
===  E N D  O F  C L A S S  M E T R I C L I S T  ===
'''


'''
===  S T A R T  O F  C L A S S  A C C U R A C Y  ===

    [About]

        EvalMetric class for creating an accuracy estimate.

    [Init Args]

        - name: String for the variable name. Defaults to `accuracy`.
        - topk: Number of top predictions to be used of the score (top-1, top-5 etc.).
        Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - update : Function to update scores.

'''
class Accuracy(EvalMetric):
    def __init__(self, name='accuracy', topk=1):
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def update(self, preds, labels, losses, lr, batch_size):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels

        self.check_label_shapes(preds, labels)
        for pred, label in zip(preds, labels):
            assert self.topk <= pred.shape[1], \
                "topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
            _, pred_topk = pred.topk(self.topk, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))
            self.sum_metric += float(correct.reshape(-1).float().sum(0, keepdim=True).numpy())
            self.num_inst += label.shape[0]
'''
===  E N D  O F  C L A S S  A C C U R A C Y  ===
'''


'''
===  S T A R T  O F  C L A S S  L O S S  ===

    [About]

        EvalMetric class for creating a loss score. The class acts a a `dummy estimate`
        as no further calculations are required for the loss. Instead it is primarily
        used to easily/directly print the loss.

    [Init Args]

        - name: String for the variable name. Defaults to `loss`.

    [Methods]

        - __init__ : Class initialiser
        - update : Function to update scores.

'''
class Loss(EvalMetric):
    def __init__(self, name='loss'):
        super(Loss, self).__init__(name)

    def update(self, preds, labels, losses, lr, batch_size):
        assert losses is not None, "Loss undefined."
        for loss in losses:
            self.sum_metric += float(loss.numpy().sum())
            self.num_inst += 1
'''
===  E N D  O F  C L A S S  L O S S  ===
'''


'''
===  S T A R T  O F  C L A S S  L O S S  ===

    [About]

        EvalMetric class for batch-size used. The class acts a a `dummy estimate`
        as no further calculations are required for the size of the batch. Instead it is primarily
        used to easily/directly print the batch size.

    [Init Args]

        - name: String for the variable name. Defaults to `batch-size`.

    [Methods]

        - __init__ : Class initialiser
        - update : Function used for updates.

'''
class BatchSize(EvalMetric):
    def __init__(self, name='batch-size'):
        super(BatchSize, self).__init__(name)

    def update(self, preds, labels, losses, lrs, batch_sizes):
        assert batch_sizes is not None, "Batch size undefined."
        self.sum_metric = batch_sizes
        self.num_inst = 1
'''
===  E N D  O F  C L A S S  L O S S  ===
'''


'''
===  S T A R T  O F  C L A S S  L E A R N I N G R A T E  ===

    [About]

        EvalMetric class for learning rate used. The class acts a a `dummy estimate`
        as no further calculations are required for the size of the lr. Instead it is primarily
        used to easily/directly print the learning rate.

    [Init Args]

        - name: String for the variable name. Defaults to `lr`.

    [Methods]

        - __init__ : Class initialiser
        - update : Function used for updates.

'''
class LearningRate(EvalMetric):
    def __init__(self, name='lr'):
        super(LearningRate, self).__init__(name)

    def update(self, preds, labels, losses, lrs, batch_sizes):
        assert lrs is not None, "Learning rate undefined."
        self.sum_metric = lrs[-1]
        self.num_inst = 1
'''
===  E N D  O F  C L A S S  L E A R N I N G R A T E  ===
'''

if __name__ == "__main__":
    import torch

    # Test Accuracy
    predicts = [torch.from_numpy(np.array([[0.7, 0.3], [0, 1.], [0.4, 0.6]]))]
    labels   = [torch.from_numpy(np.array([   0,            1,          1 ]))]
    losses   = [torch.from_numpy(np.array([   0.3,       0.4,       0.5   ]))]

    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("input pred:  {}".format(predicts))
    logging.debug("input label: {}".format(labels))
    logging.debug("input loss: {}".format(labels))

    acc = Accuracy()

    acc.update(preds=predicts, labels=labels, losses=losses, lr=0, batch_size=1)

    logging.info(acc.get())

    # Test MetricList
    metrics = MetricList(Loss(name="ce-loss"),
                         Accuracy(topk=1, name="acc-top1"),
                         Accuracy(topk=2, name="acc-top2"),
                         )
    metrics.update(preds=predicts, labels=labels, losses=losses, lr=0, batch_size=1)

    logging.info("------------")
    logging.info(metrics.get())
    acc.get_name_value()
