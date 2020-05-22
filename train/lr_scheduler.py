'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()


'''
===  S T A R T  O F  C L A S S  L R S C H E D U L E R ===

    [About]

        Object class for creating a learning rate scheduler. Primarily used as container without any
        special functionality apart from defying the step counter parameters and the base learning rate.

    [Init Args]

        - step_counter: Integer for the number osteps that have been taken. To be changed in cases
        such as continouing training from different epochs. defaults to 0.
        - base_lr: Float for the initial learning rate. defaults to 0.01.

    [Methods]

        - __init__ : Class initialiser
        - update: Function to be implemented in children classes. Currently thows a placeholder `NotImplementedError`.
        - get_lr: Function to return the learning rate.

'''
class LRScheduler(object):

    def __init__(self, step_counter=0, base_lr=0.01):
        self.step_counter = step_counter
        self.base_lr = base_lr

    def update(self):
        raise NotImplementedError('Must be implemented in child classes!')

    def get_lr(self):
        return self.lr
'''
===  E N D  O F  C L A S S  L R S C H E D U L E R ===
'''


'''
===  S T A R T  O F  C L A S S  M U L T I F A C T O R S C H E D U L E R ===

    [About]

        LRScheduler child class for the main definition of changes in the learning rate. Also takes to
        account cicles and reduces the learning rate accordingly.

    [Init Args]

        - steps: List of Integers for the epochs of which the learning rate will be reduced by a fixed rate.
        - iteration_steps: List of Itegers of the interation numbers that the learning rate should be reduced
        based on the cicle schedule. Defaults to None.
        - iterations_per_epoch: Integer for the total number of itartations per epochs. To primarily be used
        in conjunction with `iteration_steps`. Defaults to None.
        - base_lr: Float for the initial learning rate. defaults to 0.01.
        - factor: Float for the learning rate reduction factor. This factor is used in the epochs and is not
        used to reduce the lr per cicle, which is 0.5 as defined by class variable `self.cicle_factor`.
        Defaults to 0.1.
        - step_counter: Integer for counting the number of steps. Defaults to 0.

    [Methods]

        - __init__ : Class initialiser
        - adjust_on_cicle: Function for adjusting the learning rate based on long/short cicles.
        - update: Function to update the learning rate based on the epoch number and iteration number.
        - get_lr: Function to return the learning rate.

'''
class MultiFactorScheduler(LRScheduler):

    def __init__(self, steps, iteration_steps=None, iterations_per_epoch=None, base_lr=0.01, factor=0.1, step_counter=0):
        super(MultiFactorScheduler, self).__init__(step_counter, base_lr)
        assert isinstance(steps, list) and len(steps) > 0
        for i, _step in enumerate(steps):
            if i != 0 and steps[i] <= steps[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")

        logging.info("Iter %d: start with learning rate: %0.5e (next lr step: %d)" \
                                % (self.step_counter, self.base_lr, steps[0]))
        self.steps = steps
        self.steps_dict = iteration_steps
        self.steps_per_epoch = iterations_per_epoch
        self.factor = factor
        self.cicle_factor = 0.5 #factor
        self.lr = self.base_lr
        self.cursor = 0



    def adjust_on_cicle(self):

        if (self.steps_dict is None):
            logging.info('No cicle type defined')

            if self.lr < 1e-6:
                return 1e-6
            return self.lr

        if (self.step_counter%self.steps_per_epoch) in self.steps_dict['long_0']:
            self.lr = self.base_lr

        elif (self.step_counter%self.steps_per_epoch) in self.steps_dict['long_1']:
            self.lr=self.base_lr*self.cicle_factor

        elif (self.step_counter%self.steps_per_epoch) in self.steps_dict['long_2']:
            self.lr=self.base_lr*(self.cicle_factor**2)

        else:
            self.lr=self.base_lr*(self.cicle_factor**3)

        if self.lr < 1e-6:
            return 1e-6
        return self.lr

    def update(self):
        self.step_counter += 1
        if self.cursor >= len(self.steps):
            return self.adjust_on_cicle()
        while self.steps[self.cursor] < self.step_counter:
            self.base_lr *= self.factor
            self.cursor += 1

            # message
            if self.cursor >= len(self.steps):
                logging.info("Iter: %d, change learning rate to %0.5e for step [%d:Inf)" \
                                % (self.step_counter-1, self.base_lr, self.step_counter-1))
                return self.adjust_on_cicle()
            else:
                logging.info("Iter: %d, change learning rate to %0.5e for step [%d:%d)" \
                                % (self.step_counter-1, self.base_lr, self.step_counter-1, \
                                   self.steps[self.cursor]))
        return self.adjust_on_cicle()
'''
===  E N D  O F  C L A S S  M U L T I F A C T O R S C H E D U L E R ===
'''

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    # test LRScheduler()
    logging.info("testing basic class: LRScheduler()")
    LRScheduler()

    # test MultiFactorScheduler()
    logging.info("testing basic class: MultiFactorScheduler()")
    start_point = 2
    lr_scheduler = MultiFactorScheduler(step_counter=start_point,base_lr=0.1,steps=[2, 14, 18],factor=0.1)

    for i in range(start_point, 22):
        logging.info("id = {}, lr = {:f}".format(i, lr_scheduler.update()))
