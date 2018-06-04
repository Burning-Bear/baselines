class TimeStepHolder(object):

    def __init__(self, time, epoch):
        self.__timesteps = time
        self.__outer_epoch = epoch

    def set_outer_epoch(self, epoch):
        self.__outer_epoch = epoch

    def get_outer_epoch(self):
        return self.__outer_epoch

    def inc_outer_epoch(self):
        self.__outer_epoch += 1

    def set_time(self, time):
        self.__timesteps = time

    def inc_time(self):
        self.__timesteps += 1

    def get_time(self):
        return self.__timesteps
