#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-12
# Modified    :   2017-11-12
# Version     :   1.0
import matplotlib.pyplot as plt
import pickle
import time
from gym import wrappers
import os
import sys
MAX_ITER = 2000


class Tester(object):

    def __init__(self, episodes, period, env, time_step_holder, file, session, save_object=True):
        """
        Tester is a class can store all of record and parameter in your experiment.
        It will be saved as a pickle. you can load it and plot those records to curve picture.

        Parameters
        ----------
        episodes : int
            repeat time per evalate test.

        period : int
            test period

        env : construct from gym.make
            test environment

        time_step_holder: common.variable.TimeStepHolder
            glabal time step holder

        file: string
            file to store pickle
        """
        self.period = period
        self.time_step_holder = time_step_holder
        self.episodes = episodes
        self.__custom_recorder = {}
        self.super_param = {}
        self.session = session
        self.strftims = str(time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()))
        self.file = file+'/' + self.strftims + '.pkl'
        self.env = env
        self.save_object = save_object
        self.__custom_data = {}
        # sys.stdout = open('./log/'+self.strftims+'.log', 'w')
        # wrappers.Monitor(
        # env, VIDEO_PREFIX_PATH + self.strftims, video_callable=lambda: True)

    def reset_tester(self):
        self.__custom_recorder = {}
        self.__custom_data = {}
        self.strftims = str(time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()))
        self.file = file+'/' + self.strftims + '.pkl'

    def check_and_test(self, agent, use_temp=False, always=False):
        if self.time_step_holder.get_time() % self.period == 0 or always:
            e = 0
            avg_return = 0
            current_return_list = []
            state_num_list = []
            while e < self.episodes:
                itera_num = 0
                ob = self.env.reset()
                current_return = 0
                terminal = False
                state_number = 0
                while not terminal:
                    if use_temp:
                        action = agent.act_temp(ob)
                    else:
                        action, _ = agent.act(True, ob)
                    ob, reward, terminal, _ = self.env.step(action)
                    current_return += reward
                    itera_num += 1
                    state_number += 1
                    if itera_num > MAX_ITER:
                        itera_num = 0
                        break
                state_num_list.append(state_number)
                if e % int(self.episodes/10) == 0:
                    print("end episodes %s" % e)
                current_return_list.append(current_return)
                e += 1

            avg_return = sum(current_return_list) * 1.0 / \
                len(current_return_list)
            avg_state = sum(state_num_list) * 1.0 / len(state_num_list)
            print("file :%s[test] use temp is %s end test time %s, return %s, avg_state %s" %
                  (self.file, use_temp, self.time_step_holder.get_time(), avg_return, avg_state))
            self.add_custom_record("return", self.time_step_holder.get_time(
            ), avg_return, x_name='time_step', y_name='return avg %s episodes' % self.episodes)

            return avg_return

    def end_check(self, satisfied_length, end_point):
        """
        Check if average return value of recent satisfied_length number larger than end_point

        Parameters
        ----------
        satisfied_length : int

        end_point : int

        """
        if end_point == None:
            return False
        else:
            if 'return' not in self.__custom_recorder:
                return False
            length = len(self.__custom_recorder['return'][
                         self.__custom_recorder['return']['name'][1]])
            to_cal_return = self.__custom_recorder['return'][
                self.__custom_recorder['return']['name'][1]][length-satisfied_length:]
            avg_ret = sum(to_cal_return) / (len(to_cal_return) + 1)
            print(
                "-----------------------------------recent return is %s -------------------------" % avg_ret)
            if avg_ret >= end_point:
                return True
            else:
                return False

    def set_super_param(self, **argkw):
        """
        This method is to record all of super parameters to test object.

        Place pass your parameters as follow format:
            self.set_super_param(param_a=a,param_b=b)

        Note: It is invalid to pass a local object to this function.

        Parameters
        ----------
        argkw : key-value 
            for example: self.set_super_param(param_a=a,param_b=b)

        """
        self.super_param = argkw

    def add_custom_record(self, key, x, y, x_name, y_name):
        """
        This model is to add record to specific 'key'.
        After that, you can load plk file and call 'print' function to print x-y curve. 

        Parameters
        ----------
        key : string
            identify your curve

        x: float or int
            x value to be added.

        y: float or int
            y value to be added.

        x_name: string
            name of x axis, will be displayed when call print function

        y_name: string
            name of y axis, will be displayed when call print function
        """
        if key not in self.__custom_recorder:
            self.__custom_recorder[key] = {}
            self.__custom_recorder[key][x_name] = [x]
            self.__custom_recorder[key][y_name] = [y]
            self.__custom_recorder[key]['name'] = [x_name, y_name]
        else:
            self.__custom_recorder[key][x_name].append(x)
            self.__custom_recorder[key][y_name].append(y)
        self.serialize_object_and_save()

    def add_custom_data(self, key, data, dtype):
        if key not in self.__custom_data:
            if isinstance(dtype, list):
                self.__custom_data[key] = [data]
            else:
                self.__custom_data[key] = data
        else:
            if isinstance(dtype, list):
                self.__custom_data[key].append(data)
            else:
                self.__custom_data = data

    def __print_unit(self, key, style):
        if key in self.__custom_recorder:
            x_name, y_name = self.__custom_recorder[key]['name']
            plt.plot(self.__custom_recorder[key][
                x_name], self.__custom_recorder[key][y_name], style)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
        else:
            print('[Tester plot wrong] key %s not exist. ' % key)

    def print(self, key, style):
        """
        plot single curve store in key.

        Parameters
        ----------
        key : string
            the id which has been set when call self.add_custom_record function

        style : string
            define the style of curve to be plot, for example : 'r-', 'go'

        """
        self.__print_unit(key, style)

    def multi_print(self, key_list, style_list):
        """
        plot several curve store in key_list.

        Parameters
        ----------
        key_list : array of string
            the id list which has been set when call self.add_custom_record function

        type_list: array of string


        """
        for key, style in zip(key_list, style_list):
            self.__print_unit(key, style)

    def serialize_object_and_save(self):
        """
        This method is to save test object to a pickle.
        This method will be call every time you call add_custom_record or other record function like self.check_and_test
        """
        # remove object which can is not serializable
        if self.save_object:
            sess = self.session
            self.session = None
            with open(self.file, 'wb') as f:
                pickle.dump(self, f)
            self.session = sess

    def set_figure(self, fig):
        self.fig = fig

    def show(self):
        self.fig.show()

    def savefig(self, *argv, **argkw):
        self.fig.savefig(*argv, **argkw)

    def print_args(self):
        for key, value in self.super_param.items():
            print("key: %s, value: %s" % (key, value))

    def save_suctom_object(self, class_name, ob):
        with open(self.file + '-'+class_name, 'wb') as f:
            pickle.dump(ob, f)
