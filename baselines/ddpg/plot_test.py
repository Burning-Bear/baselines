#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-16
# Modified    :   2017-11-16
# Version     :   1.0

import os
import sys
import numpy as np
location = str(os.path.abspath(
    os.path.join(
        os.path.join(os.path.dirname(__file__), os.pardir), os.pardir)
)) + '/'
sys.path.append(location)

location = str(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
) + '/'
sys.path.append(location)

print(sys.path)

import pickle
from SLBDAO import tester
import json
import pprint
import glob
import os
import matplotlib.pyplot as plt
from gym.envs.registration import registry, register



# file_name = './results/2017-11-16 16:11:54.pkl'
# with open(file_name, 'rb') as f:
#     test = pickle.load(f)
#     print(test.super_param)
#     key_list = ['temp network avg return', 'return', 'temp network error']
#     type_list = ['g--', 'r', 'b']
#     # test.multi_print(key_list, type_list)
#     test.multi_print(key_list, type_list)
#     # test.print('return')
# file_name = './results/2017-11-16 15:13:15[64buget 32unit].pkl'
# with open(file_name, 'rb') as f:
#     test = pickle.load(f)
#     print(test.super_param)
#     key_list = ['temp network avg return', 'return', 'temp network error']
#     type_list = ['b--', 'y', 'r']
#     # test.multi_print(key_list, type_list)
#     test.multi_print(key_list, type_list)
#     # test.print('return')
#     test.show()


def files(prefix, curr_dir='./results/', ext='*.pkl', key_list=[], type_list=[]):
    save_prefix = './res/' + prefix + '/'
    pp = pprint.PrettyPrinter(indent=2)
    file_list = []
    for i in glob.glob(os.path.join(curr_dir, ext)):
        has_open = False
        counter = 0

        while not has_open:
            try:
                with open(i, 'rb') as f:
                    test = pickle.load(f)
                    test.print_args()
                    print("key: file , value %s" % test.file)
                    fig = plt.figure(str(i), figsize=(14, 10))
                    test.set_figure(fig)
                    # test.multi_print(key_list, type_list)
                    test.multi_print(key_list, type_list)
                    # test.print('return')
                    test.show()
                    file_name = save_prefix+test.strftims+'.png'
                    test.savefig(file_name, format='png', dpi=100, figsize='26, 24')
                    file_list.append('![]('+'../'+prefix+'/'+test.strftims+'.png'+')')
                    has_open = True
            except Exception as e:
                print("error %s" % e)
                pass
            counter += 1
            if counter >= 100000:
                break
    for item in file_list:
        print(item)
    mix_pic_func(prefix, curr_dir=curr_dir, ext=ext)
    # pp.pprint(file_list)

def mix_pic_func(prefix, curr_dir='./results/', ext='*.pkl', key_list=[], type_list=[],func='summary'):
    save_prefix = './res/' + prefix + '/'
    pp = pprint.PrettyPrinter(indent=2)
    file_list = []
    fig = plt.figure(0, figsize=(12,8))
    mix_data = [[0, [0]]]
    for i in glob.glob(os.path.join(curr_dir, ext)):
        has_open = False
        counter = 0
        while not has_open:
            try:
                with open(i, 'rb') as f:
                    test = pickle.load(f)
                    # test.print_args()
                    # print("key: file , value %s" % test.file)
                    test.set_figure(fig)
                    data = test.get_custom_recorder('avg return')
                    x_name, y_name = data['name']
                    timestep = data[x_name]
                    return_data = data[y_name]
                    def to_mix_data(mix_data, timestep, return_data):
                        search_index = 0
                        for t, r in zip(timestep, return_data):
                            while len(mix_data) > search_index:
                                if mix_data[search_index][0] < t:
                                    if len(mix_data) > search_index + 1:
                                        if abs(mix_data[search_index + 1][0] - t) < 5000:
                                            mix_data[search_index + 1][1].append(r)
                                            # search_index += 1
                                            break
                                        elif mix_data[search_index + 1][0] > t:
                                            mix_data.insert(search_index + 1, [t, [r]])
                                            # search_index += 1
                                            break
                                        else:
                                            search_index += 1
                                    else:
                                        if len(mix_data) == search_index + 1:
                                            mix_data.append([t, [r]])
                                            # search_index += 1
                                            break
                                        else:
                                            search_index += 1
                                else:
                                    search_index += 1
                        if len(mix_data) == search_index:
                            mix_data.append([t, [r]])
                            search_index += 1
                    to_mix_data(mix_data, timestep, return_data)
                    def shrink_timestep(time_step):
                        res = []
                        last = 0
                        last_index = 0
                        for t in timestep:
                            if t - last > 1000:
                                last_index +=1
                                last = t
                            res.append(last_index)
                        return res
                    # shrink_t = shrink_timestep(timestep)
                    def compress_data(shrink_t, ret_d):
                        res_x = []
                        res_y = []
                        shrink_t = np.array(shrink_t)
                        ret_d =np.array(ret_d)
                        for i in range(max(shrink_t)+1):
                            res_x.append(i)
                            res_y.append(max(ret_d[np.where(shrink_t==i)]))
                        return res_x, res_y
                    # res_x, res_y = compress_data(shrink_t, return_data)
                    # plt.plot(shrink_t, return_data, '*--')
                    plt.plot(timestep, return_data, '*--')

                    has_open = True
            except Exception as e:
                print("error %s" % e)
                pass
            counter += 1
            if counter >= 100000:
                break
    x0 = -1
    std_mean_data = []
    time_step = []
    train_scores_mean = []
    train_scores_std = []
    for x,y in mix_data:
        if x >x0:
            x0 = x
            time_step.append(x)
            std_mean_data.append(np.array(y))
            train_scores_mean.append(np.array(y).mean())
            train_scores_std.append(np.array(y).std())
        else:
            raise Exception("error value %s, %s" %(x, y))

    plt.fill_between(time_step, np.array(train_scores_mean) - np.array(train_scores_std),
                     np.array(train_scores_mean) + np.array(train_scores_std), alpha=0.1,
                     color="r")
    plt.plot(time_step, train_scores_mean, 'o-', color="r",
             label="Training score")
    fig.show()
    file_name = save_prefix + ext + '.png'
    fig.savefig(file_name, format='png', dpi=100, figsize='26, 24')
    print('![](' + '../' + prefix + '/' + ext + '.png' + ')')

file_name = './results/'
if len(sys.argv) >= 2:  # [1] is not None:
    ext = sys.argv[1]
else:
    ext = '2017-11-17 18:34:01.pkl'

if len(sys.argv) >= 3:  # [2] is not None:
    file_name = sys.argv[2]

if len(sys.argv) >= 4:
    prefix = sys.argv[3]
if len(sys.argv) >= 5:
    select = sys.argv[4]
else:
    select = 'summary'
if select == 'summary':
    key_list = ['avg return',
                # 'avg_ratios',
                'q',
                'first_q',
                'loss_actor',
                'loss_critic',
                ]

    type_list = ['y*-', 'b*-',  'r*-',  'c--', 'm--']
elif select == 'total-train':
    key_list = ['mix policy error',
                'mix policy test error',
                'return-train']
    type_list = ['g+--', 'r+-', 'y*-']
elif select == 'train times':
    pass

else:
    key_list = ['mix policy error-' + str(select),
                'mix policy test error-'+str(select)]
    type_list = ['g--+', 'r+-']

if select == 'train-times':
    mix_pic_func(prefix, curr_dir=file_name, ext=ext)
else:
    files(prefix, curr_dir=file_name, ext=ext, key_list=key_list, type_list=type_list)
input("enter to finish")
plt.close()
