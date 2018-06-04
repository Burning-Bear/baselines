#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-16
# Modified    :   2017-11-16
# Version     :   1.0

import os
import sys
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
import tester
import json
import pprint
import glob
import os
import matplotlib.pyplot as plt
from gym.envs.registration import registry, register
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=5000,
    reward_threshold=4750.0,
)


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
    save_prefix = '../res/' + prefix + '/'
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
                    file_list.append('![]('+'./'+prefix+'/'+test.strftims+'.png'+')')
                    has_open = True
            except Exception as e:
                print("error %s" % e)
                pass
            counter += 1
            if counter >= 10:
                break
    for item in file_list:
        print(item)
    # pp.pprint(file_list)


file_name = './results/'
if len(sys.argv) >= 2:  # [1] is not None:
    ext = sys.argv[1]
else:
    ext = '2017-11-17 18:34:01.pkl'

if len(sys.argv) >= 3:  # [2] is not None:
    file_name = sys.argv[2]

if len(sys.argv) >= 4:
    prefix = sys.argv[3]


key_list = ['temp network avg return',
            'temp network fist error',
            'temp network error',
            'activate point',
            'not activate point',
            'random accept',
            'sample judge',
            'temp network avg return [high]',
            'return']


type_list = ['c--', 'm', 'r', 'go:', 'ko:', 'k+:', 'k:', 'b-', 'y']
files(prefix, curr_dir=file_name, ext=ext, key_list=key_list, type_list=type_list)
input("enter to finish")
plt.close()
