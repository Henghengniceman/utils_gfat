#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
module for arguments managements
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import importlib
import glob
import argparse
import datetime as dt
from itertools import chain

#from . import logs
#from . import utils
import logs
import utils


DATE_FMT = '%Y%m%d'


def check_date_format(input_date):
    """
    Check the format of the date argument
    """

    try:
        dt_date = dt.datetime.strptime(input_date, DATE_FMT)
    except:
        msg = '{} has not the required format (YYYYMMDD)'.format(input_date)
        raise argparse.ArgumentTypeError(msg)

    return dt_date


def check_input_files(input_files):
    """
    check if the input files exist and return a list of the input files found
    """

    list_files = glob.glob(input_files)

    if len(list_files) == 0:
        msg = "No input files found corresponding to the file pattern "
        msg += input_files
        raise argparse.ArgumentTypeError(msg)
    elif len(list_files) == 1:      
        return list_files

    list_ = sorted(list_files)
    list_ = [f for f in chain.from_iterable(list_)]
    print('sorted: %s' % list_)    
    return list_


def read_list_files(file_):
    """load contains of file containing a list of files and returns it as a list"""

    with open(file_, 'r') as f_id:
        files = [line.strip() for line in f_id.readlines()]

    return files


def check_input_lists_files(input_files):
    """check input files lists and return a list of each input
    """

    # check if files exist
    input_list = check_input_files(input_files)
              
    if len(input_list) != 1:
        # build list of list of input files
        inputs_lists = []
        for file_ in input_list:
            # read the list of files
            list_files = read_list_files(file_)
            print(list_files)

            # check if files can be found
            for file_data in list_files:
                tmp = check_input_files(file_data)
                inputs_lists.append(tmp[0])
    else:
        inputs_lists = input_list
    return inputs_lists

def check_output_dir(output_file):
    """
    check if the directory provided for the output file is writable
    """

    output_file = os.path.abspath(output_file)
    out_dir = os.path.dirname(output_file)

    if not utils.check_dir(out_dir):
        msg = "output directory " + out_dir
        msg += " doesn't exist or is not writable"
        raise argparse.ArgumentTypeError(msg)

    return output_file


def check_conf_file(input_conf):
    """check if conf file can be loaded and load it if possible"""

    list_files = glob.glob(input_conf)

    if len(list_files) != 1:
        msg = "error: Configuration file {} cannot be found"
        raise argparse.ArgumentError('conf', msg.format(input_conf))

    # extract path
    file_ = os.path.abspath(list_files[0])
    module_path = os.path.dirname(file_)
    module_name = os.path.splitext(os.path.basename(file_))[0]

    # check if module path is in python path. if not add it
    if module_path not in sys.path:
        sys.path.append(module_path)

    try:
        conf = importlib.import_module(module_name)
    except ImportError:
        msg = 'error: cannot import module {}'
        raise argparse.ArgumentError('conf', msg.format(list_files[0]))

    return conf

def check_output_file(output_list_file):
    """check the file containing the list exists and check that output
    directory is writtable"""

    # check if files exist
    output_list_file = check_input_files(output_list_file)[0]

    # read files
    list_files = read_list_files(output_list_file)

    # check if we can write into the directory
    # if not program will quit
    for file_ in list_files:
        check_output_dir(file_)

    return list_files


def init_argparser(descr, prog_dir, prog_name, version):
    """function to parse input arguments"""

    parser = argparse.ArgumentParser(description=descr)

    # arguments from the workflow
    parser.add_argument('-d', '--date',
                        required=True,
                        type=check_date_format,
                        help='date to process (YYYYMMDD)')
    parser.add_argument('-i', '--input',
                        required=True,
                        action='append',
                        type=check_input_lists_files,
                        help='file(s) containing list of input files')
    parser.add_argument('-o', '--output',
                        required=False,     
                        dest='output_name',
                        default='auto',
                        #type=check_output_file,
                        help='Output file name')
    parser.add_argument('-t', '--temp',
                        required=True,
                        type=check_output_dir,
                        default=os.path.join(prog_dir, 'temp'),
                        dest='output_dir',
                        help='working directory')
    parser.add_argument('-n', '--int_log_level',
                        choices=[1, 2, 3],
                        default=0,
                        type=int,
                        help='level of log (1: CRITICAL, 2: WARNING, 3: DEBUG)')
    parser.add_argument('-v', '--version',
                        action='version',
                        version=version.split('.')[0],
                        help='show major version of code')
    parser.add_argument('-log',
                        required=False,
                        default=os.path.join(prog_dir, 'logs', prog_name + '.log'),
                        help='File where logs will be saved')
    parser.add_argument('-log_level',
                        required=False,
                        choices=logs.LOG_LEVELS,
                        default=logs.LOG_LEVELS[1],
                        help='Level of logs')

    # %print("parser %s" % parser)
    return parser
