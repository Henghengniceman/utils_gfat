#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
utilities for logging
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import logging.config
import logging


def create_logger(log_file=None, debug=False, event_level=logging.INFO):
    """
    #Process file program
    #:param log_file: route logs
    #:param event_level: level of error
    #:param error_level: level of error
    #:return: var to saves performance information (errors...)
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(levelname)s: %(module)s. %(funcName)s. L%(lineno)s: %(message)s')

    if log_file is None:
        if (logger.hasHandlers()):
            logger.handlers.clear()
        evt_h = logging.StreamHandler(sys.stdout)
        evt_h.setFormatter(log_formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(evt_h)
        logger.propagate = False
    else:
        if not os.path.exists(os.path.dirname(log_file)):
            try:
                os.makedirs(os.path.dirname(log_file))
            except Exception as e:
                raise Exception('No se pudo crear el archivo de logs, compruebe que tiene permiso de escritura en la ruta especificada')
    
        # create file handler for debug mode
        if debug:
            dbg_h = logging.FileHandler(log_file+'.dbg')
            dbg_h.setLevel(logging.DEBUG)
    
        # create console handler with a higher log level
        evt_h = logging.FileHandler(log_file)
        evt_h.setLevel(event_level)
    
        # add the handlers to logger
        if not len(logger.handlers):
            logger.addHandler(evt_h)
            if debug :
                logger.addHandler(dbg_h)
    
        # create formatter and add it to the handlers
        evt_f = logging.Formatter('[%(asctime)s] - %(message)s','%Y%m%d_%H%M')
        evt_h.setFormatter(evt_f)
    
        if debug:
            dbg_f = logging.Formatter('[%(asctime)s] - %(module)s.%(funcName)s(%(lineno)d) - %(levelname)s - %(message)s','%Y%m%d_%H%M')
            dbg_h.setFormatter(dbg_f)
    
    return logger

def get_logger():
    logger = logging.getLogger(__name__)
    if not len(logger.handlers):
        raise Exception('There is no Logger')
    return logger


##from . import utils
#import utils
#
#LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#LOG_DATE_FMT = '%Y-%m-%d %H:%M:%S'
#LOG_DIR = 'logs'
#LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
#LOG_LEVELS_INT = {
#    1: 'CRITICAL',
#    2: 'WARNING',
#    3: 'DEBUG',
#}
#
#
#def init(log_name, log_file, log_file_level, int_log_level):
#    """
#    Configure the logger and start it
#    """
#
#    # check input
#    if int_log_level != 0:
#        log_file_level = LOG_LEVELS_INT[int_log_level]
#
#    log_cmd_level = log_file_level
#
#    # Check the logs directory
#    dir_ = os.path.dirname(log_file)
#    file_ = os.path.basename(log_file)
#    dir_ok = utils.check_dir(dir_)
#    if not dir_ok:
#        print('directory {} does not exist. Creating it'.format(dir_))
#        try:
#            os.makedirs(dir_)
#        except OSError:
#            print('failed to create {}'.format(dir_))
#            print('quitting {}'.format(log_name))
#            sys.exit(1)
#
#    print('debug file : {}'.format(file_))
#    print('console debug level : {}'.format(log_cmd_level))
#    print('file debug level : {}'.format(log_file_level))
#
#    log_dict = {
#        "version": 1,
#        "disable_existing_loggers": False,
#        "formatters": {
#            "simple": {
#                "datefmt": LOG_DATE_FMT,
#                "format": LOG_FMT,
#            }
#        },
#
#        "handlers": {
#            "console": {
#                "class": "logging.StreamHandler",
#                "level": log_cmd_level,
#                "formatter": "simple",
#                "stream": "ext://sys.stdout"
#            },
#            "file_handler": {
#                "class": "logging.handlers.RotatingFileHandler",
#                "level": log_file_level,
#                "formatter": "simple",
#                "filename": os.path.join(dir_, file_),
#                "maxBytes": 10485760,
#                "backupCount": 10,
#                "encoding": "utf8"
#            }
#        },
#
#        "root": {
#            "level": 'DEBUG',
#            "handlers": ["console", "file_handler"]
#        }
#    }
#
#    logger = logging.getLogger(log_name)
#    logging.config.dictConfig(log_dict)
#
#    return logger
#