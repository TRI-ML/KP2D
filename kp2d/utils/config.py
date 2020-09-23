# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os

from yacs.config import CfgNode

from kp2d.configs.base_config import get_cfg_defaults


def get_default_config(cfg_default):
    """Get default configuration from file"""
    config = get_cfg_defaults()
    config.merge_from_list(['default', cfg_default])
    return config

def merge_cfg_file(config, cfg_file=None):
    """Merge configuration file"""
    if cfg_file is not None:
        config.merge_from_file(cfg_file)
        config.merge_from_list(['config', cfg_file])
    return config

def merge_cfgs(original, override):
    """
    Updates CfgNode with information from another one

    Parameters
    ----------
    original : CfgNode
        Original configuration node
    override : CfgNode
        Another configuration node used for overriding

    Returns
    -------
    updated : CfgNode
        Updated configuration node
    """
    for key, value in original.items():
        if key in override.keys():
            if is_cfg(value): # If it's a configuration node, recursion
                original[key] = merge_cfgs(original[key], override[key])
            else: # Otherwise, simply update key
                original[key] = override[key]
    return original

def parse_train_config(cfg_default, cfg_file):
    """
    Parse model configuration for training

    Parameters
    ----------
    cfg_default : str
        Default **.py** configuration file
    cfg_file : str
        Configuration **.yaml** file to override the default parameters

    Returns
    -------
    config : CfgNode
        Parsed model configuration
    """
    # Loads default configuration
    config = get_default_config(cfg_default)
    # Merge configuration file
    config = merge_cfg_file(config, cfg_file)
    # Return prepared configuration
    return config

def parse_train_file(file):
    """
    Parse file for training

    Parameters
    ----------
    file : str
        File, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file

    Returns
    -------
    config : CfgNode
        Parsed model configuration
    ckpt : str
        Parsed checkpoint file
    """
    # If it's a .yaml configuration file
    if file.endswith('yaml'):
        cfg_default = 'configs/default_config'
        return parse_train_config(cfg_default, file)
    # We have a problem
    else:
        raise ValueError('You need to provide a .yaml or .ckpt to train')
