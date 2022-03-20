"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :    Added module name `CONNECTS` and 'EMBEDDING' to represent all connection modules in network.

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""

from mmcv.utils import Registry, build_from_cfg
# from mmdet.models.builder import build

CONNECTS = Registry('connect')
EMBEDDING = Registry('embedding')


def build_connect(cfg):
    """ Build CONNECTS module

    Args:
        cfg(dict): module configuration

    Returns:
        obj: CONNECTS module
    """
    return build_from_cfg(cfg, CONNECTS)


def build_embedding(cfg):
    """build an embedding module based on cfg.

    Args:
        cfg (obj:`mmcv.Config`): Config file path or the config
            object.

    Returns:
        An Embedding module.
    """
    return build_from_cfg(cfg, EMBEDDING)
