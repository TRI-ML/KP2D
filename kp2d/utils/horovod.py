# Copyright 2020 Toyota Research Institute.  All rights reserved.

try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def hvd_init():
    if HAS_HOROVOD:
        hvd.init()
    return HAS_HOROVOD

def rank():
    return hvd.rank() if HAS_HOROVOD else 0

def local_rank():
    return hvd.local_rank() if HAS_HOROVOD else 0

def world_size():
    return hvd.size() if HAS_HOROVOD else 1
