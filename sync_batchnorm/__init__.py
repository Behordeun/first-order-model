# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Muhammad Abiodun Sulaiman
# Email  : prince.behordeun@gmail.com
# Date   : 13/09/2020
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback
