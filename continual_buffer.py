# coding=utf-8
# Copyright 2020 The Learning-to-Prompt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Learning-to-Prompt governing permissions and
# limitations under the License.
# ==============================================================================
# Modification:
# Added code for replay method
# -- Taeyoung Lee, slcks1@khu.ac.kr
# ==============================================================================
"""Replay buffer for continual learning."""
"""not use distributed system"""

from typing import Sequence

import numpy as np


class ReplayBuffer:
  """Replay buffer for continual learning."""

  def __init__(self, args):
    """Initializes the replay buffer.

    Args:
      continual_config: Configuration for continual learning.
      input_shape: Input shape of the data, e.g., (32, 32, 3) for cifar10
    """
    self.args = args
    self.num_tasks = args.num_tasks
    self.num_classes_per_task = args.classes_per_task
    self.num_samples_per_class = args.num_samples_per_class
    self.num_samples_per_task = self.num_classes_per_task * self.num_samples_per_class
    self.num_replay_batch_ctl = args.num_replay_batch_ctl
    # place holder for all images
    self.data = []

    # specify valid data scope, updated when data goes in!
    self.cursor = 0
    # specify data scope for the old data
    self.old_task_boundary = 0


  @property
  def cur_size(self):
    return self.cursor


  def add_example(self, high_score_idx, one_class_data):
    """Adds examples in this batch to the buffer, according to the index dict."""
    
    for i in high_score_idx:
      self.data.append(one_class_data[i])
      self.cursor += 1


  def get_random_batch(self, task_id):
    """Returns a random batch according to current valid size."""
    replay_index = []
    batch_size = self.args.batch_size
    # if global batch size > current valid size, we just sample with replacement
    replace = False if self.cursor >= batch_size*task_id*20 else True
    if self.args.include_new_task:
      range_limit = self.cursor
    else:
      range_limit = self.old_task_boundary
    random_indices = np.random.choice(
        np.arange(range_limit), size=batch_size*task_id*self.num_replay_batch_ctl, replace=replace)
    for i in random_indices:
      replay_index.append(self.data[i])
    
    return replay_index
