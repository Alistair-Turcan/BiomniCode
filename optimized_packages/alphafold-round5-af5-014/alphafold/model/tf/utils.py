# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for various components."""
import importlib
__all__ = ['tf', 'SeedMaker', 'seed_maker', 'make_random_seed']


def __getattr__(name):
  if name == 'tf':
    module = importlib.import_module('tensorflow.compat.v1')
    globals()['tf'] = module
    return module
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  return sorted(set(globals()) | {'tf'})


class SeedMaker(object):
  """Return unique seeds."""

  def __init__(self, initial_seed=0):
    self.next_seed = initial_seed

  def __call__(self):
    i = self.next_seed
    self.next_seed += 1
    return i


seed_maker = SeedMaker()


def make_random_seed():
  tensorflow = __getattr__('tf') if 'tf' not in globals() else tf
  return tensorflow.random.uniform(
      [2], tensorflow.int32.min, tensorflow.int32.max, tensorflow.int32,
      seed=seed_maker()
  )
