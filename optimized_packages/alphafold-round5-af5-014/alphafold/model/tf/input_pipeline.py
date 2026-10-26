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

"""Feature pre-processing input pipeline for AlphaFold."""

from alphafold.model.tf import data_transforms
from alphafold.model.tf import shape_placeholders
from jax import tree
import tensorflow.compat.v1 as tf


# Pylint gets confused by the curry1 decorator because it changes the number
#   of arguments to the function.
# pylint:disable=no-value-for-parameter


NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def nonensembled_map_fns(data_config):
  """Input pipeline functions which are not ensembled."""
  common_cfg = data_config.common

  map_fns = [
      data_transforms.correct_msa_restypes,
      data_transforms.cast_64bit_ints,
      data_transforms.squeeze_features,
      # Keep to not disrupt RNG.
      data_transforms.randomly_replace_msa_with_unknown(0.0),
      data_transforms.make_seq_mask,
      data_transforms.make_msa_mask,
      # Compute the HHblits profile if it's not set. This has to be run before
      # sampling the MSA.
      data_transforms.make_hhblits_profile,
      data_transforms.make_random_crop_to_size_seed,
  ]
  if common_cfg.use_templates:
    map_fns.extend([
        data_transforms.fix_templates_aatype,
        data_transforms.make_template_mask,
        data_transforms.make_pseudo_beta('template_'),
    ])
  map_fns.extend([
      data_transforms.make_atom14_masks,
  ])

  return map_fns


def ensembled_map_fns(data_config):
  """Input pipeline functions that can be ensembled and averaged."""
  common_cfg = data_config.common
  eval_cfg = data_config.eval

  map_fns = []

  if common_cfg.reduce_msa_clusters_by_max_templates:
    pad_msa_clusters = eval_cfg.max_msa_clusters - eval_cfg.max_templates
  else:
    pad_msa_clusters = eval_cfg.max_msa_clusters

  max_msa_clusters = pad_msa_clusters
  max_extra_msa = common_cfg.max_extra_msa

  map_fns.append(data_transforms.sample_msa(max_msa_clusters, keep_extra=True))

  if 'masked_msa' in common_cfg:
    # Masked MSA should come *before* MSA clustering so that
    # the clustering and full MSA profile do not leak information about
    # the masked locations and secret corrupted locations.
    map_fns.append(
        data_transforms.make_masked_msa(
            common_cfg.masked_msa, eval_cfg.masked_msa_replace_fraction
        )
    )

  if common_cfg.msa_cluster_features:
    map_fns.append(data_transforms.nearest_neighbor_clusters())
    map_fns.append(data_transforms.summarize_clusters())

  # Crop after creating the cluster profiles.
  if max_extra_msa:
    map_fns.append(data_transforms.crop_extra_msa(max_extra_msa))
  else:
    map_fns.append(data_transforms.delete_extra_msa)

  map_fns.append(data_transforms.make_msa_feat())

  crop_feats = dict(eval_cfg.feat)

  if eval_cfg.fixed_size:
    map_fns.append(data_transforms.select_feat(list(crop_feats)))
    map_fns.append(
        data_transforms.random_crop_to_size(
            eval_cfg.crop_size,
            eval_cfg.max_templates,
            crop_feats,
            eval_cfg.subsample_templates,
        )
    )
    map_fns.append(
        data_transforms.make_fixed_size(
            crop_feats,
            pad_msa_clusters,
            common_cfg.max_extra_msa,
            eval_cfg.crop_size,
            eval_cfg.max_templates,
        )
    )
  else:
    map_fns.append(data_transforms.crop_templates(eval_cfg.max_templates))

  return map_fns


def ensembled_apply_fns(data_config, draws):
  """Ensembled map fns consuming pre-drawn RNG values (pure, parallel-safe).

  Mirrors ensembled_map_fns exactly, except the stateful draws of
  sample_msa / make_masked_msa / crop_extra_msa are replaced by the
  bitwise-identical pre-drawn values from data_transforms.ensemble_predraw.
  """
  common_cfg = data_config.common
  eval_cfg = data_config.eval

  map_fns = []

  if common_cfg.reduce_msa_clusters_by_max_templates:
    pad_msa_clusters = eval_cfg.max_msa_clusters - eval_cfg.max_templates
  else:
    pad_msa_clusters = eval_cfg.max_msa_clusters

  max_msa_clusters = pad_msa_clusters
  max_extra_msa = common_cfg.max_extra_msa

  map_fns.append(lambda d: data_transforms.sample_msa_apply(
      d, max_msa_clusters, True, draws['index_order']))

  if 'masked_msa' in common_cfg:
    map_fns.append(lambda d: data_transforms.make_masked_msa_apply(
        d, draws['mask_position'], draws['bert_draw']))

  if common_cfg.msa_cluster_features:
    map_fns.append(data_transforms.nearest_neighbor_clusters())
    map_fns.append(data_transforms.summarize_clusters())

  if max_extra_msa:
    map_fns.append(lambda d: data_transforms.crop_extra_msa_apply(
        d, draws['extra_select']))
  else:
    map_fns.append(data_transforms.delete_extra_msa)

  map_fns.append(data_transforms.make_msa_feat())

  crop_feats = dict(eval_cfg.feat)

  if eval_cfg.fixed_size:
    map_fns.append(data_transforms.select_feat(list(crop_feats)))
    map_fns.append(
        data_transforms.random_crop_to_size(
            eval_cfg.crop_size,
            eval_cfg.max_templates,
            crop_feats,
            eval_cfg.subsample_templates,
        )
    )
    map_fns.append(
        data_transforms.make_fixed_size(
            crop_feats,
            pad_msa_clusters,
            common_cfg.max_extra_msa,
            eval_cfg.crop_size,
            eval_cfg.max_templates,
        )
    )
  else:
    map_fns.append(data_transforms.crop_templates(eval_cfg.max_templates))

  return map_fns


def process_tensors_from_config(tensors, data_config):
  """Apply filters and maps to an existing dataset, based on the config."""
  common_cfg = data_config.common
  eval_cfg = data_config.eval

  if common_cfg.reduce_msa_clusters_by_max_templates:
    pad_msa_clusters = eval_cfg.max_msa_clusters - eval_cfg.max_templates
  else:
    pad_msa_clusters = eval_cfg.max_msa_clusters

  max_msa_clusters = pad_msa_clusters
  max_extra_msa = common_cfg.max_extra_msa
  masked_msa_config = (
      common_cfg.masked_msa if 'masked_msa' in common_cfg else None)

  tensors = compose(nonensembled_map_fns(data_config))(tensors)

  num_ensemble = eval_cfg.num_ensemble
  if common_cfg.resample_msa_in_recycling:
    # Separate batch per ensembling & recycling step.
    num_ensemble *= common_cfg.num_recycle + 1

  def predraw_fn(i):
    """Sequential (par1) loop: the ensemble body's stateful RNG draws."""
    del i
    return data_transforms.ensemble_predraw(
        tensors, max_msa_clusters, masked_msa_config,
        eval_cfg.masked_msa_replace_fraction, max_extra_msa)

  if isinstance(num_ensemble, tf.Tensor) or num_ensemble > 1:
    draws_0 = predraw_fn(tf.constant(0))
    draws = tf.map_fn(
        predraw_fn,
        tf.range(num_ensemble),
        parallel_iterations=1,
        fn_output_signature=tree.map(tf.TensorSpec.from_tensor, draws_0),
    )

    def main_fn(draws_i):
      """Parallel (par4) loop: pure reconstruction from the pre-drawn values."""
      return compose(ensembled_apply_fns(data_config, draws_i))(
          tensors.copy())

    tensors_0 = main_fn(draws_0)
    # Parallelism is bounded by the crop size (== num_res, set in
    # make_data_config): the concurrent iterations' intermediate tensors scale
    # linearly with num_res (~0.22 GB each at 247), so wider parallelism is
    # only affordable for smaller proteins. Outputs are identical either way;
    # this is a memory-capacity scheduling decision, not a numerical one.
    main_parallelism = 2 if eval_cfg.crop_size <= 256 else 1
    tensors = tf.map_fn(
        main_fn,
        draws,
        parallel_iterations=main_parallelism,
        fn_output_signature=tree.map(tf.TensorSpec.from_tensor, tensors_0),
    )
  else:
    draws_0 = predraw_fn(tf.constant(0))
    tensors_0 = compose(ensembled_apply_fns(data_config, draws_0))(
        tensors.copy())
    tensors = tree.map(lambda x: x[None], tensors_0)
  return tensors


@data_transforms.curry1
def compose(x, fs):
  for f in fs:
    x = f(x)
  return x
