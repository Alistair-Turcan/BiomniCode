import inspect

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from omegaconf import OmegaConf

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
from datasets import MegaScaleDataset, ddgBenchDataset, FireProtDataset, Mutation
from protein_mpnn_utils import loss_smoothed, tied_featurize, write_inference_csv
from thermompnn_benchmarking import compute_centrality, ProteinMPNNBaseline, get_trained_model, ALPHABET

_ALPHABET_SET = frozenset(ALPHABET)  # iter023: O(1) membership (inputs are single chars by construction)


def get_ssm_mutations(pdb):
        # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list


def iter_ssm_mutations(pdb):
    """Tuple-yielding equivalent of parsing get_ssm_mutations() output (iter017).

    Yields (wtAA, seq_pos, mutAA) for each mutation option, or None for missing
    residues, in exactly the same order as get_ssm_mutations. Lets callers build
    Mutation objects directly without the ~19L string build + re-parse round-trip.
    """
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                yield (wtAA, seq_pos, mutAA)
        else:
            yield None


def retrieve_best_mutants(df_slice, allow_cys=True):
    # check for best mutant at each position
    pos_list = df_slice.position.unique()
    best_res_list = []
    for p in pos_list:
        p_slice = df_slice.loc[df_slice['position'] == p].reset_index(drop=True)
        if not allow_cys:  # filter out cysteine option
            p_slice = p_slice.loc[p_slice['mutation'] != 'C'].reset_index(drop=True)
        min_row = p_slice.iloc[pd.to_numeric(p_slice['ddG_pred']).idxmin()]
        best_res_list.append(min_row['mutation'])
    return best_res_list


class _ReplayDataset:
    """Iterate-once wrapper (iter034): caches (mut_pdb, mutations) batches on
    the first pass and replays them on later passes, so a multi-model scan
    parses each PDB once instead of once per model. Model forwards treat
    mut_pdb as read-only (verified empirically), so replay is value-identical.
    Attribute access (e.g. wt_seqs) passes through to the wrapped dataset;
    __len__ passes through too, preserving tqdm's total/no-total format. If a
    pass stops early (max_batches), the incomplete cache is discarded and the
    next pass re-iterates the underlying dataset exactly as before."""

    def __init__(self, ds):
        self._ds = ds
        self._cache = None

    def __iter__(self):
        if self._cache is None:
            cache = []
            for batch in self._ds:
                cache.append(batch)
                yield batch
            self._cache = cache
        else:
            yield from self._cache

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, k):
        return getattr(self._ds, k)


def run_ssm(models, datasets, args, max_batches=None, out_dir=''):
    """SSM batch-inference loop, extracted from main() and optimized (iter020).

    Behavior-preserving vs the original inline loop, with three already-proven
    efficiency changes applied to this secondary path:
      - iter017: iter_ssm_mutations() tuples replace the mutation-string
        round-trip when building Mutation objects
      - iter018: scalar_floats=True batched ddG extraction for TransferModel
        (ProteinMPNNBaseline lacks the kwarg and takes the tensor path)
      - iter020: per-batch row accumulation + pd.concat replaces per-cell
        df.loc[row, col] writes (O(n^2) index enlargement -> O(n))
    Also device-agnostic (cuda only when available) so CPU-only runs work.
    """
    # iter034: parse-once replay — wrap each dataset so the first model's pass
    # caches batches and later models replay them (see _ReplayDataset).
    datasets = {ds_name: _ReplayDataset(ds) for ds_name, ds in datasets.items()}
    results = {}
    for name, model in models.items():
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _sf = 'scalar_floats' in inspect.signature(type(model).forward).parameters
        for dataset_name, dataset in datasets.items():
            raw_pred_df = pd.DataFrame(columns=['WT Seq', 'Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation',
                                                        'neighbors', 'best_AA'])
            # iter044: single-frame assembly (non-pick_best path) - output
            # columns accumulate in plain lists across batches and the model
            # frame is built ONCE after the loop (below), replacing per-batch
            # pd.DataFrame construction (per-row NA validation + dtype
            # inference) and the per-model pd.concat. Column set/order,
            # dtypes, row order, and per-row values are unchanged.
            acc_wtseq = []; acc_ddg = []; acc_pos = []; acc_wt = []
            acc_mut = []; acc_nb = []; acc_pdb = []
            n_rows = 0
            print('Running model %s on dataset %s' % (name, dataset_name))
            for i, batch in enumerate(tqdm(dataset)):
                mut_pdb, mutations = batch

                # generate all SSM mutations (iter017 tuple path: same order and
                # values as get_ssm_mutations string parsing)
                final_mutation_list = []

                # build into list of Mutation objects
                for item in iter_ssm_mutations(mut_pdb[0]):
                  if item is None:
                    final_mutation_list.append(None)
                    continue
                  wtAA, position, mutAA = item

                  assert wtAA in _ALPHABET_SET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                  assert mutAA in _ALPHABET_SET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                  mutation_obj = Mutation(position, wtAA, mutAA, None, mut_pdb[0]['name'])
                  final_mutation_list.append(mutation_obj)

                if _sf:
                    pred, _ = model(mut_pdb, final_mutation_list, scalar_floats=True)
                else:
                    pred, _ = model(mut_pdb, final_mutation_list)

                # calculation of N neighbors
                if args.centrality:
                    coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
                    chain = coord_chain[-1]
                    neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain, radius=10.)

                # iter036/iter044: column-oriented assembly. The pick_best
                # path keeps the verbatim per-batch frame + cumulative concat
                # (the best-mutant logic below mutates raw_pred_df between
                # batches). The non-pick_best path accumulates columns into
                # the per-model accumulators; WT Seq and the pdb stem are
                # per-batch constants (mut.pdb == mut_pdb[0]['name'] for every
                # mutation in the batch) and are hoisted. Column set/order and
                # per-row values are unchanged; best_AA stays object-typed so
                # the pick_best string assignment behaves as before.
                if args.pick_best:
                    ddg_col = []; pos_col = []; wt_col = []; mut_col = []
                    nb_col = []; pdb_col = []
                    for mut, out in zip(final_mutation_list, pred):
                        if mut is not None:
                            ddg = out["ddG"]
                            if not isinstance(ddg, float):
                                ddg = ddg.cpu().item()
                            ddg_col.append(ddg)
                            pos_col.append(mut.position)
                            wt_col.append(mut.wildtype)
                            mut_col.append(mut.mutation)
                            nb_col.append(neighbors[mut.position].cpu().item() if args.centrality else np.nan)
                            pdb_col.append(mut.pdb.strip('.pdb'))
                    if pos_col:
                        if 'Megascale' not in dataset_name:
                            key = mut_pdb[0]['name']
                        else:
                            key = mut_pdb[0]['name'] + '.pdb'
                        batch_df = pd.DataFrame({'WT Seq': dataset.wt_seqs[key], 'Model': name,
                                                 'Dataset': dataset_name, 'ddG_pred': ddg_col,
                                                 'position': pos_col, 'wildtype': wt_col,
                                                 'mutation': mut_col, 'neighbors': nb_col,
                                                 'best_AA': np.nan, 'pdb': pdb_col})
                        batch_df['best_AA'] = batch_df['best_AA'].astype(object)
                        raw_pred_df = pd.concat([raw_pred_df, batch_df], ignore_index=True)
                else:
                    ddg_col = []; pos_col = []; wt_col = []; mut_col = []
                    nb_col = []
                    _cent = args.centrality
                    _nan = np.nan
                    for mut, out in zip(final_mutation_list, pred):
                        if mut is not None:
                            ddg = out["ddG"]
                            if not isinstance(ddg, float):
                                ddg = ddg.cpu().item()
                            ddg_col.append(ddg)
                            pos_col.append(mut.position)
                            wt_col.append(mut.wildtype)
                            mut_col.append(mut.mutation)
                            nb_col.append(neighbors[mut.position].cpu().item() if _cent else _nan)
                    if pos_col:
                        if not args.include_cys:
                            # same per-batch cys filter as the frame path
                            # (same rows removed, same order preserved)
                            keep = [k for k, m in enumerate(mut_col) if m != 'C']
                            ddg_col = [ddg_col[k] for k in keep]
                            pos_col = [pos_col[k] for k in keep]
                            wt_col = [wt_col[k] for k in keep]
                            mut_col = [mut_col[k] for k in keep]
                            nb_col = [nb_col[k] for k in keep]
                        if 'Megascale' not in dataset_name:
                            key = mut_pdb[0]['name']
                        else:
                            key = mut_pdb[0]['name'] + '.pdb'
                        _nb = len(ddg_col)
                        acc_wtseq.extend([dataset.wt_seqs[key]] * _nb)
                        acc_ddg.extend(ddg_col)
                        acc_pos.extend(pos_col)
                        acc_wt.extend(wt_col)
                        acc_mut.extend(mut_col)
                        acc_nb.extend(nb_col)
                        acc_pdb.extend([mut_pdb[0]['name'].strip('.pdb')] * _nb)
                        n_rows += _nb

                if args.pick_best:
                    # retrieve BEST mutation at each spot and save to DF
                    current_slice = raw_pred_df.loc[raw_pred_df['pdb'] == mut.pdb.strip('.pdb')]
                    best_res_list = retrieve_best_mutants(current_slice, allow_cys=args.include_cys)
                    for p, b in zip(current_slice.position.unique(), best_res_list):
                        raw_pred_df.loc[(raw_pred_df['pdb'] == mut.pdb.strip('.pdb')) & (raw_pred_df['position'] == p), 'best_AA'] = b

                    # only keep one row per position for AA pattern detection
                    raw_pred_df['dupe_detector'] = raw_pred_df['pdb'] + raw_pred_df['position'].astype(str)
                    raw_pred_df = raw_pred_df.drop_duplicates(subset=['dupe_detector'], keep='first')

                else:
                    # iter036: non-pick_best cys filtering is applied per batch
                    # frame above (same rows, same order as filtering the
                    # cumulative frame once per batch).
                    pass

                if max_batches is not None and i >= max_batches:
                    break

                print('Completed protein:', mut.pdb)
                print('Mutations processed:', raw_pred_df.shape if args.pick_best else (n_rows, 10))

            if not args.pick_best and n_rows:
                # iter044: build the model frame once. dtypes match the
                # per-batch-frame concat exactly: ddG_pred float64 (Python
                # floats), position int64 (Python ints), neighbors float64,
                # WT Seq/wildtype/mutation/pdb object (str), Model/Dataset
                # object (scalar str broadcast), best_AA object NaN. With no
                # rows the empty seed frame is kept (original no-batch path).
                _best = np.empty(n_rows, dtype=object)
                _best[:] = np.nan
                raw_pred_df = pd.DataFrame({'WT Seq': acc_wtseq, 'Model': name,
                                            'Dataset': dataset_name,
                                            'ddG_pred': acc_ddg,
                                            'position': acc_pos,
                                            'wildtype': acc_wt,
                                            'mutation': acc_mut,
                                            'neighbors': acc_nb,
                                            'best_AA': _best, 'pdb': acc_pdb})
            raw_pred_df = raw_pred_df.reset_index(drop=True)
            csv_name = name + '_' + dataset_name + "_SSM_preds.csv"
            write_inference_csv(raw_pred_df, os.path.join(out_dir, csv_name) if out_dir else csv_name)
            results[(name, dataset_name)] = raw_pred_df
            del raw_pred_df
    return results
def main(cfg, args):
    """Inference script that does site-saturation mutagenesis for a given protein"""
    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)
    misc_data_loc = '/nas/longleaf/home/dieckhau/protein-stability/enzyme-stability/data'

    models = {
        'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        "ThermoMPNN": get_trained_model(model_name='example_training_mega_epoch=55_val_ddG_spearman=0.79.ckpt',
                                        config=cfg)

        # "Model-CV0": TransferModelPL.load_from_checkpoint('checkpoints/CV0_epoch=32_val_ddG_spearman=0.73.ckpt', cfg=cfg).model,
        # "Model-CV1": TransferModelPL.load_from_checkpoint('checkpoints/CV1_epoch=27_val_ddG_spearman=0.76.ckpt', cfg=cfg).model,
        # "Model-CV2": TransferModelPL.load_from_checkpoint('checkpoints/CV2_epoch=23_val_ddG_spearman=0.75.ckpt', cfg=cfg).model,
        # "Model-CV3": TransferModelPL.load_from_checkpoint('checkpoints/CV3_epoch=24_val_ddG_spearman=0.74.ckpt', cfg=cfg).model,
        # "Model-CV4": TransferModelPL.load_from_checkpoint('checkpoints/CV4_epoch=31_val_ddG_spearman=0.76.ckpt', cfg=cfg).model,
    }

    datasets = {
        # "Megascale-test": MegaScaleDataset(cfg, "test"),
        # "Megascale-CV0": MegaScaleDataset(cfg, "cv_test_0"),
        # "Megascale-CV1": MegaScaleDataset(cfg, "cv_test_1"),
        # "Megascale-CV2": MegaScaleDataset(cfg, "cv_test_2"),
        # "Megascale-CV3": MegaScaleDataset(cfg, "cv_test_3"),
        # "Megascale-CV4": MegaScaleDataset(cfg, "cv_test_4"),

        # "Fireprot-test": FireProtDataset(cfg, "test")
        # "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free"),
        "P53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv')),
    }

    max_batches = None

    run_ssm(models, datasets, args, max_batches=max_batches)


if __name__ == "__main__":
    try:
        cfg = OmegaConf.load("local.yaml")
    except FileNotFoundError:
        cfg = OmegaConf.load("../local.yaml")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pick_best', action='store_true', default=False,
                        help='Keep only the BEST mutation at each position')
    parser.add_argument('--include_cys', action='store_true', default=False,
                        help='Include cysteine as potential mutation option.'
                             'Due to assay artifacts, mutations to cys are predicted poorly.')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')
    args = parser.parse_args()

    with torch.no_grad():
        main(cfg, args)
