import io
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
from datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset
from transfer_model import get_protein_mpnn, TransferModel
from protein_mpnn_utils import tied_featurize


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:

    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1
    # Compute centralities
    # centralities = {
    #     'all': torch.ones(num_neighbors.shape, device=num_neighbors.device),
    #     'core': num_neighbors >= core_threshold,
    #     # 'boundary': num_neighbors < core_threshold & num_neighbors > surface_threshold,
    #     'surface': num_neighbors <= surface_threshold,
    # }
    return num_neighbors


class ProteinMPNNBaseline(nn.Module):
    """Class for running ProteinMPNN as a ddG proxy predictor"""

    def __init__(self, cfg, version='v_48_020.pt'):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn(cfg, version=version)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                [pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        *_, log_probs = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        out = [None] * len(mutations)
        valid = [(i, m) for i, m in enumerate(mutations) if m is not None]
        if valid:
            aa_idx = {a: k for k, a in enumerate(ALPHABET)}
            pos_t = torch.tensor([m.position for _, m in valid],
                                 dtype=torch.long, device=log_probs.device)
            aa_t = torch.tensor([aa_idx[m.mutation] for _, m in valid],
                                dtype=torch.long, device=log_probs.device)
            preds = log_probs[0, pos_t, aa_t]
            ddg_parts = (-preds).split(1)
            dtm_parts = preds.split(1)
            for k, (i, m) in enumerate(valid):
                out[i] = {"ddG": ddg_parts[k], "dTm": dtm_parts[k]}
        return out, log_probs


def get_metrics():
    # lazy import: torchmetrics is only needed for benchmark metric tracking,
    # not for inference, so keep it out of the module import chain
    from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
    return {
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson":  PearsonCorrCoef().to('cuda'),
    }


def get_trained_model(model_name, config, checkpt_dir='models/', override_custom=False):
    if override_custom:
        model_loc = model_name
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
    return _load_transfer_model_direct(model_loc, config)


def _load_transfer_model_direct(ckpt_path, config):
    """Load a TransferModel from a Lightning checkpoint without importing
    pytorch_lightning/wandb (which dominate module import time).

    Exactly equivalent to
    ``TransferModelPL.load_from_checkpoint(ckpt_path, cfg=config).model``:
    TransferModelPL.__init__ builds ``TransferModel(cfg)`` plus inference-
    irrelevant torchmetrics, and the checkpoint state_dict stores those
    weights under the ``model.`` prefix (``metrics.*`` entries, if present,
    carry no model weights and are dropped).
    """
    # Buffer the checkpoint through memory: torch.load issues one small read
    # per tensor record, which is latency-bound on high-latency filesystems
    # (network/FUSE mounts); a single streaming read is much faster there and
    # equivalent on local disk. The parsed state_dict is byte-identical.
    with open(ckpt_path, 'rb') as fh:
        ckpt = torch.load(io.BytesIO(fh.read()), map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith('metrics.'):
            continue
        stripped[k[len('model.'):] if k.startswith('model.') else k] = v
    model = TransferModel(config)
    model.load_state_dict(stripped, strict=True)
    return model


def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics(),
    }
    print('Testing Model %s on dataset %s' % (name, dataset_name))

    for i, batch in enumerate(tqdm(dataset)):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    raw_pred_df = pd.DataFrame(
        columns=['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
                 'neighbors', 'best_AA'])
    metrics = {
        "ddG": get_metrics(),
    }
    print('Running model %s on dataset %s' % (name, dataset_name))
    for i, batch in enumerate(tqdm(dataset)):
        mut_pdb, mutations = batch
        pred, _ = model(mut_pdb, mutations)

        if centrality:
            coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain,
                                           radius=10.)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

                # assign raw preds and useful details to df
                col_list = ['ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                val_list = [mut.ddG.cpu().item(), out["ddG"].cpu().item(), mut.position, mut.wildtype,
                            mut.mutation, mut.pdb.strip('.pdb')]
                for col, val in zip(col_list, val_list):
                    raw_pred_df.loc[row, col] = val

                if centrality:
                    raw_pred_df.loc[row, 'neighbors'] = neighbors[mut.position].cpu().item()

            raw_pred_df.loc[row, 'Model'] = name
            raw_pred_df.loc[row, 'Dataset'] = dataset_name
            if 'Megascale' not in dataset_name: # different pdb column formatting
                key = mut.pdb
            else:
                key = mut.pdb + '.pdb'
            if 'S669' not in dataset_name: # S669 is missing WT seq info - omit to prevent error
                raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:  # , "dTm"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                pass
    results.append(column)
    raw_pred_df.to_csv(name + '_' + dataset_name + "_raw_preds.csv")
    del raw_pred_df

    return results


def main(cfg, args):

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

    models = {
        'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        "ThermoMPNN": get_trained_model(model_name='thermoMPNN_default.pt',
                                        config=cfg)

    }

    misc_data_loc = '/nas/longleaf/home/dieckhau/protein-stability/enzyme-stability/data'
    datasets = {
        # "Megascale-test": MegaScaleDataset(cfg, "test"),
        # "Fireprot-test": FireProtDataset(cfg, "test"),

        # "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free"),
        "P53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv')),
        "MYOGLOBIN": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')),

        "SSYM_dir": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')),
        "SSYM_inv": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')),
        "S669": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'S669/pdbs'),
                               csv_fname=os.path.join(misc_data_loc, 'S669/s669_clean_dir.csv')),
    }

    results = []

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                results = run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=args.centrality)
            else:
                results = run_prediction_default(name, model, dataset_name, dataset, results)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("ThermoMPNN_metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_preds', action='store_true', default=False, help='Save raw model predictions as csv')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')

    args = parser.parse_args()
    try:
        cfg = OmegaConf.load("local.yaml")
    except FileNotFoundError:
        cfg = OmegaConf.load("../local.yaml")
    with torch.no_grad():
        main(cfg, args)
