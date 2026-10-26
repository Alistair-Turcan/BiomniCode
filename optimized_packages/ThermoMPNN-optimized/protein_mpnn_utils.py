from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import re


# A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design

def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def _S_to_seq(S, mask):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x);
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x);
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                #         resn = int(resn)
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms: xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'


def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False):
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            elif side_chains:
                sidechain_atoms = ["N", "CA", "C", "O", "CB",
                                   "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                                   "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                                   "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                                   "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2"]
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list

def alt_parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x);
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x);
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    resn_list = []
    # iter015: O(1) residue-id membership via resn_seen (the original
    # `resn not in resn_list` linear scan was O(n_residues) per atom line and
    # dominated the parse), one buffered read, bytes-native fixed-width slicing,
    # and deferred array construction (plain tuples instead of one np.array per
    # atom). Parsed values, first-occurrence-wins guards, resn_list contents,
    # and all return values are unchanged (ASCII PDB lines slice identically as
    # bytes; the two small decoded fields are decoded exactly as before).
    resn_seen = set()
    resn_cache = {}
    atom_cache = {}
    last_key = None
    cur_xyz = None
    chain_b = chain.encode("utf-8") if chain is not None else None
    with open(x, "rb") as _f:
        _blob_lines = _f.read().splitlines()
    for line in _blob_lines:
        # iter039: ATOM-first dispatch - the HETATM->ATOM MSE rewrite moves to
        # the elif so ATOM lines (the overwhelming majority) skip the 6-byte
        # slice+compare. Line routing is identical to the original.
        if line[:4] == b"ATOM":
            pass
        elif line[:6] == b"HETATM" and line[17:17 + 3] == b"MSE":
            line = line.replace(b"HETATM", b"ATOM  ")
            line = line.replace(b"MSE", b"MET")
        else:
            continue

        if chain_b is None or line[21:22] == chain_b:
            # iter019: residue-change hoisting + deferred decodes + tuple
            # coord storage. PDB ATOM lines are grouped by residue, so work
            # that depends only on residue identity (resn/int parsing, dict
            # guards, min/max, seq assignment, field decodes) runs once per
            # residue change instead of once per atom line; revisited
            # residues re-run the idempotent guard block, keeping
            # non-contiguous orderings correct. atom dict keys stay bytes
            # (internal only; the final loop iterates atoms_b). All parsed
            # values, first-occurrence-wins guards, resn_list contents, and
            # return values are unchanged.
            # iter039: raw-slice strip caches for atom and resn (fixed-width
            # fields repeat per residue, so the raw 4/5-byte slice keys the
            # cache and strip/decode runs once per distinct value), and the
            # per-residue atom dict is hoisted into cur_xyz on residue change
            # (dict objects are only ever mutated in place by the guards, so
            # the cached reference stays valid; revisited residues re-enter
            # the guard block and refresh it). All parsed values,
            # first-occurrence-wins guards, resn_list contents, and return
            # values are unchanged.
            raw_atom = line[12:12 + 4]
            atom = atom_cache.get(raw_atom)
            if atom is None:
                atom = raw_atom.strip()
                atom_cache[raw_atom] = atom
            raw_resn = line[22:22 + 5]
            cached = resn_cache.get(raw_resn)
            if cached is None:
                resn = raw_resn.strip()
                # RAW resn is defined HERE
                if resn not in resn_seen:
                  resn_seen.add(resn)
                  resn_list.append(resn.decode("utf-8", "ignore")) # NEED to keep ins code here
                if resn[-1:].isalpha():
                    cached = (resn[-1:].decode("utf-8", "ignore"), int(resn[:-1]) - 1)
                else:
                    cached = ("", int(resn) - 1)
                resn_cache[raw_resn] = cached
            resa, resn_i = cached
            if (resn_i, resa) != last_key:
                last_key = (resn_i, resa)
                if resn_i < min_resn:
                    min_resn = resn_i
                if resn_i > max_resn:
                    max_resn = resn_i
                if resn_i not in xyz:
                    xyz[resn_i] = {}
                if resa not in xyz[resn_i]:
                    xyz[resn_i][resa] = {}
                if resn_i not in seq:
                    seq[resn_i] = {}
                if resa not in seq[resn_i]:
                    seq[resn_i][resa] = line[17:17 + 3].decode("utf-8", "ignore")
                cur_xyz = xyz[resn_i][resa]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if atom not in cur_xyz:
                cur_xyz[atom] = (x, y, z)
            # print(resn_list)
    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    atoms_b = [a.encode("utf-8") for a in atoms]  # iter019: atom keys are bytes
    _nan3 = np.full(3, np.nan)  # iter039: shared read-only filler (np.array
    # below copies values out, so sharing one object is safe)
    # print(seq)
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
              # CLEANED resn is defined HERE
                # resn_list.append(str(resn + 1))
                # print(resn)
                # iter039: single-key fast path - iterating a 1-key dict yields
                # that key, identical to sorted() order; multi-key (insertion
                # codes) still sorts.
                seq_resn = seq[resn]
                for k in (seq_resn if len(seq_resn) == 1 else sorted(seq_resn)):
                    seq_.append(aa_3_N.get(seq_resn[k], 20))
            else:
                seq_.append(20)
                # if str(resn - 1) not in resn_list:
                  # print(resn, '**')
                # resn_list.append('DASH')
            # print(resn, resn_list[-1])
            if resn in xyz:
                xyz_resn = xyz[resn]
                for k in (xyz_resn if len(xyz_resn) == 1 else sorted(xyz_resn)):
                    xyz_resn_k = xyz_resn[k]
                    for atom in atoms_b:
                        if atom in xyz_resn_k:
                            xyz_.append(xyz_resn_k[atom])
                        else:
                            xyz_.append(_nan3)
            else:
                for atom in atoms_b: xyz_.append(_nan3)
        # assert len(seq_) == len(list(dict.fromkeys(resn_list)))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)), list(dict.fromkeys(resn_list))
    except TypeError:
        return 'no_chain', 'no_chain', 'no_chain'


def alt_parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False):
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        my_dict['resn_list'] = []
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            elif side_chains:
                sidechain_atoms = ["N", "CA", "C", "O", "CB",
                                   "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                                   "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                                   "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                                   "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2"]
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq, resn_list = alt_parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)

            # add chain resn set to list if exists
            # new for multi-chain handling
            # if len(resn_list) > 7:
            #     check_chain = ''.join(resn_list[:8])
            #     if check_chain != 'no_chain':
            #         my_dict['resn_list'] += resn_list

            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq

        # old handling for resn lists
        my_dict['resn_list'] = list(resn_list)
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def tied_featurize(batch, device, chain_dict, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None,
                   pssm_dict=None, bias_by_res_dict=None, ca_only=False):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones([B, L_max, 21],
                                          dtype=np.float32)  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b['name']]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains
        # random.shuffle(all_chains)
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([chain_coords[c] for c in
                                        [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                         f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([chain_coords[c] for c in
                                        [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                         f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet)) == AA)[0][0] for AA in item[1]]).repeat(
                            idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx + v[0][v_count] - 1)  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ - 1)  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(fixed_position_mask_list, 0)  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list, 0)  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list,
                                      0)  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list, 0), [[0, L_max - l]], 'constant',
                                  constant_values=(0.0,))
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate([phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1)  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X
    return X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Convert raw coords to np arrays
                # for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))

            print('discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

            # print('Discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx, out=None):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    if not torch.is_grad_enabled():
        # iter052: flat index_select performs the identical element selection
        # (pure data movement, bitwise-identical outputs) with contiguous
        # row-wise copies instead of per-element reads through a stride-0
        # expanded index (~7x faster at inference shapes). Grad-enabled
        # callers keep the original gather path: index_select's backward
        # (index_add) can accumulate duplicate-index gradients in a different
        # order than gather's scatter, so training semantics are untouched.
        B, N, C = nodes.shape
        K = neighbor_idx.shape[2]
        if B == 1:
            flat = neighbor_idx.reshape(-1)
        else:
            flat = (neighbor_idx
                    + torch.arange(B, device=nodes.device).view(B, 1, 1) * N
                    ).reshape(-1)
        if out is not None:
            # iter054: write the gather directly into a caller-supplied
            # [B,N,K,C] slice of a contiguous buffer (e.g. h_EV[..., 2C:]),
            # eliminating the fresh output tensor and the subsequent copy_.
            # Same index_select kernel writing the same rows =>
            # bitwise-identical (probe_iter54_fusion.py). Falls back to
            # fresh+copy_ if the slice layout lacks the expected uniform row
            # stride. Inference-only: grad callers must pass out=None.
            row = out.stride(2)
            if (out.stride(3) == 1 and out.stride(1) == K * row
                    and out.stride(0) == N * K * row):
                torch.index_select(nodes.reshape(B * N, C), 0, flat,
                                   out=out.as_strided((B * N * K, C),
                                                      (row, 1)))
                return out
            out.copy_(torch.index_select(nodes.reshape(B * N, C), 0, flat
                                         ).view(B, N, K, C))
            return out
        return torch.index_select(nodes.reshape(B * N, C), 0, flat).view(B, N, K, C)
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def _message_input(self, h_V, h_E, E_idx, msg_buf=None):
        """Build the [B, L, K, 3C] message input in one preallocated buffer.

        Pure data movement: the buffer holds byte-identical values in the same
        contiguous layout as the original
        ``cat([h_V_expand, cat([h_E, gather(h_V)])])``, so the downstream GEMM
        is bitwise identical while avoiding one intermediate concatenation.

        iter032: when msg_buf is supplied (forward-scoped, grad-disabled
        inference only) it is reused as the output buffer. Every slice is
        fully overwritten below before the GEMM reads it, so values are
        bitwise-identical; reuse only avoids repeated first-touch page-fault
        passes over fresh ~60MB allocations.
        """
        C = self.num_hidden
        shape = (h_V.shape[0], h_V.shape[1], E_idx.shape[2], 3 * C)
        if (msg_buf is not None and msg_buf.shape == shape
                and msg_buf.dtype == h_V.dtype and msg_buf.device == h_V.device):
            h_EV = msg_buf
        else:
            h_EV = torch.empty(shape, device=h_V.device, dtype=h_V.dtype)
        h_EV[..., :C] = h_V.unsqueeze(-2)
        h_EV[..., C:2 * C] = h_E
        if not torch.is_grad_enabled():
            # iter054: gather straight into the buffer slice (same rows,
            # bitwise-identical), removing the fresh [B,L,K,C] gather output
            # and its copy_ (probe_iter54_fusion.py).
            gather_nodes(h_V, E_idx, out=h_EV[..., 2 * C:])
        else:
            h_EV[..., 2 * C:] = gather_nodes(h_V, E_idx)
        return h_EV

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, msg_buf=None,
                mask_scratch=None):
        """ Parallel computation of full transformer layer """

        h_EV = self._message_input(h_V, h_E, E_idx, msg_buf)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            if (mask_scratch is not None
                    and mask_scratch.shape == h_message.shape
                    and mask_scratch.dtype == h_message.dtype
                    and mask_scratch.device == h_message.device):
                # iter055: write the mask multiply into the forward-scoped
                # scratch (supplied only on the grad-disabled inference path)
                # instead of allocating a fresh [B,L,K,C] per layer.
                # torch.mul(out=) is elementwise-identical; the scratch is
                # consumed by torch.sum immediately below and fully
                # overwritten before every read (probe_iter55.py).
                torch.mul(mask_attend.unsqueeze(-1), h_message, out=mask_scratch)
                h_message = mask_scratch
            else:
                h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = self._message_input(h_V, h_E, E_idx, msg_buf)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        return self._forward_fused(h_V, h_EV, mask_V, mask_attend)

    def _forward_fused(self, h_V, h_EV, mask_V=None, mask_attend=None):
        """Forward with a pre-assembled [B, L, K, C+num_in] message input.

        Numerically identical to forward(); lets callers fuse the input
        construction with upstream elementwise work (pure data movement).
        """

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + (1 - mask) * (
                    2 * self.max_relative_feature + 1)
        if not torch.is_grad_enabled():
            # iter056: linear(one_hot(d)) == weight.t()[d] + bias, bitwise
            # (probe_iter55.py candidate B: max_abs_diff=0.0). A one-hot row
            # e_j makes the GEMM reduce to exactly row j of W^T (the zero
            # terms add 0.0 exactly), and the bias add is IEEE-commutative,
            # so the embedding row gather is value-identical while skipping
            # the [B,L,K,66] one_hot materialization and its GEMM.
            # Inference-only: under grad the gather backward scatter-adds
            # duplicate-index gradients in a different order than the GEMM
            # backward, so grad callers keep the verbatim one_hot path.
            E = self.linear.weight.t()[d] + self.linear.bias
        else:
            d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
            E = self.linear(d_onehot.float())
        return E


class CA_ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(CA_ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)  # NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)  # exclude CA-CA jumps
        dX = dX * dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        V, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        # Gather-first (iter028): compute distances only on the gathered
        # [B, L, K, 3] neighborhoods instead of a full [B, L, L] matrix whose
        # gathered output keeps only ~K/L of entries. Every op is elementwise
        # or a 3-element last-dim reduction on identical float32 pairs, so the
        # result is bitwise-identical to full-matrix-then-gather (verified
        # torch.equal on all 24 call-site pairs and on features.forward E).
        B_neighbors = gather_nodes(B, E_idx)  # [B, L, K, 3]
        D_A_B_neighbors = torch.sqrt(torch.sum((A.unsqueeze(2) - B_neighbors) ** 2, -1) + 1e-6)  # [B, L, K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        if torch.is_grad_enabled():
            # Verbatim reference path for grad-enabled callers (training).
            # The fused inference path below writes E_positional into a
            # non-requires-grad buffer via copy_, which would detach
            # self.embeddings.linear from the autograd graph; keeping the
            # original op sequence here preserves the training contract
            # (iter052 gather_nodes precedent).
            RBF_all = []
            RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
            RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
            RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
            RBF_all = torch.cat(tuple(RBF_all), dim=-1)

            offset = residue_idx[:, :, None] - residue_idx[:, None, :]
            offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

            d_chains = ((chain_labels[:, :, None] - chain_labels[:, None,
                                                    :]) == 0).long()  # find self vs non-self interaction
            E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
            E_positional = self.embeddings(offset.long(), E_chains)
            E = torch.cat((E_positional, RBF_all), -1)
            E = self.edge_embedding(E)
        else:
            # iter053 (re-gated iter040/041 bundle, inference-only): assemble
            # the 25 RBF blocks directly into one preallocated
            # [B, L, K, num_positional_embeddings + 25*num_rbf] edge-feature
            # buffer that also absorbs the final cat with E_positional.
            #  - the 24 _get_rbf calls gathered only 5 distinct neighbor sets;
            #    gather_nodes is deterministic, so hoisting the duplicates is
            #    bitwise-identical (19 redundant [B,L,K,3] gathers removed)
            #  - D_mu/D_sigma are constants of the layer; computed once per
            #    forward instead of 25x (identical linspace values)
            #  - each pair distance uses the identical elementwise op sequence
            #    to _get_rbf (sub -> pow2 -> sum(-1) -> +1e-6 -> sqrt) and each
            #    RBF block the identical sequence to _rbf (sub -> div -> pow2
            #    -> neg -> exp), computed in scratch and written via
            #    torch.exp(..., out=slice) straight into the buffer, so values
            #    are bitwise-identical in the same contiguous layout
            #    (exp-out-into-strided-slice verified torch.equal-exact)
            n_rbf = self.num_rbf
            n_pos = self.num_positional_embeddings
            E_buf = torch.empty((E_idx.shape[0], E_idx.shape[1], E_idx.shape[2],
                                 n_pos + 25 * n_rbf),
                                device=X.device, dtype=D_neighbors.dtype)
            D_mu = torch.linspace(2., 22., n_rbf, device=X.device).view([1, 1, 1, -1])
            D_sigma = (22. - 2.) / n_rbf

            N_nb = gather_nodes(N, E_idx)
            C_nb = gather_nodes(C, E_idx)
            O_nb = gather_nodes(O, E_idx)
            Cb_nb = gather_nodes(Cb, E_idx)
            Ca_nb = gather_nodes(Ca, E_idx)

            # forward-scoped scratch buffers reused across the 24 pair
            # evaluations (every element fully overwritten before each read,
            # so values are bitwise-identical; only allocation churn is
            # removed)
            s3 = torch.empty((E_idx.shape[0], E_idx.shape[1], E_idx.shape[2], 3),
                             device=X.device, dtype=D_neighbors.dtype)
            s1 = torch.empty((E_idx.shape[0], E_idx.shape[1], E_idx.shape[2]),
                             device=X.device, dtype=D_neighbors.dtype)
            s4 = torch.empty((E_idx.shape[0], E_idx.shape[1], E_idx.shape[2], n_rbf),
                             device=X.device, dtype=D_neighbors.dtype)

            def _pair_d(A, B_nb):
                # identical op sequence to _get_rbf's distance computation
                # (sub -> pow2 -> sum(-1) -> +1e-6 -> sqrt), out= into scratch
                torch.sub(A.unsqueeze(2), B_nb, out=s3)
                torch.pow(s3, 2, out=s3)
                torch.sum(s3, -1, out=s1)
                torch.add(s1, 1e-6, out=s1)
                torch.sqrt(s1, out=s1)
                return s1

            def _rbf_into(D, i):
                # identical op sequence to _rbf (sub -> div -> pow2 -> neg ->
                # exp), out= into scratch, then into block i of E_buf
                torch.sub(torch.unsqueeze(D, -1), D_mu, out=s4)
                torch.div(s4, D_sigma, out=s4)
                torch.pow(s4, 2, out=s4)
                torch.neg(s4, out=s4)
                torch.exp(s4, out=E_buf[..., n_pos + i * n_rbf: n_pos + (i + 1) * n_rbf])

            _rbf_into(D_neighbors, 0)          # Ca-Ca
            _rbf_into(_pair_d(N, N_nb), 1)     # N-N
            _rbf_into(_pair_d(C, C_nb), 2)     # C-C
            _rbf_into(_pair_d(O, O_nb), 3)     # O-O
            _rbf_into(_pair_d(Cb, Cb_nb), 4)   # Cb-Cb
            _rbf_into(_pair_d(Ca, N_nb), 5)    # Ca-N
            _rbf_into(_pair_d(Ca, C_nb), 6)    # Ca-C
            _rbf_into(_pair_d(Ca, O_nb), 7)    # Ca-O
            _rbf_into(_pair_d(Ca, Cb_nb), 8)   # Ca-Cb
            _rbf_into(_pair_d(N, C_nb), 9)     # N-C
            _rbf_into(_pair_d(N, O_nb), 10)    # N-O
            _rbf_into(_pair_d(N, Cb_nb), 11)   # N-Cb
            _rbf_into(_pair_d(Cb, C_nb), 12)   # Cb-C
            _rbf_into(_pair_d(Cb, O_nb), 13)   # Cb-O
            _rbf_into(_pair_d(O, C_nb), 14)    # O-C
            _rbf_into(_pair_d(N, Ca_nb), 15)   # N-Ca
            _rbf_into(_pair_d(C, Ca_nb), 16)   # C-Ca
            _rbf_into(_pair_d(O, Ca_nb), 17)   # O-Ca
            _rbf_into(_pair_d(Cb, Ca_nb), 18)  # Cb-Ca
            _rbf_into(_pair_d(C, N_nb), 19)    # C-N
            _rbf_into(_pair_d(O, N_nb), 20)    # O-N
            _rbf_into(_pair_d(Cb, N_nb), 21)   # Cb-N
            _rbf_into(_pair_d(C, Cb_nb), 22)   # C-Cb
            _rbf_into(_pair_d(O, Cb_nb), 23)   # O-Cb
            _rbf_into(_pair_d(C, O_nb), 24)    # C-O

            # iter037 scraps, re-bundled: gather the per-residue integers at
            # E_idx first and subtract/compare on [B,L,K] instead of building
            # full [B,L,L] int64 temporaries and gathering from them. Integer
            # arithmetic is exact, so values are bitwise-identical (same
            # gather semantics as gather_edges on the expanded [B,L,L,1]
            # tensors).
            ri_nb = gather_nodes(residue_idx.unsqueeze(-1), E_idx).squeeze(-1)
            offset = residue_idx.unsqueeze(-1) - ri_nb  # [B, L, K]
            cl_nb = gather_nodes(chain_labels.unsqueeze(-1), E_idx).squeeze(-1)
            E_chains = (chain_labels.unsqueeze(-1) == cl_nb).long()
            E_positional = self.embeddings(offset.long(), E_chains)
            E_buf[..., :n_pos] = E_positional
            E = self.edge_embedding(E_buf)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
                 hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False,
                decoding_order=None):
        """ Graph-conditioned sequence model """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        # iter032: one forward-scoped message buffer shared by the 6 encoder
        # _message_input calls (inference only). Each call fully overwrites
        # every slice before its GEMM reads the buffer, so values are
        # bitwise-identical; reuse avoids 5 redundant first-touch page-fault
        # passes over ~60MB allocations. The buffer is freed when forward
        # returns, so peak RSS is unchanged. Grad-enabled callers (training)
        # get the original per-call allocations, preserving autograd safety.
        enc_msg_buf = None
        enc_mask_scratch = None
        if not torch.is_grad_enabled():
            enc_msg_buf = torch.empty((h_V.shape[0], h_V.shape[1], E_idx.shape[2],
                                       3 * h_V.shape[-1]),
                                      device=h_V.device, dtype=h_V.dtype)
            # iter055: forward-scoped scratch for the encoder mask multiply.
            # One [B,L,K,C] buffer reused by all 6 EncLayer calls instead of 6
            # fresh allocations; each layer's torch.mul(out=) fully overwrites
            # it and the only read (torch.sum) happens immediately after, so
            # values are bitwise-identical (probe_iter55.py).
            enc_mask_scratch = torch.empty((h_V.shape[0], h_V.shape[1], E_idx.shape[2],
                                            h_V.shape[-1]),
                                           device=h_V.device, dtype=h_V.dtype)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, msg_buf=enc_msg_buf,
                             mask_scratch=enc_mask_scratch)
        del enc_msg_buf, enc_mask_scratch  # iter032/iter055: release before the decoder's
                                           # h_DEC allocation so peak RSS matches the
                                           # per-call-allocation original

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)

        # Dead-code elimination (iter013): order_mask_backward was unconditionally
        # overwritten by torch.ones_like(...) below, so mask_attend == 1,
        # mask_bw == mask_1D, mask_fw == 0, and h_EXV_encoder_fw == 0. The
        # h_EX_encoder/h_EXV_encoder gather passes, the decoding_order/
        # one_hot/triu/einsum block, and the +0.0 add terms are removed.
        # Bitwise-neutral: x*1.0 == x and x+0.0 == x exactly in IEEE-754; a sign
        # flip of exact -0.0 elements cannot propagate through the downstream
        # biased GEMM reductions.

        chain_M = chain_M * mask  # update chain_M to include missing regions

        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D

        all_hidden = []
        # Fused decoder message input: one reusable [B, L, K, C+384] buffer
        # combining cat_neighbors_nodes, the mask multiply-add, and DecLayer's
        # internal cat into a single preallocated tensor. Every element holds
        # byte-identical values in the same layout as the original chain, so
        # the layer GEMMs are bitwise identical; intermediate cat buffers are
        # eliminated. The h_V-independent slice is loop-invariant (the fused
        # layer never mutates the buffer), so it is written once.
        C = h_V.shape[-1]
        S_ES = h_E.shape[-1] + h_S.shape[-1]
        h_DEC = torch.empty((h_V.shape[0], h_V.shape[1], E_idx.shape[2], C + S_ES + C),
                            device=h_V.device, dtype=h_V.dtype)
        if torch.is_grad_enabled():
            # Verbatim reference path for grad-enabled callers (training):
            # materialize h_ES and use copy_ semantics exactly as the
            # original. The iter054 mul-out fusions below are inference-only
            # (iter052/iter053 guard precedent).
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
            h_DEC[..., C:C + S_ES] = mask_bw * h_ES
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_DEC[..., :C] = h_V.unsqueeze(-2)
                h_DEC[..., C + S_ES:] = mask_bw * gather_nodes(h_V, E_idx)
                h_V = layer._forward_fused(h_V, h_DEC, mask)
                all_hidden.append(h_V)
        else:
            # iter054: never materialize h_ES or the mask-multiply
            # temporaries. h_ES == cat([h_E, gather_nodes(h_S)]), and
            # mask_bw * cat([a, b]) == cat([mask_bw * a, mask_bw * b])
            # elementwise-exactly, so two torch.mul(..., out=slice) calls
            # write byte-identical values straight into h_DEC and skip the
            # [B,L,K,256] cat, the [B,L,K,256] mul temporary, and the copy_
            # (probe_iter54_fusion.py: torch.equal-exact, ~10ms/forward at
            # 1BFD scale). The decoder-loop gather term likewise writes its
            # mask multiply directly into the buffer slice.
            torch.mul(mask_bw, h_E, out=h_DEC[..., C:C + h_E.shape[-1]])
            torch.mul(mask_bw, gather_nodes(h_S, E_idx),
                      out=h_DEC[..., C + h_E.shape[-1]:C + S_ES])
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_DEC[..., :C] = h_V.unsqueeze(-2)
                torch.mul(mask_bw, gather_nodes(h_V, E_idx),
                          out=h_DEC[..., C + S_ES:])
                h_V = layer._forward_fused(h_V, h_DEC, mask)
                all_hidden.append(h_V)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_S, log_probs

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0,
               omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
               pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None,
               bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask  # update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(
            randn)))  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        # chain_mask_combined = chain_mask*chain_M_pos
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]
            mask_gathered = torch.gather(mask, 1, t[:, None])  # [B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]  # [B, 21]
            if (mask_gathered == 0).all():  # for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1,
                                               t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                                                             h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2],
                                                                                     mask_bw.shape[
                                                                                         -1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                                              layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits - constant[None, :] * 1e8 + constant_bias[None,
                                                                     :] / temperature + bias_by_res_gathered / temperature,
                                  dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[
                                         :, 0]
                    probs = (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs + pssm_multi * pssm_coef_gathered[:,
                                                                                                  None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:, None, None].repeat(1, 1,
                                                                                                              pssm_log_odds_mask.shape[
                                                                                                                  -1]))[
                                                  :, 0]  # [B, 21]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1,
                                                         t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]))[:,
                                            0]  # [B, 21]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21),
                                   (chain_mask_gathered[:, :, None, ] * probs[:, None, :]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def tied_sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0,
                    omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None,
                    pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None,
                    pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask  # update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(
            randn)))  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(
            X.shape[0], 1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:, t:t + 1, :]
                    h_E_t = h_E[:, t:t + 1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t:t + 1, :, :]
                    mask_t = mask[:, t:t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:, t:t + 1, :]
                        h_ESV_t = mask_bw[:, t:t + 1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l + 1][:, t, :] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:, t, :]
                    logit_list.append((self.W_out(h_V_t) / temperature) / len(t_list))
                    logits += tied_beta[t] * (self.W_out(h_V_t) / temperature) / len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:, t, :]  # [B, 21]
                probs = F.softmax(logits - constant[None, :] * 1e8 + constant_bias[None,
                                                                     :] / temperature + bias_by_res_gathered / temperature,
                                  dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, t]
                    pssm_bias_gathered = pssm_bias[:, t]
                    probs = (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs + pssm_multi * pssm_coef_gathered[:,
                                                                                                  None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:, t]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:, t] * S_t_repeat + (1 - chain_mask[:, t]) * S_true[:,
                                                                                       t]).long()  # hard pick fixed positions
                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t_repeat)
                    S[:, t] = S_t_repeat
                    all_probs[:, t, :] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,] + 0.0001) * (torch.abs(
                randn)))  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                               (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                               permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


# ---------------------------------------------------------------------------
# iter022: fast CSV writer for inference outputs
# ---------------------------------------------------------------------------

def _needs_csv_quote(v):
    return (',' in v) or ('"' in v) or ('\n' in v) or ('\r' in v)


def write_inference_csv(df, path):
    """Byte-identical replacement for ``df.to_csv(path)`` under pandas
    defaults (index=True, header=True, na_rep='', QUOTE_MINIMAL,
    lineterminator='\\n'), restricted to the column dtypes produced by
    ThermoMPNN inference outputs: float64 (NaN -> ''), int64/uint64, and
    object columns holding str/NaN/None.

    Falls back to ``df.to_csv(path)`` on any unexpected dtype or any value
    that would require CSV quoting, so behavior is preserved by construction.
    Byte-identity with pandas output was fuzz-verified on 2.5M+ adversarial
    rows (bench/fuzz_csv_writer.py) and is enforced by the correctness suite.
    """
    import pandas as pd  # lazy: keep this module's import graph unchanged
    cols = list(df.columns)
    fcols = []
    for c in cols:
        k = df[c].dtype.kind
        if k == 'f':
            fcols.append(['' if v != v else repr(v) for v in df[c].tolist()])
        elif k in ('i', 'u'):
            fcols.append(list(map(str, df[c].tolist())))
        elif k == 'O':
            vals = df[c].tolist()
            types = set(map(type, vals))
            if types <= {str}:
                joined = '\x00'.join(vals)
                if _needs_csv_quote(joined):
                    df.to_csv(path)
                    return
                fcols.append(vals)
            elif types <= {str, float, type(None)}:
                out = []
                ap = out.append
                for v in vals:
                    if type(v) is str:
                        if _needs_csv_quote(v):
                            df.to_csv(path)
                            return
                        ap(v)
                    elif v is None or v != v:
                        ap('')
                    else:
                        ap(repr(v))
                fcols.append(out)
            else:
                df.to_csv(path)
                return
        else:
            df.to_csv(path)
            return
    idx = df.index
    if isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1:
        idx_s = list(map(str, range(len(df))))
    else:
        idx_s = list(map(str, idx.tolist()))
    lines = [',' + ','.join(cols)]
    lines.extend(','.join(r) for r in zip(idx_s, *fcols))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
