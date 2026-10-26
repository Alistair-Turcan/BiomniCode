import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_mpnn_utils import ProteinMPNN, tied_featurize
import os


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
# iter049: O(1) residue->index lookup for the per-mutation head listcomps
# (identical mapping to ALPHABET.index; ALPHABET kept as the source of truth)
_AA_IDX = {a: k for k, a in enumerate(ALPHABET)}

MLP = True
SUBTRACT_MUT = True


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3 

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    # Buffer the checkpoint through memory: torch.load issues one small read
    # per tensor record, which is latency-bound on high-latency filesystems
    # (network/FUSE mounts); a single streaming read is much faster there and
    # equivalent on local disk. The parsed state_dict is byte-identical.
    with open(checkpoint_path, 'rb') as fh:
        checkpoint = torch.load(io.BytesIO(fh.read()), map_location='cpu')
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0)
    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model


class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128
        HIDDEN_DIM = 128

        hid_sizes = [ HIDDEN_DIM*self.num_final_layers + EMBED_DIM ]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        # print('MLP HIDDEN SIZES:', hid_sizes)

        if self.lightattn:
            # print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=HIDDEN_DIM*self.num_final_layers + EMBED_DIM )

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(self, pdb, mutations, tied_feat=True, scalar_floats=False):
        # scalar_floats=False (default): "ddG" entries are shape-[1] tensors,
        # preserving the contract relied on by training (F.mse_loss) and the
        # torchmetrics benchmarking loops. scalar_floats=True (inference fast
        # path): "ddG" entries are Python floats from one batched tolist(),
        # replacing ~N per-mutation select+unsqueeze dispatches; values are
        # bitwise-identical float32 scalars either way.
        device = next(self.parameters()).device

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)

        # Batched mutation scoring: numerically equivalent to the original
        # per-mutation loop. Each mutation's feature vector is independent, and
        # LightAttention's convs/softmax act per-sample on a length-1 sequence
        # axis, so all mutations can be scored in one batched pass.
        valid_idx = [i for i, mut in enumerate(mutations) if mut is not None]
        out = [None] * len(mutations)
        if valid_idx:
            pos = torch.tensor([mutations[i].position for i in valid_idx], dtype=torch.long, device=device)
            aa_index = torch.tensor([_AA_IDX[mutations[i].mutation] for i in valid_idx], dtype=torch.long, device=device)
            wt_aa_index = torch.tensor([_AA_IDX[mutations[i].wildtype] for i in valid_idx], dtype=torch.long, device=device)

            inputs = []
            if self.num_final_layers > 0:
                inputs.append(mpnn_hid[0][pos])  # MPNN hidden embeddings at mut positions
            inputs.append(mpnn_embed[0][pos])  # MPNN seq embeddings at mut positions

            # concatenating hidden layers and embeddings
            lin_input = torch.cat(inputs, -1)  # [N, D]

            # passing vector through lightattn
            if self.lightattn:
                lin_input = torch.unsqueeze(lin_input, -1)  # [N, D, 1]
                lin_input = self.light_attention(lin_input, mask)
                if lin_input.dim() == 1:  # N == 1: squeeze removed the batch dim
                    lin_input = torch.unsqueeze(lin_input, 0)

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)  # [N, 21, 1]
            ddg_out = self.ddg_out(both_input)  # [N, 21, 1]

            rows = torch.arange(len(valid_idx), device=device)
            if self.subtract_mut:
                ddg_all = ddg_out[rows, aa_index, 0] - ddg_out[rows, wt_aa_index, 0]
            else:
                ddg_all = ddg_out[rows, aa_index, 0]

            if scalar_floats:
                ddg_vals = ddg_all.tolist()  # one batched extraction
                for j, i in enumerate(valid_idx):
                    out[i] = {
                        "ddG": ddg_vals[j],
                    }
            else:
                for j, i in enumerate(valid_idx):
                    out[i] = {
                        "ddG": torch.unsqueeze(ddg_all[j], 0),
                    }
        return out, None


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # Fast path for a length-1 sequence axis (the only case produced by
        # TransferModel, which scores each mutation as an independent [N, D, 1]
        # feature). With L == 1 and padding = kernel_size // 2, only the center
        # tap of each conv kernel multiplies non-padding data, so the conv is
        # exactly a Linear with weight W[:, :, k//2]; softmax over a singleton
        # axis is exactly 1, making the attention branch a no-op; dropout is
        # identity in eval mode. The mask argument is unused in the original
        # path as well. Restricted to eval mode so training-time dropout
        # behavior is preserved bitwise via the original path.
        if x.shape[-1] == 1 and not self.training:
            w = self.feature_convolution.weight
            out = F.linear(x[:, :, 0], w[:, :, w.shape[-1] // 2],
                           self.feature_convolution.bias)
            return torch.squeeze(out)

        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
