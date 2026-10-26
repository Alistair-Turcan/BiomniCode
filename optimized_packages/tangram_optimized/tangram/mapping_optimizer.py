"""
Library for instantiating and running the optimizer for Tangram. The optimizer comes in two flavors,
which correspond to two different classes:
- Mapper: optimizer without filtering (i.e., all single cells are mapped onto space). At the end, the learned mapping
matrix M is returned.
- MapperConstrained: optimizer with filtering (i.e., only a subset of single cells are mapped onto space).
At the end, the learned mapping matrix M and the learned filter F are returned.
"""
import numpy as np
import logging
import torch
from torch.nn.functional import softmax
from ._optim import _make_adam


class _SoftmaxProjectionDensity(torch.autograd.Function):
    """Joint CPU VJP, using native softmax and owned incoming-gradient storage."""
    @staticmethod
    def forward(ctx, logits, expression):
        probabilities = softmax(logits, dim=1)
        ctx.save_for_backward(logits, expression, probabilities)
        return probabilities.t() @ expression, probabilities.sum(dim=0)

    @staticmethod
    def backward(ctx, projection_gradient, density_gradient):
        logits, expression, probabilities = ctx.saved_tensors
        incoming = expression @ projection_gradient.t()
        incoming.add_(density_gradient)
        if torch.is_grad_enabled():
            # Saved intermediates from a custom forward do not carry their graph.
            probabilities = softmax(logits, dim=1)
            result = torch.ops.aten._softmax_backward_data.default(
                incoming, probabilities, 1, logits.dtype)
        else:
            result = torch.ops.aten._softmax_backward_data.out(
                incoming, probabilities, 1, logits.dtype, grad_input=incoming)
        return result, None


def _unit_reference(x, dim):
    """Normalize a constant reference using cosine_similarity's epsilon rule."""
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(1e-8)


def _cosine_cached(x, reference_unit, dim):
    """Cosine with a pre-normalized constant and upstream's clamped-norm VJP.

    Clamping a clone under no_grad is intentional: PyTorch cosine_similarity
    retains the norm's derivative even below eps. Adding eps to the product,
    or differentiating through clamp_min instead, changes that gradient.
    """
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True).clone()
    with torch.no_grad():
        norm.clamp_(min=1e-8)
    return ((x / norm) * reference_unit).sum(dim=dim)


class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """
    def __init__(
        self,
        S,
        G,
        train_genes_idx=None,
        val_genes_idx=None,
        d=None,
        d_source=None,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_neighborhood_g1=0,
        voxel_weights=None,
        lambda_getis_ord=0,
        lambda_geary=0,
        lambda_moran=0,
        neighborhood_filter=None,
        ct_encode=None,
        lambda_ct_islands=0,
        spatial_weights=None,
        device="cpu",
        adata_map=None,
        random_state=None,
    ):
        """
        Instantiate the Tangram optimizer (without filtering).

        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            train_genes_idx (ndarray): Optional. Gene indices used for training.
            val_genes_idx (ndarray): Optional. Gene indices used for validation.
            d (ndarray): Spatial density of cells, shape = (number_spots,). If not provided, the density term is ignored.
                This array should satisfy the constraints d.sum() == 1.
            d_source (ndarray): Density of single cells in single cell clusters. To be used when S corresponds to cluster-level expression.
                This array should satisfy the constraint d_source.sum() == 1.
            lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
            lambda_d (float): Optional. Strength of density regularizer. Default is 0.
            lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
            lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes
                              probabilities of each cell peaked over a narrow portion of space.
                              lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_l1 (float): Optional. Strength of L1 regularizer. Default is 0.
            lambda_l2 (float): Optional. Strength of L2 regularizer. Default is 0.
            lambda_neighborhood_g1 (float): Optional. Strength of neighborhood weighted gene expression comparison. Default is 0.
            voxel_weights (ndarray): Optional. Spatial weight used for neighborhood weighting, shape = (number_spots, number_spots).
            lambda_getis_ord (float): Optional. Strength of Getis-Ord G* preservation. Default is 0.
            lambda_geary (float): Optional. Strength of Geary's C preservation. Default is 0.
            lambda_moran (float): Optional. Strength of Moran's I preservation. Default is 0.
            spatial_weights (ndarray): Optional. Spatial weight used for local spatial indicator preservation, shape = (number_spots, number_spots).
            lambda_ct_islands: Optional. Strength of ct islands enforcement. Default is 0.
            neighborhood_filter (ndarray): Optional. Neighborhood filter used for cell type island preservation, shape = (number_spots, number_spots).
            ct_encode(ndarray): Optional. One-hot encoding of cell types used for cell type island preservation, shape = (number_cells, number_celltypes).
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.device = device
        self.random_state = random_state

        # Own the input buffers, as upstream does (CPU from_numpy aliases).
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        # setup training and validation data
        if train_genes_idx is not None:
            self.S_train = self.S[:, train_genes_idx].clone()
            self.G_train = self.G[:, train_genes_idx].clone()
        else:
            self.S_train = self.S.clone()
            self.G_train = self.G.clone()
        if val_genes_idx is not None:
            self.S_val = self.S[:, val_genes_idx].clone()
            self.G_val = self.G[:, val_genes_idx].clone()
        else:
            self.S_val = self.S.clone()
            self.G_val = self.G.clone()
        
        # loss weights
        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_neighborhood_g1 = lambda_neighborhood_g1
        self.lambda_ct_islands = lambda_ct_islands
        self.lambda_getis_ord = lambda_getis_ord
        self.lambda_geary = lambda_geary
        self.lambda_moran = lambda_moran

        # density terms
        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(np.asarray(d), device=device, dtype=torch.float32)

        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)
        

        # spatial weights for neighborhood weighting 
        self.voxel_weights = voxel_weights
        if self.voxel_weights is not None:
            self.voxel_weights = torch.tensor(voxel_weights, device=device, dtype=torch.float32)
            # Precompute W@G_train and its norms (G is constant across epochs)
            self.G_neigh = self.voxel_weights @ self.G_train
            self.G_neigh_unit = _unit_reference(self.G_neigh, 0)
       
        # cell type islands
        self.neighborhood_filter = neighborhood_filter
        if self.neighborhood_filter is not None:
            self.neighborhood_filter = torch.tensor(neighborhood_filter, device=device, dtype=torch.float32)
        
        self.ct_encode = ct_encode
        if self.ct_encode is not None:
            self.ct_encode = torch.tensor(ct_encode, device=device, dtype=torch.float32)

        # spatial autocorrelation weights
        self.spatial_weights = spatial_weights
        if self.spatial_weights is not None:
            self.spatial_weights = torch.tensor(spatial_weights, device=device, dtype=torch.float32)

        # Cache normalized constant references, not just their norms.
        self.G_train_unit = _unit_reference(self.G_train, 0)
        self.G_train_unit_vg = _unit_reference(self.G_train, 1)
        # Precompute gene sparsity for validation loss (G is constant)
        self.gene_sparsity = 1 - ((self.G_train != 0).sum(axis=0) / self.G_train.shape[0]).reshape((-1,))

        # precompute spatial local indicators for target preservation
        self.getis_ord_G_star_ref, self.moran_I_ref, self.gearys_C_ref = self._spatial_local_indicators(self.G_train)
        self.getis_ord_ref_unit = _unit_reference(self.getis_ord_G_star_ref, 0) if self.getis_ord_G_star_ref is not None else None
        self.moran_ref_unit = _unit_reference(self.moran_I_ref, 0) if self.moran_I_ref is not None else None
        self.gearys_ref_unit = _unit_reference(self.gearys_C_ref, 0) if self.gearys_C_ref is not None else None

        # initialize mapping matrix
        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented
            self.M = adata_map.X  # doesn't work. maybe apply inverse softmax

        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

        # Precompute constant terms for KL divergence.
        if self.target_density_enabled:
            # xlogy(0, 0) is zero, unlike 0 * log(0). Keep the per-element
            # entropy term to retain KLDivLoss's reduction order.
            self._kl_entropy = torch.xlogy(self.d, self.d)

    def _spatial_local_indicators(self, G):
        """
        Compute spatial local indicators for given gene expression matrix.

        Args:
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
        
        Returns:
            Tuple of 3 Arrays: getis_ord_G_star, moran_I, gearys_C
        """
        # Cache W@G across all spatial indicators that need it
        W = self.spatial_weights
        WG = None  # computed lazily when first needed
        G_mean = None
        
        getis_ord_G_star = None
        if self.lambda_getis_ord > 0:
            WG = W @ G if WG is None else WG
            getis_ord_G_star = WG / G.sum(axis=0)
        
        moran_I = None
        if self.lambda_moran > 0:
            G_mean = G.mean(axis=0) if G_mean is None else G_mean
            z = G - G_mean
            # Center before multiplying: subtracting W@G - rowsum*mean
            # catastrophically cancels for a large common offset.
            Wz = W @ z
            moran_I = (G.shape[0] * z * Wz) / (z * z).sum(axis=0)
        
        gearys_C = None
        if self.lambda_geary > 0:
            n_spots, n_genes = G.shape
            G_mean = G.mean(axis=0) if G_mean is None else G_mean
            m2 = ((G - G_mean) ** 2).sum(axis=0) / (n_spots - 1)
            # Subtract a common *input* row, not a rounded mean. Differences
            # are unchanged while large offsets no longer cancel in squares.
            H = G - G[:1]
            WH = W @ H
            W_rowsum = W.sum(dim=1)  # (n_spots,)
            W_colsum = W.sum(dim=0)  # (n_spots,)
            diff_sq_sum = (W_rowsum[:, None] * H ** 2).sum(dim=0) - 2 * (H * WH).sum(dim=0) + (W_colsum[:, None] * H ** 2).sum(dim=0)
            gearys_C = diff_sq_sum / (2 * m2)
        
        return getis_ord_G_star, moran_I, gearys_C

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed. Default is True.

        Returns:
            Tuple of 5 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg
        """
        G = self.G_train
        S = self.S_train
        joint = (self.M.device.type == "cpu" and not S.requires_grad
                 and self.target_density_enabled and not self.source_density_enabled
                 and self.lambda_r == 0 and self.lambda_ct_islands <= 0)
        if joint:
            G_pred, density_sum = _SoftmaxProjectionDensity.apply(self.M, S)
        else:
            M_probs = softmax(self.M, dim=1)
            G_pred = torch.matmul(M_probs.t(), S)
        
        # gene expression similarity terms
        # Preserve the value and gradient of upstream cosine_similarity.
        gv_term = self.lambda_g1 * _cosine_cached(G_pred, self.G_train_unit, 0).mean()
        if self.lambda_g2 != 0:
            vg_term = self.lambda_g2 * _cosine_cached(G_pred, self.G_train_unit_vg, 1).mean()
            vg_reg = (vg_term / self.lambda_g2).detach() if self.lambda_g2 != 0 else np.nan
        else:
            vg_term = 0
            vg_reg = np.nan
        expression_term = gv_term + vg_term
        main_loss = (gv_term / self.lambda_g1).detach()
        
        # density regularization terms
        if self.target_density_enabled:
            # KL wants the log in first argument
            if self.source_density_enabled:
                d_pred = torch.log(self.d_source @ M_probs)
            else:
                d_pred = torch.log((density_sum if joint else M_probs.sum(dim=0))
                                   / self.M.shape[0])
            density_term = self.lambda_d * (self._kl_entropy - self.d * d_pred).sum()
            kl_reg = (density_term / self.lambda_d).detach()
        else:
            density_term, kl_reg = 0, np.nan
        
        # entropy regularization terms
        if self.lambda_r != 0:
            log_M_probs = torch.log(M_probs)
            entropy_term = self.lambda_r * -(log_M_probs * M_probs).sum()
            entropy_reg = (entropy_term / self.lambda_r).detach() if self.lambda_r != 0 else np.nan
        else:
            entropy_term = 0
            entropy_reg = np.nan

        # l1 and l2 regularization terms
        if self.lambda_l1 != 0:
            l1_term = self.lambda_l1 * self.M.abs().sum()
            l1_reg = (l1_term / self.lambda_l1).detach()
        else:
            l1_term = 0
            l1_reg = np.nan
        if self.lambda_l2 != 0:
            l2_term = self.lambda_l2 * (self.M ** 2).sum()
            l2_reg = (l2_term / self.lambda_l2).detach()
        else:
            l2_term = 0
            l2_reg = np.nan

        # spatial neighborhood-based gene expression term
        if self.lambda_neighborhood_g1 > 0:
            G_pred_neigh = self.voxel_weights @ G_pred
            gv_neighborhood_term = self.lambda_neighborhood_g1 * _cosine_cached(
                G_pred_neigh, self.G_neigh_unit, 0).mean()
            gv_neighborhood_sim = (gv_neighborhood_term / self.lambda_neighborhood_g1).detach()
        else:
            gv_neighborhood_term, gv_neighborhood_sim = 0, np.nan

        # cell type island enforcement
        if self.lambda_ct_islands > 0:
            ct_map = (M_probs.T @ self.ct_encode)
            ct_island_term = self.lambda_ct_islands * torch.maximum(
                ct_map - self.neighborhood_filter @ ct_map,
                torch.zeros((), dtype=torch.float32, device=self.device)).mean()
            ct_island_penalty = (ct_island_term / self.lambda_ct_islands).detach()
        else:
            ct_island_term, ct_island_penalty = 0, np.nan

        # spatial autocorrelation metrics
        getis_ord_G_star_pred, moran_I_pred, gearys_C_pred = self._spatial_local_indicators(G_pred)

        getis_ord_term, moran_term, gearys_term = 0, 0, 0
        getis_ord_sim, moran_sim, gearys_sim = np.nan, np.nan, np.nan
        if self.lambda_getis_ord > 0:
            getis_ord_term = self.lambda_getis_ord * _cosine_cached(
                getis_ord_G_star_pred, self.getis_ord_ref_unit, 0).mean()
            getis_ord_sim = (getis_ord_term / self.lambda_getis_ord).detach()
        if self.lambda_moran > 0:
            moran_term = self.lambda_moran * _cosine_cached(
                moran_I_pred, self.moran_ref_unit, 0).mean()
            moran_sim = (moran_term / self.lambda_moran).detach()
        if self.lambda_geary > 0:
            gearys_term = self.lambda_geary * _cosine_cached(
                gearys_C_pred, self.gearys_ref_unit, 0).mean()
            gearys_sim = (gearys_term / self.lambda_geary).detach()

        # total loss aggregation
        total_loss = -expression_term \
                 + density_term + entropy_term \
                 + l1_term + l2_term \
                 + ct_island_term - gv_neighborhood_term \
                 - getis_ord_term - moran_term - gearys_term

        if verbose:
            term_numbers = [
                main_loss,
                vg_reg, 
                kl_reg, 
                entropy_reg, 
                l1_reg, 
                l2_reg, 
                gv_neighborhood_sim, 
                ct_island_penalty, 
                getis_ord_sim, 
                moran_sim, 
                gearys_sim
            ]
            term_names = [
                "Gene-voxel score", 
                "Voxel-gene score", 
                "Cell densities reg", 
                "Entropy reg", 
                "L1 reg", 
                "L2 reg",
                "Spatial weighted score", 
                "Cell type islands penalty", 
                "Getis-Ord score", 
                "Moran score", 
                "Geary score"
            ]

            # Convert tensor terms to Python floats for printing
            term_numbers_float = []
            for tn in term_numbers:
                if isinstance(tn, torch.Tensor):
                    term_numbers_float.append(tn.item())
                else:
                    term_numbers_float.append(tn)
            d = dict(zip(term_names, term_numbers_float))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        return total_loss, main_loss, vg_reg, kl_reg, entropy_reg

    def _val_loss_fn(self, verbose=False):
        """
        Evaluates the validation loss function. Used during hyperparameter tuning.

        Args:
            verbose (bool): Optional. Whether to print the val results. If True, the loss for each individual term is printed as. Default is True.

        Returns:
            Tuple of 5 Floats: expression_sim, gv_sim, sp_sparsity_weighted_gv_sim, entropy
        """
        G = self.G_train
        S = self.S_train
        M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)

        # Reuse constant normalization and the gene-wise prediction score.
        cos_sim_gv = _cosine_cached(G_pred, self.G_train_unit, 0)
        gv_sim = cos_sim_gv.mean().item()
        vg_sim = _cosine_cached(G_pred, self.G_train_unit_vg, 1).mean().item()
        expression_sim = gv_sim + vg_sim

        sp_sparsity_weighted_gv_sim = ((cos_sim_gv * (1-self.gene_sparsity)) / (1-self.gene_sparsity).sum()).sum().item()

        log_M_probs = torch.log(M_probs)
        entropy = -((log_M_probs * M_probs).sum(axis=1) / np.log(M_probs.shape[1])).mean().item()

        if verbose:
            term_numbers = [
                gv_sim, 
                sp_sparsity_weighted_gv_sim, 
                entropy
            ]
            term_names = [
                "Val gene-voxel score", 
                "Val gene-voxel sparsity-weighted score", 
                "Val map entropy"
            ]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        return expression_sim, gv_sim, sp_sparsity_weighted_gv_sim, entropy

    def train(self, num_epochs, learning_rate=0.1, print_each=100, val_each=None):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
            val_each (int): Optional: Evaluate the val loss each val_each epochs. If None, the val loss is never computed. Default is None.
        Returns:
            output (ndarray): The optimized mapping matrix M (ndarray), with shape (number_cells, number_spots).
            training_history (dict): loss for each epoch
        """
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        # Match upstream Adam arithmetic. Fused Adam can amplify tiny
        # roundoff into different non-convex mapping trajectories.
        optimizer = _make_adam([self.M], learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["total_loss", "main_loss", "vg_reg", "kl_reg", "entropy_reg"]
        val_keys = ["val_total_loss", "val_gene_sim", "val_sp_sparsity_weighted_sim", "val_entropy"]
        training_history = {key: [] for key in keys + val_keys}

        # Pre-allocate lists for deferred history conversion
        _hist_total_loss = []
        _hist_main_loss = []
        _hist_vg_reg = []
        _hist_kl_reg = []
        _hist_entropy_reg = []
        _hist_lists = [_hist_total_loss, _hist_main_loss, _hist_vg_reg, _hist_kl_reg, _hist_entropy_reg]

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]

            # Snapshot detached scalars; transfer histories in bulk at the end.
            _hist_total_loss.append(run_loss[0].clone().detach())
            _hist_main_loss.append(run_loss[1].clone() if isinstance(run_loss[1], torch.Tensor) else run_loss[1])
            _hist_vg_reg.append(run_loss[2].clone() if isinstance(run_loss[2], torch.Tensor) else run_loss[2])
            _hist_kl_reg.append(run_loss[3].clone() if isinstance(run_loss[3], torch.Tensor) else run_loss[3])
            _hist_entropy_reg.append(run_loss[4].clone() if isinstance(run_loss[4], torch.Tensor) else run_loss[4])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if val_each is not None:
                with torch.no_grad():
                    if val_each is not None and t % val_each == 0:
                        val_loss = self._val_loss_fn(verbose=False)
                        for i in range(len(val_keys)):
                            training_history[val_keys[i]].append(val_loss[i])

        # Convert collected tensors to Python values in bulk (single sync at end)
        # Preserve original types: total_loss as numpy.ndarray, others as float
        for key, hist_list in zip(keys, _hist_lists):
            if hist_list and isinstance(hist_list[0], torch.Tensor):
                values = torch.stack(hist_list).detach().cpu().numpy()
                training_history[key] = ([np.asarray(v) for v in values]
                                         if key == "total_loss" else values.tolist())
            else:
                training_history[key] = hist_list

        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output, training_history


class MapperConstrained:
    """
    Allows instantiating and running the optimizer for Tangram, with filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping and filter results.
    """

    def __init__(
        self,
        S,
        G,
        d,
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=1,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        device="cpu",
        adata_map=None,
        random_state=None,
    ):
        """
        Instantiate the Tangram optimizer (with filtering).

        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            d (ndarray): Spatial density of cells, shape = (number_spots,).
                         This array should satisfy the constraints d.sum() == 1.
            lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 1.
            lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
            lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
            lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes
                              probabilities of each cell peaked over a narrow portion of space.
                              lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_count (float): Optional. Regularizer for the count term. Default is 1.
            lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Default is 1.
            target_count (int): Optional. The number of cells to be filtered. If None, this number defaults to the number of
                                voxels inferred by the matrix 'G'. Default is None.
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.device = device
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(np.asarray(d), device=device, dtype=torch.float32)

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_count = lambda_count
        self.lambda_f_reg = lambda_f_reg
        self.random_state = random_state

        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented
            self.M = adata_map.X  # doesn't work. maybe apply inverse softmax

        if target_count is None:
            self.target_count = self.G.shape[0]
        else:
            self.target_count = target_count

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

        self.F = np.random.normal(0, 1, S.shape[0])
        self.F = torch.tensor(
            self.F, device=device, requires_grad=True, dtype=torch.float32
        )


        # Precompute G norms for fast cosine similarity
        self.G_unit = _unit_reference(self.G, 0)
        self.G_unit_vg = _unit_reference(self.G, 1)

        # Precompute constant for manual KL divergence
        if self.target_density_enabled:
            self._kl_entropy = torch.xlogy(self.d, self.d)

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.

        Returns:
            Tuple of 7 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg, count_reg, lambda_f_reg
        """
        M_probs = softmax(self.M, dim=1)
        F_probs = torch.sigmoid(self.F)

        if self.target_density_enabled:
            # Keep upstream's multiply-then-reduce operation order. A vector
            # matmul has different FP32 reductions and can change filtering
            # decisions after hundreds of Adam steps.
            d_pred = torch.log(
                (M_probs * F_probs[:, None]).sum(dim=0) / F_probs.sum())
            density_term = self.lambda_d * (self._kl_entropy - self.d * d_pred).sum()
        else:
            density_term = None

        S_filtered = self.S * F_probs.unsqueeze(1)

        G_pred = torch.matmul(M_probs.t(), S_filtered)
        # Reuse normalized constant references.
        gv_term = self.lambda_g1 * _cosine_cached(G_pred, self.G_unit, 0).mean()
        if self.lambda_g2 != 0:
            vg_term = self.lambda_g2 * _cosine_cached(G_pred, self.G_unit_vg, 1).mean()
        else:
            vg_term = 0
        expression_term = gv_term + vg_term

        if self.lambda_r != 0:
            log_M_probs = torch.log(M_probs)
            entropy_term = self.lambda_r * (log_M_probs * M_probs).sum()
        else:
            entropy_term = 0

        _count_term = F_probs.sum() - self.target_count
        count_term = self.lambda_count * torch.abs(_count_term)

        f_reg_t = F_probs - F_probs * F_probs
        f_reg = self.lambda_f_reg * f_reg_t.sum()

        main_loss = (gv_term / self.lambda_g1).detach()
        kl_reg = (
            (density_term / self.lambda_d).detach()
            if self.target_density_enabled
            else np.nan
        )
        entropy_reg = (entropy_term / self.lambda_r).detach() if self.lambda_r != 0 else np.nan
        vg_reg = (vg_term / self.lambda_g2).detach() if self.lambda_g2 != 0 else np.nan
        count_reg = (count_term / self.lambda_count).detach()
        lambda_f_reg = (f_reg / self.lambda_f_reg).detach()

        if verbose:
            term_numbers = [
                main_loss,
                vg_reg,
                kl_reg,
                entropy_reg,
                count_reg,
                lambda_f_reg,
            ]
            term_names = [
                "Score",
                "VG reg",
                "KL reg",
                "Entropy reg",
                "Count reg",
                "Lambda f reg",
            ]

            # Convert tensor terms to Python floats for printing
            term_numbers_float = []
            for tn in term_numbers:
                if isinstance(tn, torch.Tensor):
                    term_numbers_float.append(tn.item())
                else:
                    term_numbers_float.append(tn)
            score_dict = dict(zip(term_names, term_numbers_float))
            clean_dict = {
                k: score_dict[k] for k in score_dict if not np.isnan(score_dict[k])
            }
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        total_loss = -expression_term - entropy_term + count_term + f_reg
        if self.target_density_enabled:
            total_loss = total_loss + density_term

        return (
            total_loss,
            main_loss,
            vg_reg,
            kl_reg,
            entropy_reg,
            count_reg,
            lambda_f_reg,
        )

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

        Returns:
            A tuple (output, F_out, training_history), with:
                M (ndarray): is the optimized mapping matrix, shape = (number_cells, number_spots).
                f (ndarray): is the optimized filter, shape = (number_cells,).
                training_history (dict): loss for each epoch
        """

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = _make_adam([self.M, self.F], learning_rate)

        keys = [
            "total_loss",
            "main_loss",
            "vg_reg",
            "kl_reg",
            "entropy_reg",
            "count_reg",
            "lambda_f_reg",
        ]
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}

        # Pre-allocate lists for deferred history conversion
        _hist_lists = [[] for _ in range(len(keys))]
        _loss_grad_names = []

        _is_cuda = self.M.is_cuda

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]

            # Detach every value: retaining total_loss's graph here otherwise
            # retains the entire graph of every epoch until train() returns.
            _loss_grad_names.append(type(loss.grad_fn).__name__)
            for i in range(len(keys)):
                val = run_loss[i]
                _hist_lists[i].append(val.detach().clone() if isinstance(val, torch.Tensor) else val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Convert collected tensors to strings in bulk (preserve original str() format)
        for key, hist_list in zip(keys, _hist_lists):
            if hist_list and isinstance(hist_list[0], torch.Tensor):
                values = torch.stack(hist_list).detach().cpu()
                if key == "total_loss":
                    device_suffix = f", device='{self.M.device}'" if _is_cuda else ""
                    for val, grad_name in zip(values, _loss_grad_names):
                        # Upstream stores Tensor.__str__ for total_loss.
                        text = str(val)
                        training_history[key].append(
                            text[:-1] + device_suffix + f", grad_fn=<{grad_name}>)")
                else:
                    training_history[key] = [str(v) for v in values.tolist()]
            else:
                training_history[key] = [str(v) for v in hist_list]

        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            F_out = torch.sigmoid(self.F).cpu().numpy()
            return output, F_out, training_history
