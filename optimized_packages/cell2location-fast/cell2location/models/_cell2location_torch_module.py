"""Direct PyTorch implementation of Cell2location model — no Pyro dependency."""
from typing import Optional, Dict
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

class Cell2locationTorchModule(nn.Module):
    def __init__(self, n_obs, n_vars, n_factors, n_batch, cell_state_mat,
                 n_groups=50, detection_mean=0.5, detection_alpha=20.0,
                 m_g_gene_level_prior=None, N_cells_per_location=8.0,
                 A_factors_per_location=7.0, B_groups_per_location=7.0,
                 N_cells_mean_var_ratio=1.0, alpha_g_phi_hyp_prior=None,
                 gene_add_alpha_hyp_prior=None, gene_add_mean_hyp_prior=None,
                 detection_hyp_prior=None, w_sf_mean_var_ratio=5.0,
                 init_vals=None, init_alpha=20.0, dropout_p=0.0, init_scale=0.1):
        super().__init__()
        if m_g_gene_level_prior is None: m_g_gene_level_prior = {"mean": 1, "mean_var_ratio": 1.0, "alpha_mean": 3.0}
        if alpha_g_phi_hyp_prior is None: alpha_g_phi_hyp_prior = {"alpha": 9.0, "beta": 3.0}
        if gene_add_alpha_hyp_prior is None: gene_add_alpha_hyp_prior = {"alpha": 9.0, "beta": 3.0}
        if gene_add_mean_hyp_prior is None: gene_add_mean_hyp_prior = {"alpha": 1.0, "beta": 100.0}
        if detection_hyp_prior is None: detection_hyp_prior = {"mean_alpha": 10.0}
        detection_hyp_prior["mean"] = detection_mean; detection_hyp_prior["alpha"] = detection_alpha
        self.n_obs = n_obs; self.n_vars = n_vars; self.n_factors = n_factors
        self.n_batch = n_batch; self.n_groups = n_groups; self.dropout_p = dropout_p
        self.register_buffer("detection_hyp_prior_alpha", torch.tensor(detection_hyp_prior["alpha"]))
        self.register_buffer("detection_mean_hyp_prior_alpha", torch.tensor(detection_hyp_prior["mean_alpha"]))
        self.register_buffer("detection_mean_hyp_prior_beta", torch.tensor(detection_hyp_prior["mean_alpha"] / detection_hyp_prior["mean"]))
        self.register_buffer("m_g_mu_hyp", torch.tensor(m_g_gene_level_prior["mean"]))
        self.register_buffer("m_g_mu_mean_var_ratio_hyp", torch.tensor(m_g_gene_level_prior["mean_var_ratio"]))
        self.register_buffer("m_g_alpha_hyp_mean", torch.tensor(m_g_gene_level_prior["alpha_mean"]))
        self.register_buffer("cell_state", torch.tensor(cell_state_mat.T).contiguous())
        self.register_buffer("N_cells_per_location", torch.tensor(N_cells_per_location))
        self.register_buffer("factors_per_groups", torch.tensor(A_factors_per_location / B_groups_per_location))
        self.register_buffer("B_groups_per_location", torch.tensor(B_groups_per_location))
        self.register_buffer("N_cells_mean_var_ratio", torch.tensor(N_cells_mean_var_ratio))
        self.register_buffer("alpha_g_phi_hyp_prior_alpha", torch.tensor(alpha_g_phi_hyp_prior["alpha"]))
        self.register_buffer("alpha_g_phi_hyp_prior_beta", torch.tensor(alpha_g_phi_hyp_prior["beta"]))
        self.register_buffer("gene_add_alpha_hyp_prior_alpha", torch.tensor(gene_add_alpha_hyp_prior["alpha"]))
        self.register_buffer("gene_add_alpha_hyp_prior_beta", torch.tensor(gene_add_alpha_hyp_prior["beta"]))
        self.register_buffer("gene_add_mean_hyp_prior_alpha", torch.tensor(gene_add_mean_hyp_prior["alpha"]))
        self.register_buffer("gene_add_mean_hyp_prior_beta", torch.tensor(gene_add_mean_hyp_prior["beta"]))
        self.register_buffer("w_sf_mean_var_ratio_tensor", torch.tensor(w_sf_mean_var_ratio))
        self.register_buffer("n_factors_tensor", torch.tensor(self.n_factors))
        self.register_buffer("n_groups_tensor", torch.tensor(self.n_groups))
        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_1_n_groups", torch.ones((1, self.n_groups)))
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))
        if dropout_p and dropout_p != 0: self.dropout = nn.Dropout(p=dropout_p)
        self.init_alpha = init_alpha; self.init_vals = init_vals
        # Variational parameters.
        # Scales are stored unconstrained and passed through softplus in _sl(), exactly like
        # Pyro AutoNormal's PyroParam(0.1, constraint=softplus_positive). The raw init must
        # therefore be softplus^{-1}(init_scale) = log(exp(init_scale) - 1) (~ -2.2522 for 0.1);
        # initialising the raw parameter to 0.1 would give an effective initial scale of
        # softplus(0.1) ~ 0.744, i.e. ~7.4x more initial posterior noise than the Pyro guide.
        init_scale_unconstrained = math.log(math.expm1(init_scale))
        def mk(shape): return (nn.Parameter(torch.zeros(shape)), nn.Parameter(torch.full(shape, init_scale_unconstrained)))
        (self.locs_m_g_mean, self.scales_m_g_mean) = mk((1, 1))
        (self.locs_m_g_alpha_e_inv, self.scales_m_g_alpha_e_inv) = mk((1, 1))
        (self.locs_m_g, self.scales_m_g) = mk((1, n_vars))
        (self.locs_k_r, self.scales_k_r) = mk((n_groups, 1))
        (self.locs_x_fr, self.scales_x_fr) = mk((n_groups, n_factors))
        (self.locs_dmye, self.scales_dmye) = mk((n_batch, 1))
        (self.locs_sgah, self.scales_sgah) = mk((1, 1))
        (self.locs_sgm, self.scales_sgm) = mk((n_batch, 1))
        (self.locs_sgai, self.scales_sgai) = mk((n_batch, 1))
        (self.locs_sga, self.scales_sga) = mk((n_batch, n_vars))
        (self.locs_agph, self.scales_agph) = mk((1, 1))
        (self.locs_agi, self.scales_agi) = mk((n_batch, n_vars))
        (self.locs_nsc, self.scales_nsc) = mk((n_obs, 1))
        (self.locs_bsg, self.scales_bsg) = mk((n_obs, 1))
        (self.locs_zsr, self.scales_zsr) = mk((n_obs, n_groups))
        (self.locs_wsf, self.scales_wsf) = mk((n_obs, n_factors))
        (self.locs_dys, self.scales_dys) = mk((n_obs, 1))
        self.aggressive_param_names = [n for n, _ in self.named_parameters() if any(x in n for x in ["nsc", "bsg", "zsr", "wsf", "dys"])]

    def init_to_mean(self):
        with torch.no_grad():
            self.locs_m_g_mean.copy_(torch.log(self.m_g_mu_hyp.expand_as(self.locs_m_g_mean) + 1e-10))
            self.locs_m_g_alpha_e_inv.copy_(torch.log((1.0/self.m_g_alpha_hyp_mean).expand_as(self.locs_m_g_alpha_e_inv) + 1e-10))
            mgm = torch.exp(self.locs_m_g_mean)
            self.locs_m_g.copy_(torch.log(mgm.expand_as(self.locs_m_g) + 1e-10))
            self.locs_nsc.copy_(torch.log(self.N_cells_per_location.expand_as(self.locs_nsc) + 1e-10))
            self.locs_bsg.copy_(torch.log(self.B_groups_per_location.expand_as(self.locs_bsg) + 1e-10))
            nsc = torch.exp(self.locs_nsc)
            zsr_mean = nsc / self.n_groups_tensor
            self.locs_zsr.copy_(torch.log(zsr_mean.expand_as(self.locs_zsr) + 1e-10))
            self.locs_k_r.copy_(torch.log(self.factors_per_groups.expand_as(self.locs_k_r) + 1e-10))
            self.locs_x_fr.copy_(torch.log((1.0/self.n_factors_tensor).expand_as(self.locs_x_fr) + 1e-10))
            zsr = torch.exp(self.locs_zsr); xfr = torch.exp(self.locs_x_fr)
            wsf_mu = zsr @ xfr
            self.locs_wsf.copy_(torch.log(wsf_mu + 1e-10))
            dm = self.detection_mean_hyp_prior_alpha / self.detection_mean_hyp_prior_beta
            self.locs_dmye.copy_(torch.log(dm.expand_as(self.locs_dmye) + 1e-10))
            self.locs_dys.copy_(torch.log(dm.expand_as(self.locs_dys) + 1e-10))
            sgah = self.gene_add_alpha_hyp_prior_alpha / self.gene_add_alpha_hyp_prior_beta
            self.locs_sgah.copy_(torch.log(sgah.expand_as(self.locs_sgah) + 1e-10))
            sgm = self.gene_add_mean_hyp_prior_alpha / self.gene_add_mean_hyp_prior_beta
            self.locs_sgm.copy_(torch.log(sgm.expand_as(self.locs_sgm) + 1e-10))
            self.locs_sgai.copy_(torch.log((1.0/sgah).expand_as(self.locs_sgai) + 1e-10))
            self.locs_sga.copy_(torch.log(sgm.expand_as(self.locs_sga) + 1e-10))
            agph = self.alpha_g_phi_hyp_prior_alpha / self.alpha_g_phi_hyp_prior_beta
            self.locs_agph.copy_(torch.log(agph.expand_as(self.locs_agph) + 1e-10))
            self.locs_agi.copy_(torch.log((1.0/agph).expand_as(self.locs_agi) + 1e-10))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        from scvi import REGISTRY_KEYS
        return (tensor_dict[REGISTRY_KEYS.X_KEY], tensor_dict["ind_x"].long().squeeze(), tensor_dict[REGISTRY_KEYS.BATCH_KEY]), {}

    def _sl(self, loc, scale):
        sp = torch.nn.functional.softplus(scale)
        zu = loc + sp * torch.randn_like(loc)
        return torch.exp(zu), zu, sp, loc

    def forward(self, x_data, idx, batch_index, kl_weight=1.0):
        bi = batch_index.squeeze().long()
        bs = x_data.shape[0]
        # Sample latents
        mgm, mgm_u, mgm_s, mgm_l = self._sl(self.locs_m_g_mean, self.scales_m_g_mean)
        mgai, mgai_u, mgai_s, mgai_l = self._sl(self.locs_m_g_alpha_e_inv, self.scales_m_g_alpha_e_inv)
        mgae = self.ones / mgai.pow(2)
        mg, mg_u, mg_s, mg_l = self._sl(self.locs_m_g, self.scales_m_g)
        kr, kr_u, kr_s, kr_l = self._sl(self.locs_k_r, self.scales_k_r)
        xfr, xfr_u, xfr_s, xfr_l = self._sl(self.locs_x_fr, self.scales_x_fr)
        dmye, dmye_u, dmye_s, dmye_l = self._sl(self.locs_dmye, self.scales_dmye)
        sgah, sgah_u, sgah_s, sgah_l = self._sl(self.locs_sgah, self.scales_sgah)
        sgm, sgm_u, sgm_s, sgm_l = self._sl(self.locs_sgm, self.scales_sgm)
        sgai, sgai_u, sgai_s, sgai_l = self._sl(self.locs_sgai, self.scales_sgai)
        sgae = self.ones / sgai.pow(2)
        sga, sga_u, sga_s, sga_l = self._sl(self.locs_sga, self.scales_sga)
        agph, agph_u, agph_s, agph_l = self._sl(self.locs_agph, self.scales_agph)
        agi, agi_u, agi_s, agi_l = self._sl(self.locs_agi, self.scales_agi)
        nsc, nsc_u, nsc_s, nsc_l = self._sl(self.locs_nsc[idx], self.scales_nsc[idx])
        bsg, bsg_u, bsg_s, bsg_l = self._sl(self.locs_bsg[idx], self.scales_bsg[idx])
        zsr, zsr_u, zsr_s, zsr_l = self._sl(self.locs_zsr[idx], self.scales_zsr[idx])
        wsf, wsf_u, wsf_s, wsf_l = self._sl(self.locs_wsf[idx], self.scales_wsf[idx])
        dys, dys_u, dys_s, dys_l = self._sl(self.locs_dys[idx], self.scales_dys[idx])
        # Compute mu and alpha
        wsf_mu = zsr @ xfr
        mu = ((wsf @ self.cell_state) * mg + sga[bi]) * dys
        alpha = (self.ones / agi.pow(2))[bi]
        # Ensure 2D
        x = x_data
        while x.dim() > 2: x = x.squeeze(1)
        alpha = alpha.squeeze()
        mu = mu.squeeze()
        if alpha.dim() > 2: alpha = alpha.squeeze(1)
        if mu.dim() > 2: mu = mu.squeeze(1)
        if self.dropout_p != 0: x = self.dropout(x)
        # Log-likelihood (chunked)
        def _ll(xc, ac, mc):
            return (torch.lgamma(xc+ac) - torch.lgamma(xc+1) - torch.lgamma(ac)
                    + ac*torch.log(ac+1e-10) + xc*torch.log(mc+1e-10) - (ac+xc)*torch.log(ac+mc+1e-10))
        ll_sum = torch.tensor(0.0, device=x_data.device)
        cs = 5000
        for gs in range(0, self.n_vars, cs):
            ge = min(gs+cs, self.n_vars)
            ll_sum = ll_sum + torch.utils.checkpoint.checkpoint(_ll, x[:,gs:ge], alpha[:,gs:ge], mu[:,gs:ge], use_reentrant=False).sum()
        # KL
        def lq(zu, l, sp): return -0.5*math.log(2*math.pi) - torch.log(sp+1e-10) - 0.5*((zu-l)/sp)**2 - zu
        def lpg(z, a, b): return (a-1)*torch.log(z+1e-10) - b*z + a*torch.log(b+1e-10) - torch.lgamma(a)
        def lpe(z, r): return -r*z + torch.log(r+1e-10)
        sf = self.n_obs / bs
        kl = torch.tensor(0.0, device=x_data.device)
        kl = kl + (lq(mgm_u,mgm_l,mgm_s)-lpg(mgm,self.m_g_mu_mean_var_ratio_hyp*self.m_g_mu_hyp,self.m_g_mu_mean_var_ratio_hyp)).sum()
        kl = kl + (lq(mgai_u,mgai_l,mgai_s)-lpe(mgai,self.m_g_alpha_hyp_mean)).sum()
        kl = kl + (lq(mg_u,mg_l,mg_s)-lpg(mg,mgae,mgae/(mgm+1e-10))).sum()
        kl = kl + sf*(lq(nsc_u,nsc_l,nsc_s)-lpg(nsc,self.N_cells_per_location*self.N_cells_mean_var_ratio,self.N_cells_mean_var_ratio)).sum()
        kl = kl + sf*(lq(bsg_u,bsg_l,bsg_s)-lpg(bsg,self.B_groups_per_location,self.ones)).sum()
        sz = self.ones_1_n_groups * bsg / self.n_groups_tensor
        rz = self.ones_1_n_groups / (nsc / (bsg + 1e-10))
        kl = kl + sf*(lq(zsr_u,zsr_l,zsr_s)-lpg(zsr,sz,rz)).sum()
        kl = kl + (lq(kr_u,kr_l,kr_s)-lpg(kr,self.factors_per_groups,self.ones)).sum()
        c2f = kr / self.n_factors_tensor
        kl = kl + (lq(xfr_u,xfr_l,xfr_s)-lpg(xfr,c2f,kr)).sum()
        kl = kl + sf*(lq(wsf_u,wsf_l,wsf_s)-lpg(wsf,wsf_mu*self.w_sf_mean_var_ratio_tensor,self.w_sf_mean_var_ratio_tensor)).sum()
        kl = kl + (lq(dmye_u,dmye_l,dmye_s)-lpg(dmye,self.ones*self.detection_mean_hyp_prior_alpha,self.ones*self.detection_mean_hyp_prior_beta)).sum()
        da = self.detection_hyp_prior_alpha
        db = da / (dmye[bi] + 1e-10)
        kl = kl + sf*(lq(dys_u,dys_l,dys_s)-lpg(dys,da,db)).sum()
        kl = kl + (lq(sgah_u,sgah_l,sgah_s)-lpg(sgah,self.ones*self.gene_add_alpha_hyp_prior_alpha,self.ones*self.gene_add_alpha_hyp_prior_beta)).sum()
        kl = kl + (lq(sgm_u,sgm_l,sgm_s)-lpg(sgm,self.gene_add_mean_hyp_prior_alpha,self.gene_add_mean_hyp_prior_beta)).sum()
        kl = kl + (lq(sgai_u,sgai_l,sgai_s)-lpe(sgai,sgah)).sum()
        kl = kl + (lq(sga_u,sga_l,sga_s)-lpg(sga,sgae,sgae/(sgm+1e-10))).sum()
        kl = kl + (lq(agph_u,agph_l,agph_s)-lpg(agph,self.ones*self.alpha_g_phi_hyp_prior_alpha,self.ones*self.alpha_g_phi_hyp_prior_beta)).sum()
        kl = kl + (lq(agi_u,agi_l,agi_s)-lpe(agi,agph)).sum()
        # Scale log-likelihood by n_obs/batch_size to match Pyro's plate subsample scaling
        ll_sum = ll_sum * (self.n_obs / bs)
        return -(ll_sum - kl_weight * kl)

    def get_posterior_means(self):
        m = {}
        for name in ["m_g_mean","m_g_alpha_e_inv","m_g","n_s_cells_per_location","b_s_groups_per_location",
                     "z_sr_groups_factors","k_r_factors_per_groups","x_fr_group2fact","w_sf",
                     "detection_mean_y_e","detection_y_s","s_g_gene_add_alpha_hyp","s_g_gene_add_mean",
                     "s_g_gene_add_alpha_e_inv","s_g_gene_add","alpha_g_phi_hyp","alpha_g_inverse"]:
            attr_map = {"m_g_mean":"locs_m_g_mean","m_g_alpha_e_inv":"locs_m_g_alpha_e_inv","m_g":"locs_m_g",
                        "n_s_cells_per_location":"locs_nsc","b_s_groups_per_location":"locs_bsg",
                        "z_sr_groups_factors":"locs_zsr","k_r_factors_per_groups":"locs_k_r",
                        "x_fr_group2fact":"locs_x_fr","w_sf":"locs_wsf","detection_mean_y_e":"locs_dmye",
                        "detection_y_s":"locs_dys","s_g_gene_add_alpha_hyp":"locs_sgah",
                        "s_g_gene_add_mean":"locs_sgm","s_g_gene_add_alpha_e_inv":"locs_sgai",
                        "s_g_gene_add":"locs_sga","alpha_g_phi_hyp":"locs_agph","alpha_g_inverse":"locs_agi"}
            m[name] = torch.exp(getattr(self, attr_map[name])).detach().cpu().numpy()
        return m

    def compute_expected(self, samples, adata_manager, ind_x=None, hide_ambient=False, hide_cell_type=False):
        from scvi import REGISTRY_KEYS
        if ind_x is None: ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else: ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = np.eye(self.n_batch)[obs2sample.flatten().astype(int)][ind_x, :]
        mu = np.ones((1, 1))
        if not hide_cell_type: mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state.T.cpu().numpy()) * samples["m_g"]
        if not hide_ambient: mu = mu + np.dot(obs2sample, samples["s_g_gene_add"])
        mu = mu * samples["detection_y_s"][ind_x, :]
        alpha = np.dot(obs2sample, 1 / np.power(samples["alpha_g_inverse"], 2))
        return {"mu": mu, "alpha": alpha, "ind_x": ind_x}

    @property
    def cell_state_mat(self):
        return self.cell_state.T.cpu().numpy()
