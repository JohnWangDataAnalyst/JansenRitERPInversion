"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from ..datatypes import Parameter as par
from ..datatypes import AbstractLoss 
from .cost_TS import CostsTS
from ..functions.arg_type_check import method_arg_type_check

import torch.nn.functional

def band_mask(freqs, f_lo, f_hi):
    m = (freqs>=f_lo) & (freqs<=f_hi)
    return m.float()

def stft_csd(x, nperseg=512, noverlap=256):
    # x: [batch, chan, time]
    # return cross-spectral density S: [batch, freq, chan, chan], freqs
    X = torch.stft(x, n_fft=nperseg, hop_length=nperseg-noverlap,
                   win_length=nperseg, return_complex=True)  # [C,F,T]
    X = X.permute(1,2,0)  # [F,T,C]
    S = torch.einsum('ftc,ftd->fcd', X, X.conj())  # avg over T
    S = S / (X.shape[2] + 1e-8)
    P = torch.einsum('ftc,ftc->fc', X, X.conj()) / (X.shape[2] + 1e-8)    # [F, C]
    # frequency axis
    freqs = torch.linspace(0, 0.5, S.shape[0], device=x.device)  # normalized if fs=1
    return S, P, freqs

def imaginary_coherence(S, P):
    # S: [B,F,C,C] cross-spectrum
    #Pii = S.real.diagonal(dim1=-2, dim2=-1)  # [B,F,C]
    den = torch.sqrt(P.unsqueeze(-1) * P.unsqueeze(-2)) + 1e-12  # [F,C,C]
    coh = (S.abs() ** 2) / (den ** 2)

    return torch.real(S.imag.abs() / (den + 1e-12))  # |ImCoh|  in [0,1]

def spectral_fc_loss(x_sim, x_emp, fs, bands, w_fc=1.0):
    # resample / ensure same fs outside this function if needed
    S_sim, P_sim, freqs = stft_csd(x_sim, nperseg= int(2*fs), noverlap=int(1.5*fs))
    S_emp, P_emp, _     = stft_csd(x_emp,  nperseg= int(2*fs), noverlap=int(1.5*fs))

    C_sim = imaginary_coherence(S_sim, P_sim)
    C_emp = imaginary_coherence(S_emp, P_emp)
    #print(C_sim)
    loss = 0.0
    for (lo, hi, w) in bands:  # e.g., [(8,13,1.0),(13,30,0.7)]
        m = band_mask(freqs*fs, lo, hi)[:,None,None]  # [F,1,1]

        # average within band
        Cb_sim = (C_sim * m).sum(dim=0) / (m.sum(dim=0)+1e-8)
        Cb_emp = (C_emp * m).sum(dim=0) / (m.sum(dim=0)+1e-8)
        #print(Cb_sim.shape)
        # Frobenius loss on off-diagonal (optional)
        off = 1 - torch.eye(Cb_sim.shape[-1], device=x_sim.device)
        #print(Cb_sim)
        loss = loss + w * torch.torch.nn.functional.mse_loss(Cb_sim*off, Cb_emp*off)
    return w_fc * loss

class CostsJR(AbstractLoss):
    def __init__(self, model):
        
        self.simKey = model.output_names[0]
        self.mainLoss = CostsTS(simKey = self.simKey, model=model)
        self.batch_size = model.TRs_per_window
        
        
    def loss(self, simData: dict, empData: torch.Tensor):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
       
        # define some constants
        w_cost = 10

    

        loss_main = self.mainLoss.main_loss(sim, emp)

        loss_EI = spectral_fc_loss(sim[self.simKey], emp, self.batch_size//2, [(8,30,0.5)])
        loss_prior = self.mainLoss.prior_loss()

        
        loss = 0.1 * w_cost * loss_main + 1 * sum(loss_prior) + 1 * loss_EI
        return loss, loss_main
