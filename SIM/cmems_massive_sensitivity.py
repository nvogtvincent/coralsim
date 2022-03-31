#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from CMEMS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from coralsim import Experiment
from scipy.stats.stats import pearsonr

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
# Using parameters for A. millepora from Connolly et al. (2010)
parameters = {# Competency parameters
              'a': 1/timedelta(days=5.6).total_seconds(),  # Rate of competency acquisition (1/s)
              'b': 1/timedelta(days=20).total_seconds(),   # Rate of competency loss (1/s)
              'tc': timedelta(days=3.2).total_seconds(),   # Minimum competency period (s)

              # Settling parameters
              'μs': 1/timedelta(days=1).total_seconds(),   # Settling rate (1/s)

              # Mortality parameters
              'σ': 0.,                                     # GW shape parameter 1
              'ν': 0.57,                                   # GW shape parameter 2
              'λ': 1/timedelta(days=23).total_seconds(),}  # GW mortality rate parameter

min_thresh_pct = 1

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/CMEMS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/'

##############################################################################
# CARRY OUT SENSITIVITY TESTS                                                #
##############################################################################

sey = Experiment()
sey.config(dirs, preset='CMEMS', releases_per_month=1) ## Add ability to read from netcdf
sey.generate_dict()

num_tests = 13
matrices = []
subsets = 2**np.arange(num_tests)
particle_num = (2**16)/subsets

for subset in subsets:
    matrices.append(sey.generate_matrix(fh='CMEMS_massive_test*',
                                   parameters=parameters,
                                   filters={'eez': [690]},
                                   subset=subset)[:2])

n_months = np.shape(matrices[0][0])[-1]

FUV_logp = np.zeros((n_months+1, len(particle_num))) # FUV for log probability matrix
FUV_logf = np.zeros_like(FUV_logp) # FUV for log flux matrix

MAE_logp = np.zeros_like(FUV_logp) # MAE for log probability matrix
MAE_logf = np.zeros_like(FUV_logp) # MAE for log flux matrix

missed_connections = np.zeros_like(FUV_logp) # Fraction of missed connections relative to 'truth'

# An issue with comparison of these matrices is dealing with zeros, since
# log(0) is undefined, so we need some way of comparing zeros in subsets against
# nonzeros in the 'truth'. We are going to define a minimum detection threshold
# (unfortunately somewhat arbitrary). All nonzero values below this minimum
# detection threshold will be set to that value. All values below this threshold
# (nonzero AND zero) in subsets that are nonzero in the 'truth' will be set to
# that value. This minimum detection threshold should physically represent (i)
# the smallest meaningful probability of a larval transport and (ii) the smallest
# meaningful larval flux. Both are extremely difficult to physically define since
# we have no good generalisable estimates for reef fecundity. So we will set this
# as the 1st percentile for nonzero log values in the 'truth'.

logp_truth_matrix = matrices[0][0]
logp_truth = np.log10(np.ma.masked_where(logp_truth_matrix == 0, logp_truth_matrix))
logpm_truth = np.log10(np.ma.masked_where(np.mean(logp_truth_matrix, axis=2) == 0, np.mean(logp_truth_matrix, axis=2)))
logp_mask = logp_truth.mask
logpm_mask = logpm_truth.mask

logf_truth_matrix = matrices[0][1]
logf_truth = np.log10(np.ma.masked_where(logf_truth_matrix == 0, logf_truth_matrix))
logfm_truth = np.log10(np.ma.masked_where(np.mean(logf_truth_matrix, axis=2) == 0, np.mean(logf_truth_matrix, axis=2)))
logf_mask = logf_truth.mask
logfm_mask = logfm_truth.mask

logp_min = np.percentile(logp_truth.compressed(), min_thresh_pct)
logf_min = np.percentile(logf_truth.compressed(), min_thresh_pct)
logpm_min = np.percentile(logpm_truth.compressed(), min_thresh_pct)
logfm_min = np.percentile(logfm_truth.compressed(), min_thresh_pct)

logp_truth[logp_truth < logp_min] = logp_min
logf_truth[logf_truth < logf_min] = logf_min
logpm_truth[logpm_truth < logpm_min] = logpm_min
logfm_truth[logfm_truth < logfm_min] = logfm_min

# Now loop through the subsets
for i in range(len(subsets)):
    for ts in range(n_months):
        logp_subset = np.ma.masked_array(matrices[i][0][:, :, ts], mask=logp_mask[:, :, ts]).compressed()
        frac_zeros = np.sum((logp_subset == 0))/len(logp_subset)
        logp_subset[logp_subset < 10**logp_min] = 10**logp_min
        logp_subset = np.log10(logp_subset)

        logf_subset = np.ma.masked_array(matrices[i][1][:, :, ts], mask=logf_mask[:, :, ts]).compressed()
        logf_subset[logf_subset < 10**logf_min] = 10**logf_min
        logf_subset = np.log10(logf_subset)

        FUV_logp[ts, i] = 1 - pearsonr(logp_truth[:, :, ts].compressed(), logp_subset)[0]**2
        FUV_logf[ts, i] = 1 - pearsonr(logf_truth[:, :, ts].compressed(), logf_subset)[0]**2

        MAE_logp[ts, i] = np.mean(np.abs(logp_truth[:, :, ts].compressed() - logp_subset))
        MAE_logf[ts, i] = np.mean(np.abs(logf_truth[:, :, ts].compressed() - logf_subset))

        missed_connections[ts, i] = frac_zeros

    logpm_subset = np.ma.masked_array(np.mean(matrices[i][0], axis=2), mask=logpm_mask).compressed()
    frac_zeros = np.sum((logpm_subset == 0))/len(logpm_subset)
    logpm_subset[logpm_subset < 10**logpm_min] = 10**logpm_min
    logpm_subset = np.log10(logpm_subset)

    logfm_subset = np.ma.masked_array(np.mean(matrices[i][1], axis=2), mask=logfm_mask).compressed()
    logfm_subset[logfm_subset < 10**logfm_min] = 10**logfm_min
    logfm_subset = np.log10(logfm_subset)

    FUV_logp[-1, i] = 1 - pearsonr(logpm_truth.compressed(), logpm_subset)[0]**2
    FUV_logf[-1, i] = 1 - pearsonr(logfm_truth.compressed(), logfm_subset)[0]**2

    MAE_logp[-1, i] = np.mean(np.abs(logpm_truth.compressed() - logpm_subset))
    MAE_logf[-1, i] = np.mean(np.abs(logfm_truth.compressed() - logfm_subset))

    missed_connections[-1, i] = frac_zeros

# Plot results
f, ax = plt.subplots(3, 1, figsize=(10,20))
f.tight_layout()
offset = 1.05

# Plot MAE
ax[0].errorbar(particle_num[1:], np.mean(MAE_logp[:-1, 1:], axis=0),
               yerr=np.std(MAE_logp[:-1, 1:], axis=0),
               linewidth=1, color='darkred', capsize=2, fmt='none')
ax[0].scatter(particle_num[1:], np.mean(MAE_logp[:-1, 1:], axis=0),
              s=25, c='darkred', marker='s', label='Connection probability')
ax[0].scatter(particle_num[0], np.mean(MAE_logp[:-1, 0]), s=25, c='darkred', marker='s')
ax[0].scatter(particle_num[1:], MAE_logp[-1, 1:], s=25, marker='o',
              facecolors='none', edgecolors='darkred', label='Time-mean connection probability')

ax[0].errorbar(particle_num[1:]*offset, np.mean(MAE_logf[:-1, 1:], axis=0),
               yerr=np.std(MAE_logf[:-1, 1:], axis=0),
               linewidth=1, color='steelblue', capsize=2, fmt='none')
ax[0].scatter(particle_num[1:]*offset, np.mean(MAE_logf[:-1, 1:], axis=0),
              s=25, c='steelblue', marker='s', label='Connection flux')
ax[0].scatter(particle_num[0]*offset, np.mean(MAE_logf[:-1, 0]), s=25, c='steelblue', marker='s')
ax[0].scatter(particle_num[1:], MAE_logf[-1, 1:], s=25, marker='o',
              facecolors='none', edgecolors='steelblue', label='Time-mean connection flux')

ax[0].plot([particle_num[-1]/1.5, particle_num[0]*1.5], [np.log10(2), np.log10(2)],
           'k--', linewidth=1)

ax[0].set_xscale('log')
ax[0].set_ylabel('Mean absolute log error', fontsize=12)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xlim([particle_num[-1]/1.5, particle_num[0]*1.5])
ax[0].set_ylim([0, 3])
ax[0].legend(frameon=False, fontsize=12)

# Plot FUV
ax[1].errorbar(particle_num[1:], np.mean(FUV_logp[:-1, 1:], axis=0),
               yerr=np.std(MAE_logp[:-1, 1:], axis=0),
               linewidth=1, color='darkred', capsize=2, fmt='none')
ax[1].scatter(particle_num[1:], np.mean(FUV_logp[:-1, 1:], axis=0),
              s=25, c='darkred', marker='s')
ax[1].scatter(particle_num[0], np.mean(FUV_logp[:-1, 0]), s=25, c='darkred', marker='s')
ax[1].scatter(particle_num[1:], FUV_logp[-1, 1:], s=25, marker='o',
              facecolors='none', edgecolors='darkred')

ax[1].errorbar(particle_num[1:]*offset, np.mean(FUV_logf[:-1, 1:], axis=0),
               yerr=np.std(MAE_logf[:-1, 1:], axis=0),
               linewidth=1, color='steelblue', capsize=2, fmt='none')
ax[1].scatter(particle_num[1:]*offset, np.mean(FUV_logf[:-1, 1:], axis=0),
              s=25, c='steelblue', marker='s')
ax[1].scatter(particle_num[0]*offset, np.mean(FUV_logf[:-1, 0]), s=25, c='steelblue', marker='s')
ax[1].scatter(particle_num[1:], FUV_logf[-1, 1:], s=25, marker='o',
              facecolors='none', edgecolors='steelblue')

ax[1].plot([particle_num[-1]/1.5, particle_num[0]*1.5], [0.05, 0.05],
           'k--', linewidth=1)

ax[1].set_xscale('log')
ax[1].set_ylabel('FUV', fontsize=12)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_ylim([0, 1])
ax[1].set_xlim([particle_num[-1]/1.5, particle_num[0]*1.5])

# Plot missed connections
ax[2].errorbar(particle_num[1:], np.mean(missed_connections[:-1, 1:], axis=0),
               yerr=np.std(missed_connections[:-1, 1:], axis=0),
               linewidth=1, color='k', capsize=2, fmt='none')
ax[2].scatter(particle_num[1:], np.mean(missed_connections[:-1, 1:], axis=0),
              s=25, c='k', marker='s')
ax[2].scatter(particle_num[0], np.mean(missed_connections[:-1, 0]), s=25, c='k', marker='s')
ax[2].scatter(particle_num[1:], missed_connections[-1, 1:], s=25, marker='o',
              facecolors='none', edgecolors='k')

ax[2].plot([particle_num[-1]/1.5, particle_num[0]*1.5], [0.05, 0.05],
           'k--', linewidth=1)

ax[2].set_xscale('log')
ax[2].set_ylabel('Proportion of connections missed', fontsize=12)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[2].set_ylim([0, 1])
ax[2].set_xlim([particle_num[-1]/1.5, particle_num[0]*1.5])

ax[2].set_xlabel('Particles released per 1/12° reef cell', fontsize=12)

plt.savefig(dirs['fig'] + 'CMEMS_massive_sensitivity.png', dpi=300, bbox_inches='tight')