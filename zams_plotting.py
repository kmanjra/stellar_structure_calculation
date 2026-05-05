#### Plotting script that produces the figure presented within the report. These are the stellar interior profiles, but not the MESA comparison. MESA results are within the notebooks folder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from constants import *

# Read the full stellar structure solution produced by main.py
FULL_STRUCTURE_PATH = 'full_structure.csv'

df_full = pd.read_csv(FULL_STRUCTURE_PATH)

m_all= df_full['m'].values * M_sun
r_all  = df_full['r'].values * R_sun
rho_all = df_full['rho'].values
T_all = df_full['T'].values
P_all = df_full['P'].values
l_all  = df_full['l'].values * L_sun
eps_all  = df_full['eps'].values
eps_pp_arr= df_full['eps_pp'].values
eps_cno_arr = df_full['eps_cno'].values
grad_all = df_full['nabla'].values
grad_ad_all = df_full['nabla_ad'].values
transport_all = df_full['transport'].values

rho_all= np.array(rho_all)
eps_all = np.array(eps_all)
grad_all= np.array(grad_all)
grad_ad_all = np.array(grad_ad_all)

n_out = np.sum(df_full['branch'].values == 'outward')
q = m_all / M_star

# Color scheme
c_out = 'steelblue'
c_in  = 'red'  
lw = 1.8

def plot_split(ax, y, log=False):
    plot_fn = ax.semilogy if log else ax.plot
    ax.plot(q[:n_out],  y[:n_out],  color=c_out, lw=lw, label='Outward')
    plot_fn(q[n_out-1:], y[n_out-1:], color=c_in,  lw=lw, label='Inward')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
plt.subplots_adjust(wspace=0.35)

# Plot for the temperature profile
plot_split(axes[0], T_all)
axes[0].set_xlabel(r"$M/M_\ast$")
axes[0].set_ylabel(r"$T\ ({\rm K})$")
axes[0].tick_params(direction='in', top=True, right=True)
axes[0].minorticks_on()
axes[0].legend(fontsize=9)

# Plot for the pressure profile
plot_split(axes[1], P_all, log=True)
axes[1].set_xlabel(r"$M/M_\ast$")
axes[1].set_ylabel(r"$P\ ({\rm dyne\ cm^{-2}})$")
axes[1].tick_params(direction='in', top=True, right=True)
axes[1].minorticks_on()

# Plot for the density profile
plot_split(axes[2], rho_all, log=True)
axes[2].set_xlabel(r"$M/M_\ast$")
axes[2].set_ylabel(r"$\rho\ ({\rm g\ cm^{-3}})$")
axes[2].tick_params(direction='in', top=True, right=True)
axes[2].minorticks_on()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, figsize=(5, 8))

# Plot for the luminosity profile as a function of enclosed mass
axes[0].plot(q[:n_out],   np.array(l_all[:n_out]) / L_sun,   color=c_out, lw=lw, label='Outward')
axes[0].plot(q[n_out-1:], np.array(l_all[n_out-1:]) / L_sun, color=c_in,  lw=lw, label='Inward')
axes[0].set_xlabel(r"$M/M_\ast$")
axes[0].set_ylabel(r"$l/L_\odot$")
axes[0].tick_params(direction='in', top=True, right=True)
axes[0].minorticks_on()
axes[0].legend(fontsize=9)

# Plot for the total energy generation rate, and the contribution of both pp chain and cno cycle
axes[1].semilogy(q[:n_out], eps_all[:n_out],   color=c_out, lw=lw, label=r'$\epsilon_\mathrm{total}$ (out)')
axes[1].semilogy(q[n_out-1:], eps_all[n_out-1:], color=c_in, lw=lw, label=r'$\epsilon_\mathrm{total}$ (in)')
axes[1].semilogy(q, eps_cno_arr, color='black', lw=1, linestyle='--', label=r'$\epsilon_\mathrm{CNO}$')
axes[1].semilogy(q, eps_pp_arr, color='black', lw=1, linestyle='-.', label=r'$\epsilon_\mathrm{PP}$')
axes[1].set_xlabel(r"$M/M_\ast$")
axes[1].set_ylabel(r"$\epsilon\ ({\rm erg\ g^{-1}\ s^{-1}})$")
axes[1].set_ylim(1e-2, 1e4)
axes[1].tick_params(direction='in', top=True, right=True)
axes[1].minorticks_on()
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

q = np.array(m_all) / np.array(m_all).max()
nabla = np.array(grad_all)
nabla_ad = np.array(grad_ad_all)
transport_arr = np.array(transport_all)

conv_mask = transport_arr == 'convective'
rad_mask  = transport_arr == 'radiative'

def get_regions_q(mask, q):
    regions = []
    in_region = False
    for i, val in enumerate(mask):
        if val and not in_region:
            start = q[i-1] + 0.5*(q[i]-q[i-1]) if i > 0 else q[i]
            in_region = True
        elif not val and in_region:
            end = q[i-1] + 0.5*(q[i]-q[i-1])
            regions.append((start, end))
            in_region = False
    if in_region:
        regions.append((start, 1.0))
    return regions


# Plot illustrating the radiative and adiabatic gradients indicating the radiative and convective layers within the interior
fig, ax = plt.subplots(figsize=(5, 6))

for q0, q1 in get_regions_q(rad_mask, q):
    ax.axhspan(q0, q1+0.005, color='steelblue', alpha=0.12, lw=0, zorder=0)

for q0, q1 in get_regions_q(conv_mask, q):
    ax.axhspan(q0, q1, color='tomato', alpha=0.25, lw=0, zorder=0)

transitions = np.where(np.diff(conv_mask.astype(int)) != 0)[0]
for t in transitions:
    ax.axhline(q[t] + 0.5*(q[t+1]-q[t]), color='gray', lw=0.8, zorder=1)

# Surface convective shell marker
ax.axhline(0.9984, color='tomato', lw=1.5, alpha=0.6, zorder=1)
ax.annotate('surface conv. zone', xy=(0.02, 0.965),xycoords=('axes fraction', 'data'),va='bottom', ha='left', fontsize=7, color='tomato')

ax.plot(nabla,    q, color='steelblue', lw=1.5, label=r'$\nabla$',        zorder=2)
ax.plot(nabla_ad, q, color='black', linestyle='--', lw=1.5,label=r'$\nabla_{\rm ad}$', zorder=2)

ax.tick_params(direction='in', top=True, right=True, pad=6)
ax.set_xlabel('Temperature Gradient', labelpad=8, fontsize = 11)
ax.set_ylabel(r'$M/M_\star$', labelpad=8, fontsize = 11)
ax.set_title('Energy Transport Regime', fontsize = 12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 0.45)
ax.tick_params(direction='in', top=True, right=True)
ax.minorticks_on()

handles = [
    plt.Line2D([], [], color='steelblue', lw=1.5, label=r'$\nabla$'),
    plt.Line2D([], [], color='black', linestyle='--', lw=1.5, label=r'$\nabla_{\rm ad}$'),
    Patch(facecolor='steelblue', alpha=0.15, label='Radiative'),
    Patch(facecolor='tomato',    alpha=0.25, label='Convective'),
]
ax.legend(handles=handles, loc='lower left', fontsize = 12)
plt.tight_layout()
plt.show()