#### Script producing the coupled differential equations that will be iteratively used within the solver. Corresponds to Step 9 of the instructions and Homework 5

import numpy as np
from zams_eos import G, compute_density
from zams_opacity import get_opacity
from zams_energygeneration import nuclear_energy_rates
from zams_gradients import actual_nabla

def derivs(m, y, X, Y, Z):
    # Inputs: enclosed mass, state vector [luminosity, pressure, radius, temperature], composition fractions
    # Outputs: derivatives [dl/dm, dP/dm, dr/dm, dT/dm]
    l, P, r, T = y

    # Compute the density and opacity
    rho, beta = compute_density(P, T, X, Y, Z)
    kappa = get_opacity(T, rho)

    # Compute the energy generation rate and transport values
    eps_pp, eps_cno, eps = nuclear_energy_rates(rho, T, X, Y)
    grad, grad_rad, grad_ad, transport = actual_nabla(l, P, T, m, kappa, beta)

    # Compute the coupled differential equations below
    dl_dm = eps
    dP_dm = -G * m / (4.0 * np.pi * r**4)
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)
    dT_dm = -(G * m * T / (4.0 * np.pi * r**4 * P)) * grad

    return np.array([dl_dm, dP_dm, dr_dm, dT_dm])
