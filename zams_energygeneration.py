#### Computing the total energy generation rate as a combination of CNO and pp-chain contributions. Corresponds to Homework 4 and Step 5 of instructions.

import numpy as np

def zeta_mixture(X, Y):
    # Effective charge weighting from composition
    return 2.0*X + 1.5*Y

def nuclear_energy_rates(rho, T, X, Y, psi=1.0, use_screening=True):
    # Inputs: density (rho), temperature (T), hydrogen mass fraction (X), and helium mass fraction (Y). 
    # Outputs: energy generation rate from pp-chain, energy generation rate from CNO cycle, and total energy generation rate

    Z = 1.0 - X - Y    # Mass fraction of heavies

    T7 = T / 1e7
    T9 = T / 1e9

    if use_screening:
        zeta = zeta_mixture(X, Y)
        ED_over_kT = 5.92e-3 * np.sqrt((zeta * rho) / (T7**3))
        f11 = np.exp(ED_over_kT)
    else:
        f11 = 1.0

    ### pp-chain energy generation ###
    
    g11 = 1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4

    # pp-chain rate evaluation
    eps_pp = (2.57e4 * f11 * g11 *rho * X**2 *T9**(-2/3) *np.exp(-3.381 * T9**(-1/3)))

    ### CNO cycle energy generation ###

    # Temperature correction factor
    g14_1 = 1 - 2.00*T9 + 3.41*T9**2 - 2.43*T9**3
    
    # Assumption that heavies abundance is CNO abundance
    X_CNO = Z 
    
    # CNO rate evaluation
    eps_cno = (8.24e25 * g14_1 * X_CNO * X * rho *T9**(-2/3) *np.exp(-15.231 * T9**(-1/3) - (T9/0.8)**2))

    eps_total = eps_pp + eps_cno
    
    return eps_pp, eps_cno, eps_total
