#### Establish the inner and outer boundary conditions using the series expansion near the center and outer boundary. Corresponds to Step 8 of the instructions and the solution to Homework 5

import numpy as np
from scipy.optimize import brentq
from zams_eos import G, sigma_sb, a_rad, c, compute_density
from zams_opacity import get_opacity
from zams_energygeneration import nuclear_energy_rates
from zams_gradients import nabla_rad, nabla_ad

def load1(M, M_r, P_c, T_c, X, Y, Z):
    # Compute central density and gas-pressure fraction from the guessed central P and T
    # Inputs: total mass, enclosed mass, central pressure, central temperature, composition fractions
    # Outputs: initial luminosity, pressure, radius, temperature for outward integration
    
    rho_c, beta_c = compute_density(P_c, T_c, X, Y, Z)
    # Compute the central nuclear energy generation rate
    eps_pp_c, eps_cno_c, epsilon_c = nuclear_energy_rates(rho_c, T_c, X, Y)

    r = (3 * M_r / (4 * np.pi * rho_c))**(1/3)

    # Use l ≈ epsilon_c * M_r near the center
    l = epsilon_c * M_r

    # Use the central pressure expansion to move slightly away from P_c
    P = P_c - (3 * G / (8 * np.pi)) * (4 * np.pi * rho_c / 3)**(4/3) * M_r**(2/3)

    # Evaluate central opacity for the transport-gradient calculation
    kappa_c = get_opacity(T_c, rho_c)

    # Compare radiative and adiabatic gradients to choose the temperature expansion
    grad_rad_c = nabla_rad(l, P_c, T_c, M_r, kappa_c)
    grad_ad_c = nabla_ad(beta_c)

    if grad_rad_c <= grad_ad_c:
        T = (T_c**4 - (1 / (2 * a_rad * c)) * (3 / (4 * np.pi))**(2/3)* kappa_c * epsilon_c * rho_c**(4/3) * M_r**(2/3))**(1/4)
    else:
        lnT = np.log(T_c) - (np.pi / 6)**(1/3) * G * grad_ad_c * rho_c**(4/3) / P_c * M_r**(2/3)
        T = np.exp(lnT)
    
    # Return the initial interior values for outward integration
    return [l, P, r, T]




def load2(M, R, L, X, Y, Z):
    # Set the surface temperature using the luminosity-radius relation
    # Inputs: total mass, radius, luminosity, composition fractions
    # Outputs: surface luminosity, pressure, radius, temperature for inward integration
    
    T = (L / (4 * np.pi * R**2 * sigma_sb))**(1/4)
    # Compute surface gravity
    g = G * M / R**2

    def pressure_residual(logP):
        # Inputs: log10 of pressure
        # Outputs: difference between trial pressure and photospheric pressure
        
        P = 10**logP
        
        # Compute density and opacity at the trial surface pressure
        rho, beta = compute_density(P, T, X, Y, Z)
        kappa = get_opacity(T, rho)
        
        # Reject invalid opacity values during root finding
        if kappa is None or np.isnan(kappa) or kappa <= 0:
            return np.nan
        
        # Photospheric boundary condition: P ≈ (2/3) g / kappa
        P_target = (2/3) * g / kappa

        # Root occurs when trial pressure matches the photospheric target pressure
        return np.log10(P) - np.log10(P_target)
    
    # Find the pressure satisfying the photospheric boundary condition, helped to fix the numerical solution, ran into errors otherwise
    logP_sol = brentq(pressure_residual, 3.0, 4.0, xtol=1e-10)
    P = 10**logP_sol
    
    # Return the outer boundary values for inward integration
    return [L, P, R, T]
