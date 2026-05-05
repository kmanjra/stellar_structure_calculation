#### Helper functions and energy transport expression to compute the gradients

import numpy as np
from zams_eos import a_rad, c, G

def nabla_rad(l, P, T, m, kappa):
    # Computing the radiation gradient
    # Inputs: luminosity, pressure, temperature, enclosed mass, opacity
    # Outputs: radiative temperature gradient
    return (3.0 * kappa * l * P) / (16.0 * np.pi * a_rad * c * G * m * T**4)


def nabla_ad(beta):
    # Computing the adiabatic gradient
    # Inputs: gas pressure fraction (P_gas / P_total)
    # Outputs: adiabatic temperature gradient
    return (8.0 - 6.0*beta) / (32.0 - 24.0*beta - 3.0*beta**2)


def actual_nabla(l, P, T, m, kappa, beta):
    # Computing both gradients, then asserting radiative or convective energy transport
    # Inputs: luminosity, pressure, temperature, enclosed mass, opacity, gas pressure fraction
    # Outputs: actual gradient, radiative gradient, adiabatic gradient, transport type
    
    grad_rad = nabla_rad(l, P, T, m, kappa)
    grad_ad = nabla_ad(beta)
    grad = min(grad_rad, grad_ad)

    # Asserting which is the optimal method of energy transport
    if grad_rad < grad_ad:
        transport = "radiative"
    else:
        transport = "convective"

    return grad, grad_rad, grad_ad, transport
