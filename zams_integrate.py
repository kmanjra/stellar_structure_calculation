#### Solvers initiating the inward and outward solver until the meeting point. Corresponds Step 6 of the instructions

import numpy as np
from scipy.integrate import solve_ivp
from constants import M_star, X, Y, Z
from zams_initialboundaries import load1, load2
from zams_structure_equations import derivs

m_inner = 1e-8 * M_star              # Inner boundary starting location of integrator. Due to singularity at r = 0, small shift is applied
m_fit   = 0.5 * M_star               # Intermediate meeting place directly half way between the boundaries
m_outer = (1 - 1e-8) * M_star        # Outer starting location of integrator, takes the location at a small distance from the exact outer surface


def integrate_outward(Pc, Tc, Ls, Rs):
    # Outward integrator starting at center of the star
    # Input: Central pressure and temperature, with the stellar luminosity and radius
    # Output: Outward solver's solution at the meeting point

    # Initiate the inner boundary values for the guess provided
    y0 = load1(M_star, m_inner, Pc, Tc, X, Y, Z)

    # Solve the profiles simultaneously give an inner boundary and an ending location
    sol = solve_ivp(lambda m, y: derivs(m, y, X, Y, Z),[m_inner, m_fit],y0,method="RK45",rtol=1e-8,atol=1e-10)
    return sol

def integrate_inward(Pc, Tc, Ls, Rs):
    # Inward integrator starting near the surface of the star
    # Input: Central pressure and temperature, with the stellar luminosity and radius
    # Output: Inward solver's solution at the meeting point

    # Initiate the outer boundary values for the guess provided
    y0 = load2(M_star, Rs, Ls, X, Y, Z)

    # Solve coupled differential equations simultaneously from the outer boundary to meeting location
    sol = solve_ivp(lambda m, y: derivs(m, y, X, Y, Z),[m_outer, m_fit],y0,method="RK45",rtol=1e-9,atol=1e-10)
    return sol

def residuals(log_guess):
    # Compute the residuals between the two solvers at the meeting location
    # Input: Inputs the guess parameters for the central pressure and temperature, with the stellar luminosity and radius in an array
    # Output: Computes the residuals between the inward and outward solvers

    # Initiate the guess from the solver
    logPc, logTc, logLs, logRs = log_guess

    Pc = 10**logPc
    Tc = 10**logTc
    Ls = 10**logLs
    Rs = 10**logRs

    # Solve the inner and outer integrators to find their ending location at the meeting location
    sol_out = integrate_outward(Pc, Tc, Ls, Rs)
    sol_in  = integrate_inward(Pc, Tc, Ls, Rs)

    # Get the ending locations for each solver
    y_out = sol_out.y[:, -1]
    y_in  = sol_in.y[:, -1]
    
    # Compute relative mismatches to help scaling
    return np.array([(y_out[0] - y_in[0]) / (0.5 * (abs(y_out[0]) + abs(y_in[0])) + 1e-30),(y_out[1] - y_in[1]) / (0.5 * (abs(y_out[1]) + abs(y_in[1])) + 1e-30),(y_out[2] - y_in[2]) / (0.5 * (abs(y_out[2]) + abs(y_in[2])) + 1e-30),(y_out[3] - y_in[3]) / (0.5 * (abs(y_out[3]) + abs(y_in[3])) + 1e-30),])
