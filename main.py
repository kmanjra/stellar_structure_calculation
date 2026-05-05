#### Main python script that demonstrates the usage of the attached scripts to solve the ZAMS stellar interior

import pandas as pd
import numpy as np
from scipy.optimize import root

from constants import *
from params import *
from zams_integrate import integrate_outward, integrate_inward, residuals
from zams_eos import compute_density
from zams_opacity import get_opacity
from zams_energygeneration import nuclear_energy_rates
from zams_gradients import actual_nabla

FULL_STRUCTURE_PATH = 'full_structure.csv'


def solve_zams_model():
    # Outputs: The full dataframe containing the final results for the ZAMS solver
    
    guess0 = np.array([np.log10(1.4e17),np.log10(2.0e7), np.log10(16.4 * L_sun),np.log10(1.6 * R_sun),])

    sol_root = root(residuals, guess0, method="hybr", tol=1e-10,options={'factor': 0.001})
    print(sol_root.success, sol_root.message)
    print(sol_root.x)

    Pc_sol = 10**sol_root.x[0]
    Tc_sol = 10**sol_root.x[1]
    Ls_sol = 10**sol_root.x[2]
    Rs_sol = 10**sol_root.x[3]

    print("Pc =", Pc_sol)
    print("Tc =", Tc_sol)
    print("L* =", Ls_sol)
    print("R* =", Rs_sol)

    sol_out = integrate_outward(Pc_sol, Tc_sol, Ls_sol, Rs_sol)
    sol_in  = integrate_inward(Pc_sol, Tc_sol, Ls_sol, Rs_sol)

    m_out = sol_out.t
    m_in  = sol_in.t[::-1]

    y_out = sol_out.y
    y_in  = sol_in.y[:, ::-1]

    m_all = np.concatenate([m_out, m_in[1:]])
    y_all = np.hstack([y_out, y_in[:, 1:]])

    l_all = y_all[0]
    P_all = y_all[1]
    r_all = y_all[2]
    T_all = y_all[3]


    ## Create a series of arrays containing the final results of the solver
    rho_all = []
    eps_all = []
    eps_pp_all = []
    eps_cno_all = []
    kappa_all = []
    grad_ad_all = []
    grad_all = []
    transport_all = []

    for m, l, P, r, T in zip(m_all, l_all, P_all, r_all, T_all):
        rho, beta = compute_density(P, T, X, Y, Z)
        kappa = get_opacity(T, rho)
        eps_pp, eps_cno, eps = nuclear_energy_rates(rho, T, X, Y)
        grad, grad_rad, grad_ad, transport = actual_nabla(l, P, T, m, kappa, beta)

        rho_all.append(rho)
        eps_all.append(eps)
        eps_pp_all.append(eps_pp)
        eps_cno_all.append(eps_cno)
        kappa_all.append(kappa)
        grad_ad_all.append(grad_ad)
        grad_all.append(grad)
        transport_all.append(transport)

    print("Residuals at fitting point:", sol_root.fun)
    print("Converged:", sol_root.success)

    print("Number of function evaluations:", sol_root.nfev)
    print("Converged:", sol_root.success)
    print("Message:", sol_root.message)

    n_out = len(sol_out.t)
    branch_all = np.array(['outward'] * n_out + ['inward'] * (len(m_all) - n_out), dtype=object)

    # Build full DataFrame for exporting the final results
    df_full = pd.DataFrame({
        'm':np.array(m_all) / M_sun,
        'r': np.array(r_all) / R_sun,
        'rho':rho_all,
        'T':T_all,
        'P': P_all,
        'l': np.array(l_all) / L_sun,
        'eps': eps_all,
        'eps_pp': eps_pp_all,
        'eps_cno': eps_cno_all,
        'kappa': kappa_all,
        'nabla_ad': grad_ad_all,
        'nabla': grad_all,
        'transport': transport_all,
        'branch': branch_all,
    })

    return df_full


if __name__ == "__main__":
    df_full = solve_zams_model()
    df_full.to_csv(FULL_STRUCTURE_PATH, index=False)
    print(f"Saved full stellar structure to {FULL_STRUCTURE_PATH}")
