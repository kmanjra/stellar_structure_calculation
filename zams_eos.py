#### Equation of States calculation assuming complete ionization given pressure, temperature, and composition. Corresponds to Step 2 and Homework 2, assumes complete ionization and sums the contribution from radiative pressure support and ideal gas pressure.


kB = 1.380649e-16         # erg/K 
R  = 8.314462618e7        # g
a_rad   = 7.5657e-15      # erg cm^-3 K^-4
c = 2.99792458e10         # cm/s
mH = 1.6735575e-24        # g
G = 6.67430e-8            # cgs
sigma_sb = 5.670374419e-5 # erg cm^-2 s^-1 K^-4

def compute_density(P, T, X, Y, Z = None):
    # Inputs: Pressure, temperature, hydrogen mass fraction, helium mass fraction, and heavy metal mass fraction
    # Outputs: Density and pressure ratio
    
    if Z is None:
        Z = 1 - X - Y

    # Radiation pressure
    P_rad = (a_rad/3) * T**4

    # Ideal gas pressure
    P_gas = P - P_rad

    mu = 1 / (2.0*X + 0.75*Y + 0.5*Z)

    # Computing the density
    rho = (mu * mH / kB) * (P_gas / T)

    beta = P_gas / P
    
    return rho, beta
