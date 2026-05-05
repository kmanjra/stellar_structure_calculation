### Guess parameters needed to initiate the solvers at and establish the meeting location

from constants import *

from constants import *

P_c_guess = 1.4e17 # Central pressure guess
T_c_guess = 2.0e7  # Central temperature guess
L_star_guess = 16.4 * L_sun # Stellar luminosity guess
R_star_guess = 1.6 * R_sun # Stellar radius guess

m_inner = 1e-8 * M_star  # Inner mass coordinate
m_fit   = 0.5 * M_star  # Meeting point for the inner and outer solvers
m_outer = (1 - 1e-8) * M_star # Outer mass coordinate