#### Initialize the relevant starting parameters. Corresponds to Step 1 of instructions

G = 6.67430e-8   # cgs
M_sun = 1.989e33 # g
R_sun = 6.957e10 # cm
L_sun = 3.828e33 # erg/s

M_star = 2.0 * M_sun 
X = 0.70  # Hydrogen mass fraction
Y = 0.28   # Helium mass fraction
Z = 1 - X - Y  # Heavies mass fraction

q_fit = 0.5   # Intermediate mass regime where solutions meet
q_inner = 1e-8   # Part of mass near the core
q_outer = 1.0 - 1e-8  # Part of the mass near the outer envelope

m_fit = q_fit * M_star
m_inner = q_inner * M_star
m_outer = q_outer * M_star