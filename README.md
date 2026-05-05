# Stellar Structure Calculation

Code to compute the ZAMS stellar interior structure of a 2M⊙ star (X = 0.70, Y = 0.28, Z = 0.02) using a shooting method to solve the four coupled differential equations of stellar structure.

Repository Structure

- `constants.py` — physical constants and stellar parameters
- `params.py` — initial guess parameters for the solver
- `zams_eos.py` — equation of state assuming complete ionization
- `zams_opacity.py` — OPAL opacity table loader and interpolator
- `zams_energygeneration.py` — pp-chain and CNO energy generation rates
- `zams_gradients.py` — radiative and adiabatic temperature gradients
- `zams_initialboundaries.py` — inner (load1) and outer (load2) boundary conditions
- `zams_structure_equations.py` — coupled ODEs (derivs)
- `zams_integrate.py` — outward and inward integrators with residual function
- `main.py` — driver script that converges the model and saves the full structure
- `zams_plotting.py` — produces the interior profile figures from the report
- `notebooks/` — analysis and MESA comparison notebooks

Requirements

Place the opacity table in the repo root before running:

```text
opacity_x0.7y0.28z0.02.txt
```
Available from the OPAL website. The module will fall back to this local copy if the absolute path in `zams_opacity.py` does not exist on your machine.

Notebooks

`notebooks/analysis.ipynb` — runs the full solver, builds the stellar structure profiles, and exports the sampled structure table
`notebooks/MESA_comparison.ipynb` — compares the converged model against MESA, produces Figure 4 and Table 1 from the report

Output Files

Running `main.py` or `analysis.ipynb` will produce `full_structure.csv` in the repo root - the complete high-resolution stellar structure from the solver. A sampled version for the report table is saved to `notebooks/stellar_structure_table.csv`.

How to run

```bash
python main.py
```