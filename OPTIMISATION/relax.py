from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# --- SETTINGS ---
input_cif = "cubic-LLZO.cif"
output_cif = "relaxed.cif"
device = "cuda"  # or "cuda"
fmax = 1e-2     # relaxation force threshold

# --- LOAD CIF ---
atoms = read(input_cif)

# --- SET UP ORB CALCULATOR ---
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)
atoms.calc = calc

# --- RELAX CELL + POSITIONS ---
ucf = UnitCellFilter(atoms)  # allows both cell & positions to relax
dyn = BFGS(ucf)
dyn.run(fmax=fmax)

# --- SAVE RESULT ---
write(output_cif, atoms)

print(f"Relaxation complete. Relaxed structure written to {output_cif}")

