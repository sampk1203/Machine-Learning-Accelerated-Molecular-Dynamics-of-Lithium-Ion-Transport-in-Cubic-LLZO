import numpy as np
from ase.optimize import BFGS
from ase.io import read
from elastic import get_elementary_deformations, get_elastic_tensor
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import ase.units as units
import pandas as pd

# === STEP 1: Load ORB ML model ===
device = "cuda"  # or "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)

# === STEP 2: Load structure ===
atoms = read('cubic-LLZO.cif')
atoms.calc = calc

# === STEP 3: Relax the structure ===
print("Initial Energy:", atoms.get_potential_energy())
dyn = BFGS(atoms)
dyn.run(fmax=0.02)
print("Relaxed Energy:", atoms.get_potential_energy())

# === STEP 4: Loop over n and d ===
n_values = [3, 5, 7]
d_values = [0.1, 0.2, 0.3]

results = []
all_C_matrices = []

for n in n_values:
    for d in d_values:
        print(f"\n=== Elastic calculation: n={n}, d={d} ===")
        # Generate deformations
        systems = get_elementary_deformations(atoms, n=n, d=d)
        for s in systems:
            s.calc = calc
            s.get_potential_energy()
        # Compute elastic tensor
        Cij, Bij = get_elastic_tensor(atoms, systems=systems)
        Cij_GPa = Cij / units.GPa
        all_C_matrices.append(Cij_GPa)

        # For cubic symmetry
        C11 = Cij_GPa[0]
        C12 = Cij_GPa[3]
        C44 = Cij_GPa[6]
        bulk = (C11 + 2 * C12) / 3
        shear = (C11 - C12 + 3 * C44) / 5
        youngs = 9 * bulk * shear / (3 * bulk + shear)
        poisson = (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))

        print("--- Elastic Stiffness Matrix (GPa) ---")
        print(Cij_GPa)
        print(f"C11={C11:.2f} GPa, C12={C12:.2f} GPa, C44={C44:.2f} GPa")
        print("--- Derived Mechanical Properties ---")
        print(f"Bulk modulus     = {bulk:.2f} GPa")
        print(f"Shear modulus    = {shear:.2f} GPa")
        print(f"Young's modulus  = {youngs:.2f} GPa")
        print(f"Poisson's ratio  = {poisson:.4f}")

        results.append({
            'n': n, 'd': d,
            'C11': C11, 'C12': C12, 'C44': C44,
            'bulk': bulk, 'shear': shear, 'youngs': youngs, 'poisson': poisson
        })

# === STEP 5: Compute averages ===
C_array = np.array(all_C_matrices)  # shape: (len(results), 6x6 flattened)
C_avg = np.mean(C_array, axis=0)

# extract average C11, C12, C44 (same indices as before)
C11_avg = C_avg[0]
C12_avg = C_avg[3]
C44_avg = C_avg[6]
bulk_avg = (C11_avg + 2 * C12_avg) / 3
shear_avg = (C11_avg - C12_avg + 3 * C44_avg) / 5
youngs_avg = 9 * bulk_avg * shear_avg / (3 * bulk_avg + shear_avg)
poisson_avg = (3 * bulk_avg - 2 * shear_avg) / (2 * (3 * bulk_avg + shear_avg))

print("\n=== AVERAGED RESULTS ===")
print("--- Averaged Elastic Stiffness Matrix (GPa) ---")
print(C_avg)
print(f"C11={C11_avg:.2f} GPa, C12={C12_avg:.2f} GPa, C44={C44_avg:.2f} GPa")
print("--- Averaged Mechanical Properties ---")
print(f"Bulk modulus     = {bulk_avg:.2f} GPa")
print(f"Shear modulus    = {shear_avg:.2f} GPa")
print(f"Young's modulus  = {youngs_avg:.2f} GPa")
print(f"Poisson's ratio  = {poisson_avg:.4f}")

# Append averaged row to results
results.append({
    'n': 'avg', 'd': 'avg',
    'C11': C11_avg, 'C12': C12_avg, 'C44': C44_avg,
    'bulk': bulk_avg, 'shear': shear_avg, 'youngs': youngs_avg, 'poisson': poisson_avg
})

# === STEP 6: Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("elastic_scan_results.csv", index=False)
print("\nResults saved to elastic_scan_results.csv (with averaged values).")

