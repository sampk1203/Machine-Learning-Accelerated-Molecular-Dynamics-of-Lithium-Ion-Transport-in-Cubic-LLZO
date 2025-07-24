# Machine-Learning-Accelerated-Molecular-Dynamics-of-Lithium-Ion-Transport-in-Cubic-LLZO
Dataset for the paper.

**Note:** Each main folder contains subfolders corresponding to the temperature at which the data were computed.

---

## 📁 LAMMPS
Contains the input files used to generate the MSD, NGP, and dump files.

- `in_LLZO` – LAMMPS input script.  
- `gnnp_driver.py` – Python driver file to interface ORB‑models with LAMMPS.  
- `cubic-LLZO.data` – LAMMPS input data file generated via `cif2lmpdat.py` in `../OPTIMISATION`.  
- `dump.lammpstrj` – LAMMPS trajectory dump file.

---

## 📁 MECH
Contains the CIF file and Python script to obtain the mechanical properties of cubic‑LLZO at 0 K.

- `cubic-LLZO.cif` – Input CIF file.  
- `get_elastic_moduli.py` – Uses static deformations to compute the Born elastic matrix.  
- `elastic_scan_results.csv` – Born matrix elements and derived moduli in GPa.

---

## 📁 MSD_NGP
Contains MSD and NGP data for Li at each sampling temperature.

- `get_diffusion.py` – Fits the MSD vs. time data and returns diffusivity in cm²/s.  
- `plot_arrhenius.py` – Plots and fits hard‑coded diffusion coefficients.  
- `plot_MSD-NGP.py` – Visualizes MSD and NGP.  
- `plot_MSD_vs_NGP.py` – Self‑explanatory: plots MSD vs. NGP.

---

## 📁 OPTIMISATION
Contains Python scripts used to relax the `cubic-LLZO.cif` structure to obtain `relaxed.cif` for use in LAMMPS simulations.  
The CIF is converted to a LAMMPS data file via `cif2lmp.dat`.

---

## 📁 RDF
Contains the data for RDF between Li and all other atoms.  
Labels correspond as:  
**1 → Li, 2 → La, 3 → Zr, 4 → O.**

- `plot_RDF.py` – Plots RDF for all temperatures and all pairs in each corresponding temperature folder.

---

**Each folder is organized by temperature subfolders, making it easy to locate the simulation data at a given temperature.**

