import os
import numpy as np
import matplotlib.pyplot as plt

# Temperature folders
temps = [500, 600, 700, 800, 900, 1000]

# RDF pair files and labels
pairs = [
    ("rdf_1-1.dat", "Li–Li"),
    ("rdf_1-2.dat", "Li–La"),
    ("rdf_1-3.dat", "Li–Zr"),
    ("rdf_1-4.dat", "Li–O")
]

# Colors for each pair
pair_colors = {
    "Li–Li": "tab:blue",
    "Li–La": "tab:orange",
    "Li–Zr": "tab:green",
    "Li–O": "tab:red"
}

# Linestyles for temperatures
linestyles = [
    "-",   # 500K
    "--",  # 600K
    "-.",  # 700K
    ":",   # 800K
    (0, (3, 1, 1, 1)), # 900K
    (0, (5, 1))        # 1000K
]

plt.figure(figsize=(18, 9))

for t_idx, temp in enumerate(temps):
    folder = str(temp)
    for filename, label in pairs:
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        data = np.loadtxt(filepath, comments="#")
        r = data[:, 0]
        g = data[:, 1]
        # Plot with full label for every curve
        plt.plot(
            r, g,
            linestyle=linestyles[t_idx],
            color=pair_colors[label],
            label=f"{label} @ {temp}K"
        )

# Beautify
plt.xlabel("r (Å)", fontsize=14)
plt.ylabel("g(r)", fontsize=14)
plt.title("g(r) for All RDF Pairs and Temperatures", fontsize=16)
plt.xlim(0, 8)
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)

# Legend with all entries
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)

plt.tight_layout()
plt.savefig("rdf_all.png", dpi=300)
plt.show()
