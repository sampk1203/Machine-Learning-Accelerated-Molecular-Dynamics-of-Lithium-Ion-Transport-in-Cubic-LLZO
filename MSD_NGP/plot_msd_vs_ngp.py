import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
folders = ['500', '600', '700', '800', '900', '1000']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
labels = ['500 K', '600 K', '700 K', '800 K', '900 K', '1000 K']
filename = 'msd_ngp_Li.txt'

# --- Plot MSD (3rd col) vs NGP (4th col) ---
plt.figure(figsize=(10,6))
for folder, color, label in zip(folders, colors, labels):
    data = np.loadtxt(os.path.join(folder, filename))
    msd = data[:,2]   # 3rd column
    ngp = data[:,3]   # 4th column
    plt.plot(msd, ngp, color=color, label=label)

plt.xlabel('MSD (Å²)', fontsize=18)
plt.ylabel('NGP', fontsize=18)
plt.title('Mean squared displacement vs Non-gaussianity parameter', fontsize=14)
plt.legend(title='Temperature', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.grid(True)
plt.show()

