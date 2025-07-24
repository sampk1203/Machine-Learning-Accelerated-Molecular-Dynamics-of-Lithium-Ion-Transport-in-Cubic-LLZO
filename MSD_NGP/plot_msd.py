import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
folders = ['500', '600', '700', '800', '900', '1000']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
labels = ['500 K', '600 K', '700 K', '800 K', '900 K', '1000 K']
filename = 'msd_ngp_Li.txt'

# --- Plot 1: MSD vs Time ---
plt.figure(figsize=(10,6))
for folder, color, label in zip(folders, colors, labels):
    data = np.loadtxt(os.path.join(folder, filename))
    time = data[:,1]           # 2nd column
    msd = data[:,2]            # 3rd column
    plt.plot(time, msd, color=color, label=label)
plt.xlabel('Time (ps)', fontsize=18)
plt.ylabel('MSD (Å²)', fontsize=18)
plt.title('Mean squared displacement (MSD) vs Time', fontsize=18)
plt.legend(title='Temperature', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.show()

# --- Plot 2: NGP vs Time ---
plt.figure(figsize=(10,6))
for folder, color, label in zip(folders, colors, labels):
    data = np.loadtxt(os.path.join(folder, filename))
    time = data[:,1]           # 2nd column
    ngp = data[:,3]            # 4th column
    plt.plot(time, ngp, color=color, label=label)
plt.yscale('log')
plt.xlabel('Time (ps)', fontsize=18)
plt.ylabel('NGP', fontsize=18)
plt.title('Non-gaussianity parameter (NGP) vs Time', fontsize=18)
plt.legend(title='Temperature', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

