import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# given data
T = np.array([500, 600, 700, 800, 900, 1000], dtype=float)  # Kelvin
y = np.array([1.385e-06, 7.355e-07, 2.464e-06, 4.549e-06, 7.439e-06, 6.841e-06], dtype=float)

# constants
kB = 8.617333262e-5  # eV/K

# transform for data
x_new = 1000.0 / T
ln_y = np.log(y)

# linear regression
slope, intercept, r_value, p_value, std_err = linregress(x_new, ln_y)

# ---- reference curve ----
D0_ref = 3.06e-4
Ea_ref = 0.31  # eV
ln_D_ref = np.log(D0_ref) - Ea_ref/(kB*T)

# ---- plotting ----
plt.figure(figsize=(8,6))
# scatter for data
plt.scatter(x_new, ln_y, color='blue', label='Data')
# fitted line
plt.plot(x_new, slope*x_new + intercept, color='orange',
         label=r'Fit: $D = 5.9478 \times 10^{-5} \exp\!\left(\frac{-0.18493}{k_B T}\right)$')
# reference dashed
plt.plot(x_new, ln_D_ref, 'g--',
         label=r'Ref: $D = 3.06 \times 10^{-4} \exp\!\left(\frac{-0.31}{k_B T}\right)$')

plt.xlabel(r"$1000/T$ (1/K)")
plt.ylabel(r"$\ln(D)$ (ln(cm$^2$/s))")
plt.title("Arrhenius Plot for Diffusivity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.6f}")
print(f"Intercept: {intercept:.6f}")

