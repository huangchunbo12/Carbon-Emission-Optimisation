import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'

# Define the upper limit function
def upper_limit_function(delta):
    return np.exp(-1 / (4 * np.arctan(delta)))

# Define the range for delta
delta_values = np.linspace(0, 1, 500)
upper_limits = upper_limit_function(delta_values)

# Calculate the integral result, first integrating with respect to r, then delta
def integrand_4(delta):
    upper_limit = upper_limit_function(delta)
    return upper_limit  # The integral result is the upper limit, since the integral of r is 1

# Perform numerical integration
result_4, error_4 = quad(integrand_4, 0, 1)

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(delta_values, upper_limits, label=r'$e^{-\frac{1}{4} \arctan(\delta)}$', color='b')
plt.fill_between(delta_values, 0, upper_limits, alpha=0.3, color='b', label='Area under curve')

# Add title and labels
plt.title('Visualization of the Integral Region')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$e^{-\frac{1}{4} \arctan(\delta)}$')
plt.legend()
plt.grid(True)
plt.show()

# Print the integral result and error
print(f"Integral Result: {result_4:.4f}")
print(f"Error Estimate: {error_4:.4e}")