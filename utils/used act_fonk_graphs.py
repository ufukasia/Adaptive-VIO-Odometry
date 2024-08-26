import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def quartic_unit_step(x):
    return np.minimum(x**4, 1)

def cubic_unit_step(x):
    return np.minimum(x**3, 1)

def quadratic_unit_step(x):
    return np.minimum(x**2, 1)

def relu(x):
    return np.maximum(0, x)

def double_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def triple_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + np.exp(-2*x))

def quadruple_exponential_sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + np.exp(-2*x) + np.exp(-3*x))

def step(x):
    return np.where(x >= 0, 1, 0)

activation_functions = [
    (quartic_unit_step, "Quartic Unit Step", (0, 1)),
    (cubic_unit_step, "Cubic Unit Step", (0, 1)),
    (quadratic_unit_step, "Quadratic Unit Step", (0, 1)),
    (relu, "ReLU", (-1, 5)),
    (double_exponential_sigmoid, "Double Exponential Sigmoid", (-5, 5)),
    (triple_exponential_sigmoid, "Triple Exponential Sigmoid", (-5, 5)),
    (quadruple_exponential_sigmoid, "Quadruple Exponential Sigmoid", (-5, 5)),
    (step, "Step", (-1, 2))
]

# Belirgin farklı renkler
colors = ['#FF0000', '#00FF00', '#0000FF', '#FAAF00', '#FF00FF', '#00FFFF', '#800000', '#FFA500']

fig = plt.figure(figsize=(20, 12), constrained_layout=True)
gs = GridSpec(2, 4, figure=fig)

for i, ((func, name, (x_min, x_max)), color) in enumerate(zip(activation_functions, colors)):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)
    ax.plot(x, y, label=name, color=color, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    if name in ["Quadratic Unit Step", "Cubic Unit Step", "Quartic Unit Step"]:
        ax.set_ylim(0, 1.1)
    elif name == "Step":
        ax.set_ylim(-0.1, 1.1)
    else:
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Reordered Activation Functions', fontsize=24, fontweight='bold')
plt.savefig('reordered_activation_functions_4x2.png', dpi=300, bbox_inches='tight')
plt.show()