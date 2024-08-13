import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softplus(x):
    return np.log1p(np.exp(x))

def softsign(x):
    return x / (1 + np.abs(x))

def exponential(x):
    return np.exp(x)

def hard_sigmoid(x):
    return np.clip((0.2 * x) + 0.5, 0.0, 1.0)

def arctanh(x):
    return np.arctanh(np.clip(x, -0.99, 0.99))

def sinc(x):
    x = np.where(x == 0, 1e-20, x)
    return np.sin(x) / x

def gaussian(x):
    return np.exp(-x**2)

def bent_identity(x):
    return (np.sqrt(x**2 + 1) - 1) / 2 + x

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def prelu(x, alpha=0.5):
    return np.where(x >= 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def step(x):
    return np.where(x >= 0, 1, 0)

def pentanh(x):
    return (4/3) * (np.tanh(x) - (1/3) * np.tanh(3*x))

def eswish(x, beta=1.0):
    return beta * x * sigmoid(x)

def mila(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def sqnl(x):
    return np.where(x > 2, 1, np.where(x < -2, -1, x - (x**2)/4))

def celu(x, alpha=1.0):
    return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x/alpha) - 1))

def pelu(x, alpha=1.0, c=1.0):
    return np.where(x >= 0, c*x, alpha*(np.exp(x/alpha) - 1))

def silu(x):
    return x * sigmoid(x)

def gcu(x):
    return x * np.cos(x)

def isru(x, alpha=1.0):
    return x / np.sqrt(1 + alpha * x**2)

def isrlu(x, alpha=1.0):
    return np.where(x < 0, x / np.sqrt(1 + alpha * x**2), x)

def softclip(x, alpha=0.5):
    return (1/alpha) * np.log(1 + np.exp(alpha*x))

def loglog(x):
    return 1 - np.exp(-np.exp(x))

def erf(x):
    def erf_scalar(t):
        return 2 / np.sqrt(np.pi) * sum((-1)**n * t**(2*n+1) / (math.factorial(n) * (2*n+1)) for n in range(10))
    return np.vectorize(erf_scalar)(x)

def unitball(x):
    return np.where(np.abs(x) < 1, x * (2 - np.abs(x)), np.sign(x))

def phish(x):
    return x * np.tanh(gelu(x))

def elliot(x):
    return x / (1 + np.abs(x))

def srs(x):
    return np.cbrt((1 + x)**3 - 1)

def smelu(x, beta=1.0):
    return np.log(1 + np.exp(beta * x)) / beta

def nlrelu(x, beta=1.0):
    return np.where(x >= 0, x, np.log(1 + np.abs(beta * x)) / beta)

def cosine(x):
    return np.cos(x)

def bipolar(x):
    return np.tanh(x) * (1 - np.tanh(x)**2)

activation_functions = [
    (linear, "Linear", (-5, 5)),
    (relu, "ReLU", (-5, 5)),
    (leaky_relu, "Leaky ReLU", (-5, 5)),
    (elu, "ELU", (-5, 5)),
    (selu, "SELU", (-5, 5)),
    (gelu, "GELU", (-5, 5)),
    (sigmoid, "Sigmoid", (-7, 7)),
    (tanh, "Tanh", (-5, 5)),
    (softplus, "Softplus", (-5, 5)),
    (softsign, "Softsign", (-5, 5)),
    (exponential, "Exponential", (-5, 2)),
    (hard_sigmoid, "Hard Sigmoid", (-5, 5)),
    (arctanh, "Arctanh", (-0.99, 0.99)),
    (sinc, "Sinc", (-10, 10)),
    (gaussian, "Gaussian", (-3, 3)),
    (bent_identity, "Bent Identity", (-5, 5)),
    (swish, "Swish", (-5, 5)),
    (mish, "Mish", (-5, 5)),
    (prelu, "PReLU", (-5, 5)),
    (softmax, "Softmax", (-5, 5)),
    (step, "Step", (-5, 5)),
    (pentanh, "Pentanh", (-5, 5)),
    (eswish, "E-Swish", (-5, 5)),
    (mila, "MILA", (-5, 5)),
    (sqnl, "SQNL", (-5, 5)),
    (celu, "CELU", (-5, 5)),
    (pelu, "PELU", (-5, 5)),
    (silu, "SiLU", (-5, 5)),
    (gcu, "GCU", (-5, 5)),
    (isru, "ISRU", (-5, 5)),
    (isrlu, "ISRLU", (-5, 5)),
    (softclip, "Soft Clipping", (-5, 5)),
    (loglog, "LogLog", (-5, 5)),
    (erf, "Error Function", (-3, 3)),
    (unitball, "Unit Ball", (-2, 2)),
    (phish, "Phish", (-5, 5)),
    (elliot, "Elliot", (-5, 5)),
    (srs, "SRS", (-5, 5)),
    (smelu, "SMELU", (-5, 5)),
    (nlrelu, "NLReLU", (-5, 5)),
    (cosine, "Cosine", (-np.pi, np.pi)),
    (bipolar, "Bipolar", (-5, 5))
]

# Belirgin farklı renkler
colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#FFA500', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E',
    '#FF7F50', '#6495ED', '#ACC83C', '#DC143C', '#00008B', '#008B8B',
    '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F',
    '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
    '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969'
]

# Renk listesini fonksiyon sayısına göre genişlet
while len(colors) < len(activation_functions):
    colors.extend(colors)
colors = colors[:len(activation_functions)]

fig = plt.figure(figsize=(30, 35), constrained_layout=True)
gs = GridSpec(7, 6, figure=fig)

for i, ((func, name, (x_min, x_max)), color) in enumerate(zip(activation_functions, colors)):
    row = i // 6
    col = i % 6
    ax = fig.add_subplot(gs[row, col])
    
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)
    ax.plot(x, y, label=name, color=color, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Extended Activation Functions', fontsize=24, fontweight='bold')
plt.savefig('activation_functions_distinct_colors.png', dpi=300, bbox_inches='tight')
plt.show()