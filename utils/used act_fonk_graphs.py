import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Orijinal fonksiyonlar ---------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x >= 0, 1, 0)

# --- Parametreli exp_unit ----------------------------------------------------

def exp_unit(x, a):
    x_clip = np.minimum(x, 1.0)
    return (np.exp(a * x_clip) - 1) / (np.exp(a) - 1)

# --- Çizdirilecek aktivasyon fonksiyonları -----------------------------------

# Burada istediğiniz a değerlerini ekleyin:
a_values = [4, -4]

# activation_functions listesini otomatik oluşturuyoruz:
activation_functions = []

# exp_unit varyantları:
for a in a_values:
    func = lambda x, a=a: exp_unit(x, a)
    name = f"Exp Unit (a={a})"
    activation_functions.append((func, name, (-0.1, 1)))

# Diğer fonksiyonlar:
activation_functions += [
    (relu, "ReLU", (-1, 5)),
    (step, "Step", (-1, 2))
]

# Renk paleti (fonksiyon sayısına yetecek uzunlukta olsun):
colors = ['#0000FF', '#FAAF00', '#FF00FF', '#228B22', '#8A2BE2', '#FF1493', '#FFA500']

# --- Çizim -------------------------------------------------------------------

fig = plt.figure(figsize=(22, 6), constrained_layout=True)
gs = GridSpec(1, len(activation_functions), figure=fig)

for i, ((func, name, (x_min, x_max)), color) in enumerate(zip(activation_functions, colors)):
    ax = fig.add_subplot(gs[0, i])
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)
    
    ax.plot(x, y, label=name, color=color, linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim(x_min, x_max)

    # Tüm fonksiyonları 0-1 aralığına sabitlemek istiyorsanız burayı açabilir
    # ax.set_ylim(-0.1, 1.1)
    # Aksi halde otomatik sınır:
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('f(x)', fontsize=9, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

#plt.suptitle('Aktivasyon Fonksiyonları ve Alternatifler', fontsize=22, fontweight='bold')
plt.savefig('activation_functions_alternatives.png', dpi=300, bbox_inches='tight')
plt.show()
