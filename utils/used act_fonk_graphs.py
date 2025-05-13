import numpy as np
import matplotlib.pyplot as plt

# --- Aktivasyon fonksiyonları ------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x >= 0, 1, 0)

def casef(x, a):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF)
    Negatif girdilerde 0, 0–1 aralığında üssel olarak ölçeklenir,
    1 üzerindeki girdilerde doygunluk (saturasyon) ile 1 değerini alır.
    """
    x_clip = np.clip(x, 0.0, 1.0)
    return (np.exp(a * x_clip) - 1) / (np.exp(a) - 1)

def exponential_surge(x):
    return np.piecewise(
        x,
        [x < 0, (x >= 0) & (x <= 1), x > 1],
        [0, lambda x: 1 - (1 - x) ** 5, 1]
    )

def quartic_unit_step(x):
    return np.minimum(x**4, 1)

# --- Parametreler -----------------------------------------------------------

# İlk satırda 4 "klasik" fonksiyon
top_functions = [
    (quartic_unit_step,     "Quartic Unit Step"),
    (relu,                  "ReLU"),
    (exponential_surge,     "Exponential Surge"),
    (step,                  "Step"),
]

# Alt satırda 4 farklı CASEF varyantı
a_values = [4, 0.01, -4, -300]
bottom_functions = [
    (lambda x, a=a: casef(x, a), f"CASEF (a={a})")
    for a in a_values
]

# Renkler (toplam 8 adet)
colors = [
    '#0000FF', '#FAAF00', '#FF00FF', '#228B22',
    '#8A2BE2', '#FF1493', '#FFA500', '#00CED1'
]

# --- Grafik çizimi ---------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(22, 8), constrained_layout=True)

for idx, (func, name) in enumerate(top_functions + bottom_functions):
    ax = axes[idx // 4, idx % 4]
    x = np.linspace(-0.1, 1, 1000)
    y = func(x)

    ax.plot(x, y, label=name, color=colors[idx], linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.1, 1.01)
    ax.set_ylim(-0.1, 1.01)

    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('f(x)', fontsize=9, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.savefig('activation_functions_2rows.png', dpi=300, bbox_inches='tight')
plt.show()
