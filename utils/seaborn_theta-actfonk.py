import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Veriyi yükleme
file_path = "run_results.csv"
data = pd.read_csv(file_path)

# Euler açılarındaki RMSE değerlerini parse etme ve ortalama hesaplama
data['rmse_euler_angles_parsed'] = data['rmse_euler_angles'].apply(ast.literal_eval)
data[['rmse_roll', 'rmse_pitch', 'rmse_yaw']] = pd.DataFrame(data['rmse_euler_angles_parsed'].to_list(), index=data.index)
data['rmse_euler_mean'] = data[['rmse_roll', 'rmse_pitch', 'rmse_yaw']].mean(axis=1)

# MH3, MH4 ve MH5 sekanslarına odaklanma
filtered_sequences = data[data['sequence_name'].isin(['MH_03_medium', 'MH_04_difficult', 'MH_05_difficult'])]

# Aktivasyon Fonksiyonlarına Göre Kutu Grafik
selected_activations = ["double_exponential_sigmoid", "relu", "triple_exponential_sigmoid"]
filtered_activations_data = filtered_sequences[filtered_sequences['activation_function'].isin(selected_activations)]

plt.figure(figsize=(14, 6))
sns.boxplot(x='activation_function', y='rmse_euler_mean', hue='sequence_name', data=filtered_activations_data)
plt.xlabel('Activation Function', labelpad=10)
plt.ylabel('RMSE (Euler Angles)', labelpad=10)
plt.title('RMSE Distribution for Selected Activation Functions (MH_03, MH_04, MH_05)', pad=15)
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
for i in range(len(filtered_activations_data['activation_function'].unique()) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)
plt.legend(title='Sequence', loc='upper left', frameon=True)
plt.tight_layout()
plt.show()

# Theta Threshold Değerlerine Göre Kutu Grafik
plt.figure(figsize=(14, 6))
sns.boxplot(x='theta_threshold', y='rmse_euler_mean', hue='sequence_name', data=filtered_sequences)
plt.xlabel('Theta Threshold', labelpad=10)
plt.ylabel('RMSE (Euler Angles)', labelpad=10)
plt.title('RMSE Distribution for Theta Thresholds (MH_03, MH_04, MH_05)', pad=15)
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
for i in range(len(filtered_sequences['theta_threshold'].unique()) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)
plt.legend(title='Sequence', loc='upper left', frameon=True)
plt.tight_layout()
plt.show()