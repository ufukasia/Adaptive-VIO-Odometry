from itertools import product

# Define parameter ranges
alpha_values = [round(0.7 + i * 0.1, 1) for i in range(6)]  # 0.7 to 1.3 in 0.1 steps
beta_values = [round(0.8 + i * 0.1, 1) for i in range(1)]   # Only one beta value: 0.8
activation_functions = ['relu', 'double_exponential_sigmoid', 'triple_exponential_sigmoid']
theta_threshold_values = [round(0.2 + i * 0.025, 2) for i in range(5)]  # 0.2 to 0.3 in 0.025 steps
sequences = ["MH_03_medium", "MH_04_difficult", "MH_05_difficult"]

# Generate all combinations
all_combinations = list(product(alpha_values, beta_values, activation_functions, theta_threshold_values, sequences))

# Prepare the output commands
commands = [
    f"python main.py --alpha {alpha} --beta {beta} --activation_function {activation} --theta_threshold {theta} --sequence {sequence}"
    for alpha, beta, activation, theta, sequence in all_combinations
]

# Output results to a file
with open("batch_test_outputs.txt", "w") as f:
    for command in commands:
        f.write(command + "\n")

print(f"{len(commands)} combinations generated and saved to output.txt")