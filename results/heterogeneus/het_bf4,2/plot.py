import matplotlib.pyplot as plt
import re
import ast
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'output_swap0.85.txt')

with open(file_path, 'r') as file:
    file_content = file.read()

json_like_content = re.search(r'\{.*\}', file_content, re.DOTALL).group()
data = ast.literal_eval(json_like_content)

protocol_lengths = []
num_distillations = []
secret_key_rates = []

distillation_pattern = re.compile(r"d\d+")

for protocol, key_rate in data.items():
    steps = eval(protocol)
    protocol_length = len(steps)
    distillations = len(distillation_pattern.findall(protocol))
    protocol_lengths.append(protocol_length)
    num_distillations.append(distillations)
    secret_key_rates.append(key_rate)

fig, ax = plt.subplots()

ax.scatter(num_distillations, secret_key_rates, color='blue')

ax.set_xlabel('Number of Distillations')
ax.set_ylabel('Secret Key Rate')
ax.set_title('Optimality of Rounds of Distillation')
ax.grid(True)

plt.savefig(os.path.join(script_dir, 'output_swap0.85.png'), dpi=1000)
