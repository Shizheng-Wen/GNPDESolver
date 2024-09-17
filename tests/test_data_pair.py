
num_timesteps = 14
t_in_indices = []
t_out_indices = []
for lag in range(2, num_timesteps+1, 2):
    num_pairs = (num_timesteps - lag)//2 + 1
    for i in range(0, num_timesteps-lag+1, 2):
        t_in_idx = i
        t_out_idx = i + lag
        t_in_indices.append(t_in_idx)
        t_out_indices.append(t_out_idx)

print(t_in_indices)
print(t_out_indices)
