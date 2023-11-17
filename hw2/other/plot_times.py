import matplotlib.pyplot as plt
import numpy as np

# filter = None to disable
filter = 12000

# data [vector length, t_memToDevice, t_kernel, t_memFromDevice]
data = [
    [1024, 0.055, 0.031, 0.022], 
    [8092, 0.088, 0.029, 0.07], 
    [65000, 0.307, 0.059, 0.384], 
    [250000, 1.134, 0.1, 1.722], 
    [2000000, 7.369, 0.232, 14.229], 
    [120000, 0.634, 0.094, 0.903], 
    [12000, 0.083, 0.024, 0.11],
    [4000, 0.059, 0.024, 0.044],
    [1, 0.032, 0.025, 0.016],
    [128, 0.034, 0.025, 0.018],
    [131070, 0.728, 0.105, 0.864]
]

# Filter data
if filter is not None:
    data = [d for d in data if d[0] <= filter]

# Sort data by vector length
data.sort(key=lambda x: x[0])

# Separate data into different lists
vector_length, t_memToDevice, t_kernel, t_memFromDevice = zip(*data)

x = np.arange(len(vector_length))

# Plot stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(x, t_memToDevice, label='t_memToDevice')
plt.bar(x, t_kernel, bottom=t_memToDevice, label='t_kernel')
plt.bar(x, t_memFromDevice, bottom=np.array(t_memToDevice)+np.array(t_kernel), label='t_memFromDevice')

# Add labels and title
plt.xlabel('Vector Length')
plt.ylabel('Time')
plt.title('Stacked Bar Chart')
plt.xticks(x, vector_length, rotation='vertical')

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.savefig('stacked_bar_chart.png')
plt.show()
