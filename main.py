import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Function to read the dataOld file
def read_data(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    value = float(line)
                    times.append(value)
                except ValueError:
                    pass  # Skip lines that can't be converted to float
    return np.array(times)

# Path to your dataOld file
file_path = 'dataOld/domain_app_overnight_zt.csv'  # Replace with your actual file path

# Read the dataOld
travel_times = read_data(file_path)

# Convert to milliseconds for better readability
travel_times_ms = travel_times * 1000

# Calculate statistics for annotations
mean_time = np.mean(travel_times_ms)
median_time = np.median(travel_times_ms)
min_time = np.min(travel_times_ms)
max_time = np.max(travel_times_ms)
std_time = np.std(travel_times_ms)

# Create the histogram plot
plt.figure(figsize=(10, 6))

# Histogram with KDE
hist_bins = min(50, int(len(travel_times_ms) / 20))  # Dynamic number of bins
sns.histplot(travel_times_ms, bins=hist_bins, kde=True, color='skyblue',
             edgecolor='black', line_kws={'linewidth': 2, 'color': 'darkblue'})
plt.title('Distribution of Network Travel Times', fontsize=14, fontweight='bold')
plt.xlabel('Travel Time (milliseconds)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add a text box with statistics
stats_text = (
    f'Statistics:\n'
    f'Mean: {mean_time:.2f} ms\n'
    f'Median: {median_time:.2f} ms\n'
    f'Min: {min_time:.2f} ms\n'
    f'Max: {max_time:.2f} ms\n'
    f'Std Dev: {std_time:.2f} ms'
)
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add vertical lines for mean and median
plt.axvline(x=mean_time, color='r', linestyle='-', label=f'Mean: {mean_time:.2f} ms')
plt.axvline(x=median_time, color='g', linestyle='--', label=f'Median: {median_time:.2f} ms')
plt.legend()

plt.tight_layout()

# Save the figure
plt.savefig('network_time_histogram.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Histogram of network travel times created and saved as 'network_time_histogram.png'")