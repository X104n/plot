import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Configure matplotlib for large datasets
mpl.rcParams['agg.path.chunksize'] = 10000  # Increase chunk size as suggested in the error
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5  # Higher value = more simplification

print("Starting dataOld analysis...")
start_time = time.time()


# Function to read the dataOld file - optimized for large files
def read_data(file_path):
    print(f"Reading dataOld from {file_path}...")
    try:
        # Use pandas with chunking for large files
        df = pd.read_csv(file_path, header=None)
        data = df.iloc[:, 0].to_numpy()
        print(f"Successfully read {len(data)} dataOld points")
        return data
    except Exception as e:
        print(f"Error with pandas: {e}")
        try:
            # Fallback to line-by-line reading
            with open(file_path, 'r') as file:
                times = []
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            value = float(line)
                            times.append(value)
                        except ValueError:
                            pass  # Skip lines that can't be converted to float
                print(f"Successfully read {len(times)} dataOld points using fallback method")
                return np.array(times)
        except Exception as e:
            print(f"Error reading file: {e}")
            return np.array([])


# Function to remove outliers using different methods
def remove_outliers(data, method='iqr', threshold=1.5):
    """
    Remove outliers from the dataset

    Parameters:
    - dataOld: numpy array of dataOld points
    - method: 'iqr' (Interquartile Range), 'std' (Standard Deviation), or 'percentile'
    - threshold: multiplier for IQR or std methods, or percentile range (0-100) for percentile method

    Returns:
    - Filtered dataOld without outliers
    - Boolean mask indicating which points are outliers
    """
    if method == 'iqr':
        # IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)
        print(f"IQR outlier removal: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

    elif method == 'std':
        # Standard deviation method
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        mask = (data >= lower_bound) & (data <= upper_bound)
        print(f"STD outlier removal: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

    elif method == 'percentile':
        # Percentile method (threshold is the percentile range to keep, e.g., 5 means keep 5th to 95th percentile)
        lower_bound = np.percentile(data, threshold)
        upper_bound = np.percentile(data, 100 - threshold)
        mask = (data >= lower_bound) & (data <= upper_bound)
        print(f"Percentile outlier removal: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

    else:
        # No outlier removal
        print("No outlier removal performed")
        return data, np.ones(len(data), dtype=bool)

    filtered_data = data[mask]
    print(
        f"Removed {len(data) - len(filtered_data)} outliers ({100 * (len(data) - len(filtered_data)) / len(data):.2f}%)")

    return filtered_data, mask


# Function to downsample dataOld for plotting
def downsample_data(data, indices=None, max_points=100_000):
    """Downsample dataOld to a manageable size for plotting"""
    if indices is None:
        indices = np.arange(len(data))

    n = len(data)
    if n <= max_points:
        return indices, data

    # Calculate downsampling factor
    factor = int(np.ceil(n / max_points))
    print(f"Downsampling by factor of {factor} (original: {n}, target: {max_points})")

    # Method 1: Simple striding (fastest)
    idx_indices = np.arange(0, n, factor)
    downsampled_indices = indices[idx_indices]
    downsampled = data[idx_indices]

    # Find global min and max to make sure they're included
    global_min_idx = np.argmin(data)
    global_max_idx = np.argmax(data)

    # Ensure we add global min/max if they weren't in our sample
    if global_min_idx % factor != 0:
        downsampled_indices = np.append(downsampled_indices, indices[global_min_idx])
        downsampled = np.append(downsampled, data[global_min_idx])
    if global_max_idx % factor != 0:
        downsampled_indices = np.append(downsampled_indices, indices[global_max_idx])
        downsampled = np.append(downsampled, data[global_max_idx])

    # Sort by indices
    sorted_order = np.argsort(downsampled_indices)
    return downsampled_indices[sorted_order], downsampled[sorted_order]


# Path to your dataOld file
file_path = 'data/nzt5/domain_app5.csv'  # Change to your actual file path

# Read the dataOld
travel_times = read_data(file_path)

# Check if we have dataOld
if len(travel_times) == 0:
    print("No dataOld was read. Exiting.")
    exit()

# Convert to milliseconds for better readability
travel_times_ms = travel_times * 1000

# Calculate statistics on the full dataset
print("Calculating statistics for original dataOld...")
orig_mean = np.mean(travel_times_ms)
orig_median = np.median(travel_times_ms)
orig_min = np.min(travel_times_ms)
orig_max = np.max(travel_times_ms)
orig_std = np.std(travel_times_ms)

print(f"Original dataOld statistics:")
print(f"- Total points: {len(travel_times_ms):,}")
print(f"- Min: {orig_min:.2f} ms")
print(f"- Max: {orig_max:.2f} ms")
print(f"- Mean: {orig_mean:.2f} ms")
print(f"- Median: {orig_median:.2f} ms")
print(f"- Std Dev: {orig_std:.2f} ms")

# Remove outliers
print("\nRemoving outliers...")
# Choose outlier removal method: 'iqr', 'std', 'percentile', or None
outlier_method = 'percentile'  # Standard IQR method
outlier_threshold = 0.5  # Standard threshold for IQR method

if outlier_method:
    filtered_data, outlier_mask = remove_outliers(travel_times_ms, method=outlier_method, threshold=outlier_threshold)

    # Keep track of original indices for the filtered dataOld
    original_indices = np.arange(len(travel_times_ms))[outlier_mask]

    # Calculate statistics on the filtered dataset
    print("\nCalculating statistics for filtered dataOld...")
    mean_time = np.mean(filtered_data)
    median_time = np.median(filtered_data)
    min_time = np.min(filtered_data)
    max_time = np.max(filtered_data)
    std_dev = np.std(filtered_data)

    print(f"Filtered dataOld statistics:")
    print(f"- Total points: {len(filtered_data):,}")
    print(f"- Min: {min_time:.2f} ms")
    print(f"- Max: {max_time:.2f} ms")
    print(f"- Mean: {mean_time:.2f} ms")
    print(f"- Median: {median_time:.2f} ms")
    print(f"- Std Dev: {std_dev:.2f} ms")
else:
    # No outlier removal
    filtered_data = travel_times_ms
    original_indices = np.arange(len(travel_times_ms))
    mean_time = orig_mean
    median_time = orig_median
    min_time = orig_min
    max_time = orig_max
    std_dev = orig_std

# Downsample dataOld for plotting
print("\nDownsampling dataOld for plotting...")
indices, downsampled_data = downsample_data(filtered_data, indices=original_indices, max_points=50000)

# Create a figure
print("Creating figure...")
plt.figure(figsize=(12, 8))

# Plot the downsampled dataOld
plt.scatter(indices, downsampled_data, s=2, color='blue', alpha=0.5,
            label=f'Data points (downsampled from {len(filtered_data):,} to {len(indices):,})')

# For very large datasets, don't connect the points with lines
if len(indices) < 10000:
    plt.plot(indices, downsampled_data, 'b-', alpha=0.2, linewidth=0.3)

# Add mean and median lines
plt.axhline(y=mean_time, color='r', linestyle='-',
            label=f'Mean: {mean_time:.2f} ms')
plt.axhline(y=median_time, color='g', linestyle='--',
            label=f'Median: {median_time:.2f} ms')

# Add min and max annotations - we know these are in our downsampled dataOld
min_plot_idx = np.argmin(downsampled_data)
max_plot_idx = np.argmax(downsampled_data)

plt.annotate(f'Min: {min_time:.2f} ms',
             xy=(indices[min_plot_idx], downsampled_data[min_plot_idx]),
             xytext=(indices[min_plot_idx], downsampled_data[min_plot_idx] - 0.05 * (max_time - min_time)),
             arrowprops=dict(arrowstyle="->", color='darkblue'),
             color='darkblue', fontsize=9)

plt.annotate(f'Max: {max_time:.2f} ms',
             xy=(indices[max_plot_idx], downsampled_data[max_plot_idx]),
             xytext=(indices[max_plot_idx], downsampled_data[max_plot_idx] + 0.05 * (max_time - min_time)),
             arrowprops=dict(arrowstyle="->", color='darkred'),
             color='darkred', fontsize=9)

# Set labels and title
if outlier_method:
    plt.title(f'Time Series of Network Travel Times - Outliers Removed ({outlier_method.upper()})',
              fontsize=14, fontweight='bold')
else:
    plt.title(f'Time Series of Network Travel Times (from {len(travel_times_ms):,} points)',
              fontsize=14, fontweight='bold')

plt.xlabel('Measurement Index', fontsize=12)
plt.ylabel('Travel Time (milliseconds)', fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend in a fixed position rather than 'best' to avoid slow performance
plt.legend(loc='upper right')

# Set y-axis limits to give some padding
y_range = max_time - min_time
plt.ylim(min_time - 0.2 * y_range, max_time + 0.2 * y_range)

# Make the plot look nice
plt.tight_layout()

# Save the figure
print("Saving figure...")
if outlier_method:
    filename = f'images/{outlier_method}/{file_path.split('/')[1]}.png'
else:
    filename = 'large_network_times.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Figure saved as '{filename}'")

# Print execution time
print(f"Total execution time: {time.time() - start_time:.2f} seconds")

# Show the plot
plt.show()