import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def read_speed_data(filepath):
    """
    Read speed measurement data from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pandas.Series: Series containing speed measurements
    """
    try:
        # Read CSV without header since it's just a single column of values
        data = pd.read_csv(filepath, header=None, names=['speed'])
        return data['speed']
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def calculate_statistics(speed_data):
    """
    Calculate basic statistics for speed data.

    Args:
        speed_data (pandas.Series): Speed measurements

    Returns:
        dict: Dictionary containing statistics
    """
    return {
        'count': len(speed_data),
        'mean': speed_data.mean(),
        'std': speed_data.std(),
        'min': speed_data.min(),
        'max': speed_data.max(),
        'median': speed_data.median()
    }


def create_speed_plot(speed_data, title="Speed Measurements",
                      save_path=None, show_plot=True,
                      y_min=None, y_max=None, x_min=None, x_max=None,
                      filter_data=False):
    """
    Create a matplotlib plot for speed measurements over time.

    Args:
        speed_data (pandas.Series): Speed measurements
        title (str): Plot title
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
        y_min (float): Minimum y-axis value (optional)
        y_max (float): Maximum y-axis value (optional)
        x_min (int): Minimum x-axis value (measurement number, optional)
        x_max (int): Maximum x-axis value (measurement number, optional)
        filter_data (bool): If True, filter data to only show values within y bounds

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Calculate statistics
    stats = calculate_statistics(speed_data)

    # Apply data filtering if requested
    original_data = speed_data.copy()
    measurement_sequence = range(1, len(speed_data) + 1)

    if filter_data and (y_min is not None or y_max is not None):
        if y_min is not None and y_max is not None:
            mask = (speed_data >= y_min) & (speed_data <= y_max)
        elif y_min is not None:
            mask = speed_data >= y_min
        else:  # y_max is not None
            mask = speed_data <= y_max

        speed_data = speed_data[mask]
        measurement_sequence = [i + 1 for i, keep in enumerate(mask) if keep]

        print(f"Filtered data: {len(speed_data):,} points remaining out of {len(original_data):,}")

    # Apply x-axis filtering if requested
    if x_min is not None or x_max is not None:
        start_idx = (x_min - 1) if x_min is not None else 0
        end_idx = x_max if x_max is not None else len(speed_data)

        start_idx = max(0, min(start_idx, len(speed_data)))
        end_idx = max(0, min(end_idx, len(speed_data)))

        if start_idx < end_idx:
            speed_data = speed_data.iloc[start_idx:end_idx]
            measurement_sequence = list(range(start_idx + 1, end_idx + 1))
            print(f"X-axis filtered: showing measurements {start_idx + 1} to {end_idx}")

    # Create measurement sequence for plotting
    if not isinstance(measurement_sequence, list):
        measurement_sequence = list(measurement_sequence)

    # Create the main figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot all data points
    ax.plot(measurement_sequence, speed_data, linewidth=0.5, alpha=0.7, color='blue')
    ax.set_title(f'{title} ({len(speed_data):,} measurements)')

    ax.set_xlabel('Measurement Number')
    ax.set_ylabel('Speed')
    ax.grid(True, alpha=0.3)

    # Apply axis limits
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    if x_min is not None or x_max is not None:
        ax.set_xlim(x_min, x_max)

    # Add statistics text box
    stats_text = (f'Count: {stats["count"]:,}\n'
                  f'Mean: {stats["mean"]:.6f}\n'
                  f'Std: {stats["std"]:.6f}\n'
                  f'Min: {stats["min"]:.6f}\n'
                  f'Max: {stats["max"]:.6f}\n'
                  f'Median: {stats["median"]:.6f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def process_multiple_files(file_paths, output_dir=None,
                           y_min=None, y_max=None, x_min=None, x_max=None,
                           filter_data=False):
    """
    Process multiple CSV files and create plots for each.

    Args:
        file_paths (list): List of file paths to process
        output_dir (str): Directory to save plots (optional)
        y_min (float): Minimum y-axis value (optional)
        y_max (float): Maximum y-axis value (optional)
        x_min (int): Minimum x-axis value (measurement number, optional)
        x_max (int): Maximum x-axis value (measurement number, optional)
        filter_data (bool): If True, filter data to only show values within y bounds
    """
    results = {}

    for filepath in file_paths:
        print(f"\nProcessing: {filepath}")

        # Read data
        speed_data = read_speed_data(filepath)
        if speed_data is None:
            continue

        # Create title from filename
        filename = Path(filepath).stem
        title = f"Speed Measurements - {filename}"

        # Determine save path
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{filename}_plot.png")

        # Create plot
        fig = create_speed_plot(speed_data, title=title, save_path=save_path,
                                show_plot=False, y_min=y_min, y_max=y_max,
                                x_min=x_min, x_max=x_max, filter_data=filter_data)

        # Store results
        results[filepath] = {
            'data': speed_data,
            'stats': calculate_statistics(speed_data),
            'figure': fig
        }

        print(f"Completed: {filepath} ({len(speed_data):,} measurements)")

    return results


# Example usage functions
def analyze_single_file(filepath, y_min=None, y_max=None, x_min=None, x_max=None,
                        filter_data=False):
    """
    Analyze a single CSV file and create a plot.

    Args:
        filepath (str): Path to the CSV file
        y_min (float): Minimum y-axis value (optional)
        y_max (float): Maximum y-axis value (optional)
        x_min (int): Minimum x-axis value (measurement number, optional)
        x_max (int): Maximum x-axis value (measurement number, optional)
        filter_data (bool): If True, filter data to only show values within y bounds
    """
    print(f"Analyzing: {filepath}")

    # Read data
    speed_data = read_speed_data(filepath)
    if speed_data is None:
        return None

    # Create title from filename
    filename = Path(filepath).stem
    title = f"Speed Measurements - {filename}"

    # Create and show plot
    fig = create_speed_plot(speed_data, title=title, y_min=y_min, y_max=y_max,
                            x_min=x_min, x_max=x_max, filter_data=filter_data)

    # Print statistics
    stats = calculate_statistics(speed_data)
    print("\nStatistics:")
    for key, value in stats.items():
        if key == 'count':
            print(f"{key.capitalize()}: {value:,}")
        else:
            print(f"{key.capitalize()}: {value:.6f}")

    return {'data': speed_data, 'stats': stats, 'figure': fig}


# Example usage:
if __name__ == "__main__":
    # Example 1: Process a single file - shows ALL measurements
    result = analyze_single_file('data/nzt1/domain_app1.csv', y_min=0, y_max=0.002)

    # Example 2: Process a single file with custom bounds
    # result = analyze_single_file('your_file.csv', y_min=0.001, y_max=0.002)

    # Example 3: Process a single file with filtered data (only show values in range)
    # result = analyze_single_file('your_file.csv', y_min=0.001, y_max=0.002, filter_data=True)

    # Example 4: Focus on specific measurement range (e.g., measurements 1000-2000)
    # result = analyze_single_file('your_file.csv', x_min=1000, x_max=2000)

    # Example 5: Process multiple files with same bounds
    # file_list = ['file1.csv', 'file2.csv', 'file3.csv']
    # results = process_multiple_files(file_list, output_dir='plots',
    #                                 y_min=0.0015, y_max=0.0017)

    # Example 6: Process all CSV files in directory with filtering
    # import glob
    # csv_files = glob.glob('*.csv')
    # results = process_multiple_files(csv_files, output_dir='plots',
    #                                 y_min=0.001, y_max=0.002, filter_data=True)

    print("Modular speed measurement plotter ready!")
    print("All measurements will be plotted - no sampling or reduction")
    print("Use analyze_single_file('filename.csv', y_min=0.001, y_max=0.002) for bounded plots")
    print("Add filter_data=True to only show data within the y bounds")
    print("Use x_min/x_max to focus on specific measurement ranges")