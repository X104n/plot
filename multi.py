import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob


def plot_domain_performance(folder_name="nzt5"):
    """
    Plot domain performance from CSV files in data/{folder_name}/

    Args:
        folder_name: "nzt1", "nzt3", or "nzt5"
    """

    # Define nice colors for up to 5 domains
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#9B59B6', '#E67E22']

    # Find all CSV files in the specified folder
    folder_path = f"data/{folder_name}"
    csv_files = glob.glob(f"{folder_path}/*.csv")

    if not csv_files:
        print(f"No CSV files found in {folder_path}/")
        return

    print(f"Found {len(csv_files)} CSV files in {folder_path}/")

    # Set up the plot
    plt.figure(figsize=(15, 10))

    domain_stats = {}

    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        filename = os.path.basename(csv_file)
        domain_name = filename.replace('.csv', '')  # Use filename as domain name
        color = colors[i % len(colors)]

        print(f"Processing {filename}...")

        # Read and parse the CSV file
        times = []
        values = []

        try:
            with open(csv_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()

                    # Skip empty lines and "No product found"
                    if not line or line == "No product found" or "No product found" in line:
                        continue

                    try:
                        # Try to parse as float (response time in seconds)
                        value = float(line)

                        # Only include successful operations (< 0.1 seconds = 100ms)
                        if value < 0.1:
                            times.append(line_num)
                            values.append(value * 1000)  # Convert to milliseconds

                    except ValueError:
                        # Skip non-numeric lines
                        continue

            if values:
                # Plot the data
                plt.scatter(times, values,
                            c=color,
                            alpha=0.6,
                            s=2,  # Small dots
                            label=f'{domain_name} (n={len(values)})')

                # Store stats
                domain_stats[domain_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

                print(f"  ✓ {domain_name}: {len(values)} successful operations")
            else:
                print(f"  ⚠ {domain_name}: No successful operations found")

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            continue

    # Customize the plot
    plt.xlabel('Measurement Index', fontsize=12)
    plt.ylabel('Response Time (ms)', fontsize=12)
    plt.title(f'{folder_name.upper()} - Domain Response Times (Successful Operations Only)',
              fontsize=14, fontweight='bold')

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Set reasonable y-axis limits
    plt.ylim(0, 15)  # Focus on 0-15ms range

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"STATISTICS FOR {folder_name.upper()}")
    print(f"{'=' * 60}")

    for domain, stats in domain_stats.items():
        print(f"\n{domain}:")
        print(f"  Operations: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f} ms")
        print(f"  Std Dev: {stats['std']:.2f} ms")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} ms")


def plot_side_by_side_grid():
    """
    Plot all folders side by side in a nice grid layout
    """
    folders = ['nzt1', 'nzt3', 'nzt5']

    # Create a 2x2 grid (3 individual plots + 1 comparison)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Domain Performance Analysis - Side by Side Comparison', fontsize=16, fontweight='bold')

    # Colors for different domains within each folder
    domain_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#9B59B6', '#E67E22']

    overall_stats = {}

    # Plot individual folders in first 3 subplots
    for folder_idx, folder_name in enumerate(folders):
        if folder_idx < 3:  # First 3 plots
            row = folder_idx // 2
            col = folder_idx % 2
            ax = axes[row, col]
        else:
            break

        folder_path = f"data/{folder_name}"
        csv_files = glob.glob(f"{folder_path}/*.csv")

        if not csv_files:
            ax.text(0.5, 0.5, f'No data found for {folder_name}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{folder_name.upper()} - No Data')
            continue

        domain_stats = {}
        all_folder_values = []

        # Process each CSV file in this folder
        for i, csv_file in enumerate(csv_files):
            filename = os.path.basename(csv_file)
            domain_name = filename.replace('.csv', '')
            color = domain_colors[i % len(domain_colors)]

            times = []
            values = []

            try:
                with open(csv_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()

                        if not line or line == "No product found" or "No product found" in line:
                            continue

                        try:
                            value = float(line)
                            if value < 0.1:  # Successful operations only
                                times.append(line_num)
                                values.append(value * 1000)  # Convert to ms
                        except ValueError:
                            continue

                if values:
                    ax.scatter(times, values,
                               c=color,
                               alpha=0.6,
                               s=2,
                               label=f'{domain_name} (n={len(values)})')

                    all_folder_values.extend(values)
                    domain_stats[domain_name] = {
                        'count': len(values),
                        'mean': np.mean(values)
                    }

            except Exception as e:
                continue

        # Customize individual subplot
        ax.set_xlabel('Measurement Index', fontsize=10)
        ax.set_ylabel('Response Time (ms)', fontsize=10)
        ax.set_title(f'{folder_name.upper()} - {len(csv_files)} Domain(s)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 15)  # Focus on successful operations

        # Add legend if there are multiple domains
        if len(csv_files) > 1:
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.legend(fontsize=9)

        # Store overall stats
        if all_folder_values:
            overall_stats[folder_name] = {
                'domains': len(csv_files),
                'operations': len(all_folder_values),
                'mean': np.mean(all_folder_values),
                'std': np.std(all_folder_values)
            }

    # Fourth subplot: Comparison histogram
    ax_comp = axes[1, 1]
    comp_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for folder_idx, folder_name in enumerate(folders):
        if folder_name in overall_stats:
            folder_path = f"data/{folder_name}"
            csv_files = glob.glob(f"{folder_path}/*.csv")

            all_values = []
            for csv_file in csv_files:
                with open(csv_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line != "No product found" and "No product found" not in line:
                            try:
                                value = float(line)
                                if value < 0.1:
                                    all_values.append(value * 1000)
                            except ValueError:
                                continue

            if all_values:
                ax_comp.hist(all_values, bins=30, alpha=0.7,
                             color=comp_colors[folder_idx],
                             label=f'{folder_name.upper()} (μ={np.mean(all_values):.1f}ms)',
                             density=True)

    ax_comp.set_xlabel('Response Time (ms)', fontsize=10)
    ax_comp.set_ylabel('Density', fontsize=10)
    ax_comp.set_title('Response Time Distribution Comparison', fontsize=12, fontweight='bold')
    ax_comp.legend(fontsize=9)
    ax_comp.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary stats
    print(f"\n{'=' * 80}")
    print("📊 SIDE-BY-SIDE COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    for folder, stats in overall_stats.items():
        print(f"\n🔹 {folder.upper()}:")
        print(f"   Domains: {stats['domains']}")
        print(f"   Total Operations: {stats['operations']:,}")
        print(f"   Mean Response Time: {stats['mean']:.2f} ms")
        print(f"   Std Deviation: {stats['std']:.2f} ms")


def plot_horizontal_comparison():
    """
    Plot all folders in a horizontal layout (1 row, 3 columns)
    """
    folders = ['nzt1', 'nzt3', 'nzt5']

    # Create horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Domain Performance - Horizontal Comparison', fontsize=16, fontweight='bold')

    domain_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#9B59B6']

    for folder_idx, folder_name in enumerate(folders):
        ax = axes[folder_idx]

        folder_path = f"data/{folder_name}"
        csv_files = glob.glob(f"{folder_path}/*.csv")

        if not csv_files:
            ax.text(0.5, 0.5, f'No data found\nfor {folder_name}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(f'{folder_name.upper()} - No Data', fontsize=14)
            continue

        all_values_for_stats = []

        # Process each CSV file
        for i, csv_file in enumerate(csv_files):
            filename = os.path.basename(csv_file)
            domain_name = filename.replace('.csv', '').replace('domain_', '').replace('_', '.')
            color = domain_colors[i % len(domain_colors)]

            times = []
            values = []

            with open(csv_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or "No product found" in line:
                        continue

                    try:
                        value = float(line)
                        if value < 0.1:  # Successful operations
                            times.append(line_num)
                            values.append(value * 1000)
                    except ValueError:
                        continue

            if values:
                ax.scatter(times, values,
                           c=color, alpha=0.7, s=3,
                           label=f'{domain_name}' if len(csv_files) > 1 else f'Domain ({len(values)} ops)')
                all_values_for_stats.extend(values)

        # Customize subplot
        ax.set_xlabel('Measurement Index', fontsize=11)
        ax.set_ylabel('Response Time (ms)', fontsize=11)

        # Dynamic title with stats
        if all_values_for_stats:
            mean_time = np.mean(all_values_for_stats)
            ax.set_title(
                f'{folder_name.upper()}\n{len(csv_files)} Domain(s) • μ={mean_time:.1f}ms • n={len(all_values_for_stats):,}',
                fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{folder_name.upper()}\n{len(csv_files)} Domain(s)', fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 12)

        # Smart legend placement
        if len(csv_files) > 1:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()


def analyze_folder_structure():
    """
    Analyze what's in your data folder
    """
    print("📁 FOLDER STRUCTURE ANALYSIS")
    print("=" * 50)

    if not os.path.exists("data"):
        print("❌ 'data' folder not found!")
        print("Make sure you're running this script from the right directory.")
        return

    folders = ['nzt1', 'nzt3', 'nzt5']

    for folder in folders:
        folder_path = f"data/{folder}"
        if os.path.exists(folder_path):
            csv_files = glob.glob(f"{folder_path}/*.csv")
            print(f"\n📂 {folder}/")
            print(f"   CSV files found: {len(csv_files)}")

            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                file_size = os.path.getsize(csv_file)
                print(f"   - {filename} ({file_size} bytes)")
        else:
            print(f"\n📂 {folder}/ - NOT FOUND")


if __name__ == "__main__":
    print("🚀 DOMAIN PERFORMANCE PLOTTER")
    print("=" * 50)

    # First, analyze the folder structure
    analyze_folder_structure()

    print("\n" + "=" * 50)
    print("Choose what to plot:")
    print("1. Plot nzt1 (1 domain)")
    print("2. Plot nzt3 (3 domains)")
    print("3. Plot nzt5 (5 domains)")
    print("4. 📊 Grid Layout (2x2) - Side by side + comparison")
    print("5. 📊 Horizontal Layout (1x3) - All folders side by side")
    print("6. Plot all individually")

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == "1":
        plot_domain_performance("nzt1")
    elif choice == "2":
        plot_domain_performance("nzt3")
    elif choice == "3":
        plot_domain_performance("nzt5")
    elif choice == "4":
        print("📊 Creating side-by-side grid layout...")
        plot_side_by_side_grid()
    elif choice == "5":
        print("📊 Creating horizontal comparison layout...")
        plot_horizontal_comparison()
    elif choice == "6":
        for folder in ['nzt1', 'nzt3', 'nzt5']:
            print(f"\n📊 Plotting {folder}...")
            plot_domain_performance(folder)
    else:
        print("Invalid choice. Creating grid layout by default...")
        plot_side_by_side_grid()