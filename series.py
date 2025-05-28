import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import time
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


class NetworkSimulationAnalyzer:
    """Enhanced analyzer for network simulation data to support thesis on scalability"""

    def __init__(self, base_path: str = "data", lower_limit_ms: Optional[float] = None,
                 upper_limit_ms: Optional[float] = None):
        self.base_path = Path(base_path)
        self.lower_limit_ms = lower_limit_ms  # User-defined lower boundary
        self.upper_limit_ms = upper_limit_ms  # User-defined upper boundary
        self.simulations = {}
        self.colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD']

    def set_lower_limit(self, limit_ms: float):
        """Set the lower boundary limit for data filtering"""
        self.lower_limit_ms = limit_ms
        print(f"Lower limit set to {limit_ms} ms")

    def set_upper_limit(self, limit_ms: float):
        """Set the upper boundary limit for data filtering"""
        self.upper_limit_ms = limit_ms
        print(f"Upper limit set to {limit_ms} ms")

    def set_limits(self, lower_ms: float, upper_ms: float):
        """Set both lower and upper boundary limits for data filtering"""
        self.lower_limit_ms = lower_ms
        self.upper_limit_ms = upper_ms
        print(f"Limits set to {lower_ms} - {upper_ms} ms")

    def load_simulation_data(self, short) -> Dict:
        """Load all simulation data from organized directory structure"""
        print("Loading simulation data...")

        # Expected structure: zt1, zt3, zt5 directories
        simulation_dirs = [short + '1', short + '3', short + '5']

        for sim_dir in simulation_dirs:
            sim_path = self.base_path / sim_dir
            if not sim_path.exists():
                print(f"Warning: Directory {sim_path} not found")
                continue

            # Extract number of active domains from directory name
            num_domains = int(sim_dir.replace(short, ''))

            # Load all CSV files in this simulation directory
            csv_files = list(sim_path.glob("*.csv"))

            if not csv_files:
                print(f"Warning: No CSV files found in {sim_path}")
                continue

            simulation_data = []
            domain_names = []

            for csv_file in csv_files:
                try:
                    # Load the CSV file
                    df = pd.read_csv(csv_file, header=None)
                    if len(df.columns) > 0:
                        # Convert to milliseconds and store
                        times_ms = df.iloc[:, 0].values * 1000
                        simulation_data.append(times_ms)
                        domain_names.append(csv_file.stem)
                        print(f"  Loaded {csv_file.name}: {len(times_ms):,} data points")
                except Exception as e:
                    print(f"  Error loading {csv_file.name}: {e}")

            if simulation_data:
                self.simulations[num_domains] = {
                    'data': simulation_data,
                    'domain_names': domain_names,
                    'path': sim_path
                }

        print(f"Loaded {len(self.simulations)} simulations")
        return self.simulations

    def _filter_data(self, data: np.ndarray) -> np.ndarray:
        """Filter data using both lower and upper boundary limits"""
        filtered_data = data.copy()

        # Apply lower limit if set
        if self.lower_limit_ms is not None:
            filtered_data = filtered_data[filtered_data >= self.lower_limit_ms]

        # Apply upper limit if set
        if self.upper_limit_ms is not None:
            filtered_data = filtered_data[filtered_data <= self.upper_limit_ms]

        return filtered_data

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics for each simulation (without standard deviation)"""
        stats_summary = {}

        for num_domains, sim_data in self.simulations.items():
            domain_stats = []
            combined_data = []

            for i, domain_data in enumerate(sim_data['data']):
                # Apply boundary filtering (both upper and lower limits if set)
                original_count = len(domain_data)
                cleaned_data = self._filter_data(domain_data)
                filtered_count = original_count - len(cleaned_data)

                if len(cleaned_data) == 0:
                    print(f"Warning: No data remaining for {sim_data['domain_names'][i]} after filtering")
                    continue

                # Calculate statistics (excluding standard deviation)
                domain_stat = {
                    'domain_name': sim_data['domain_names'][i],
                    'count': len(cleaned_data),
                    'mean': np.mean(cleaned_data),
                    'median': np.median(cleaned_data),
                    'min': np.min(cleaned_data),
                    'max': np.max(cleaned_data),
                    'q25': np.percentile(cleaned_data, 25),
                    'q75': np.percentile(cleaned_data, 75),
                    'q90': np.percentile(cleaned_data, 90),
                    'q95': np.percentile(cleaned_data, 95),
                    'filtered_out': filtered_count
                }
                domain_stats.append(domain_stat)
                combined_data.extend(cleaned_data)

            # Overall simulation statistics
            if combined_data:
                stats_summary[num_domains] = {
                    'domains': domain_stats,
                    'overall': {
                        'count': len(combined_data),
                        'mean': np.mean(combined_data),
                        'median': np.median(combined_data),
                        'min': np.min(combined_data),
                        'max': np.max(combined_data),
                        'q25': np.percentile(combined_data, 25),
                        'q75': np.percentile(combined_data, 75),
                        'q90': np.percentile(combined_data, 90),
                        'q95': np.percentile(combined_data, 95),
                        'range': np.max(combined_data) - np.min(combined_data)
                    }
                }

        return stats_summary

    def print_statistics_summary(self, statistics: Dict):
        """Print a comprehensive statistics summary"""
        print("\n" + "=" * 80)
        print("NETWORK SIMULATION STATISTICS SUMMARY")
        if self.lower_limit_ms or self.upper_limit_ms:
            limits = []
            if self.lower_limit_ms:
                limits.append(f"Lower: {self.lower_limit_ms} ms")
            if self.upper_limit_ms:
                limits.append(f"Upper: {self.upper_limit_ms} ms")
            print(f"Data filters - {', '.join(limits)}")
        print("=" * 80)

        # Overall comparison table
        print(
            f"\n{'Domains':<8} {'Mean (ms)':<12} {'Median (ms)':<14} {'Range (ms)':<12} {'Q95 (ms)':<10} {'Count':<10}")
        print("-" * 70)

        for num_domains in sorted(statistics.keys()):
            overall = statistics[num_domains]['overall']
            print(f"{num_domains:<8} {overall['mean']:<12.2f} {overall['median']:<14.2f} "
                  f"{overall['range']:<12.2f} {overall['q95']:<10.2f} {overall['count']:<10,}")

        # Detailed breakdown for each simulation
        for num_domains in sorted(statistics.keys()):
            print(f"\n{num_domains} Active Domains - Detailed Breakdown:")
            print("-" * 50)

            for domain in statistics[num_domains]['domains']:
                print(f"  {domain['domain_name']}:")
                print(f"    Mean: {domain['mean']:.2f} ms, Median: {domain['median']:.2f} ms")
                print(f"    Range: {domain['min']:.2f} - {domain['max']:.2f} ms")
                print(f"    Count: {domain['count']:,}, Filtered out: {domain['filtered_out']:,}")

    def create_comparison_plots(self, statistics: Dict, output_dir: str = "images"):
        """Create comprehensive comparison plots for thesis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Time series comparison plot
        self._create_time_series_plot(output_path)

        # 2. Box plot comparison
        self._create_box_plot_comparison(statistics, output_path)

        # 3. Distribution comparison
        self._create_distribution_comparison(output_path)

        # 4. Scalability analysis
        self._create_scalability_analysis(statistics, output_path)

        # 5. Statistical summary table
        self._create_summary_table(statistics, output_path)

    def _create_time_series_plot(self, output_path: Path):
        """Create time series plot showing stability over time using line-by-line averages"""
        fig, axes = plt.subplots(len(self.simulations), 1,
                                 figsize=(14, 4 * len(self.simulations)),
                                 sharex=False)

        if len(self.simulations) == 1:
            axes = [axes]

        for idx, (num_domains, sim_data) in enumerate(sorted(self.simulations.items())):
            ax = axes[idx]

            # First, filter all domain data and find the minimum length
            filtered_domain_data = []
            for domain_data in sim_data['data']:
                cleaned_data = self._filter_data(domain_data)
                if len(cleaned_data) > 0:
                    filtered_domain_data.append(cleaned_data)

            if not filtered_domain_data:
                print(f"Warning: No valid data for {num_domains} domains after filtering")
                continue

            # Find the minimum length across all domains
            min_length = min(len(data) for data in filtered_domain_data)
            print(
                f"Processing {num_domains} domains: {len(filtered_domain_data)} files, min length: {min_length:,} points")

            # Calculate line-by-line averages
            averaged_data = []

            # Process in chunks to handle large datasets efficiently
            chunk_size = 10_000  # Process 10k lines at a time

            for start_idx in range(0, min_length, chunk_size):
                end_idx = min(start_idx + chunk_size, min_length)

                # Create array to hold chunk data from all files
                chunk_data = np.zeros((len(filtered_domain_data), end_idx - start_idx))

                # Fill chunk data
                for file_idx, domain_data in enumerate(filtered_domain_data):
                    chunk_data[file_idx, :] = domain_data[start_idx:end_idx]

                # Calculate mean across all files for each time point
                chunk_averages = np.mean(chunk_data, axis=0)
                averaged_data.extend(chunk_averages)

            # Convert to numpy array
            averaged_data = np.array(averaged_data)

            # Create indices for plotting
            indices = np.arange(len(averaged_data))

            # Downsample for plotting if still too large
            if len(averaged_data) > 50_000:
                step = len(averaged_data) // 50_000
                averaged_data = averaged_data[::step]
                indices = indices[::step]
                print(f"  Downsampled to {len(averaged_data):,} points for visualization")

            # Plot the averaged data
            ax.scatter(indices, averaged_data, s=1, alpha=0.6,
                       color=self.colors[idx % len(self.colors)])

            # Add statistical lines
            mean_val = np.mean(averaged_data)
            median_val = np.median(averaged_data)
            ax.axhline(y=mean_val, color='green', linestyle='-', alpha=0.7,
                       label=f'Mean: {mean_val:.2f} ms')
            ax.axhline(y=median_val, color='orange', linestyle='--', alpha=0.7,
                       label=f'Median: {median_val:.2f} ms')

            # Add limit lines if set
            if self.lower_limit_ms:
                ax.axhline(y=self.lower_limit_ms, color='red', linestyle=':', alpha=0.7,
                           label=f'Lower Limit: {self.lower_limit_ms:.0f} ms')
            if self.upper_limit_ms:
                ax.axhline(y=self.upper_limit_ms, color='purple', linestyle=':', alpha=0.7,
                           label=f'Upper Limit: {self.upper_limit_ms:.0f} ms')

            ax.set_title(f'{num_domains} Active Domains - Averaged Travel Time Stability '
                         f'({len(filtered_domain_data)} files, n={len(averaged_data):,})',
                         fontweight='bold')
            ax.set_ylabel('Average Travel Time (ms)')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Time Index (Line Number)')
        plt.suptitle('Network Travel Time Stability Analysis - Line-by-Line Averages',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'time_series_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'time_series_stability.png'}")

    def _create_box_plot_comparison(self, statistics: Dict, output_path: Path):
        """Create box plot comparison across simulations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare data for box plots
        domain_data = []
        domain_labels = []
        simulation_data = []
        simulation_labels = []

        for num_domains, sim_data in sorted(self.simulations.items()):
            # Collect individual domain data
            for i, domain_data_raw in enumerate(sim_data['data']):
                cleaned = self._filter_data(domain_data_raw)

                if len(cleaned) > 0:
                    domain_data.append(cleaned)
                    domain_labels.append(f"{num_domains}D-{sim_data['domain_names'][i]}")

            # Collect combined simulation data
            combined = []
            for domain_data_raw in sim_data['data']:
                cleaned = self._filter_data(domain_data_raw)
                combined.extend(cleaned)

            if len(combined) > 0:
                simulation_data.append(combined)
                simulation_labels.append(f"{num_domains} Domains")

        # Individual domains box plot
        if domain_data:
            bp1 = ax1.boxplot(domain_data, labels=domain_labels, patch_artist=True)
            for i, patch in enumerate(bp1['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
                patch.set_alpha(0.7)

        ax1.set_title('Travel Time Distribution by Domain', fontweight='bold')
        ax1.set_ylabel('Travel Time (ms)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Simulation comparison box plot
        if simulation_data:
            bp2 = ax2.boxplot(simulation_data, labels=simulation_labels, patch_artist=True)
            for i, patch in enumerate(bp2['boxes']):
                patch.set_facecolor(self.colors[i])
                patch.set_alpha(0.7)

        ax2.set_title('Travel Time Distribution by Load Level', fontweight='bold')
        ax2.set_ylabel('Travel Time (ms)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'boxplot_comparison.png'}")

    def _create_distribution_comparison(self, output_path: Path):
        """Create distribution comparison plot"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (num_domains, sim_data) in enumerate(sorted(self.simulations.items())):
            # Combine all data for this simulation
            combined_data = []
            for domain_data in sim_data['data']:
                cleaned_data = self._filter_data(domain_data)
                combined_data.extend(cleaned_data)

            if len(combined_data) > 0:
                # Create histogram/density plot
                ax.hist(combined_data, bins=50, alpha=0.6,
                        label=f'{num_domains} Domains (n={len(combined_data):,})',
                        color=self.colors[i], density=True)

        # Add limit lines if set
        if self.lower_limit_ms:
            ax.axvline(x=self.lower_limit_ms, color='red', linestyle=':', alpha=0.7,
                       label=f'Lower Limit: {self.lower_limit_ms:.0f} ms')
        if self.upper_limit_ms:
            ax.axvline(x=self.upper_limit_ms, color='purple', linestyle=':', alpha=0.7,
                       label=f'Upper Limit: {self.upper_limit_ms:.0f} ms')

        ax.set_xlabel('Travel Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title('Travel Time Distribution Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'distribution_comparison.png'}")

    def _create_scalability_analysis(self, statistics: Dict, output_path: Path):
        """Create scalability summary plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Extract data for plotting
        domains = sorted(statistics.keys())
        means = [statistics[d]['overall']['mean'] for d in domains]
        medians = [statistics[d]['overall']['median'] for d in domains]
        ranges = [statistics[d]['overall']['range'] for d in domains]

        # Create twin axis for range (since it likely has different scale)
        ax_twin = ax.twinx()

        # Plot the metrics
        line1 = ax.plot(domains, means, 'o-', linewidth=2, markersize=8,
                        color=self.colors[0], label='Mean')
        line2 = ax.plot(domains, medians, 's--', linewidth=2, markersize=8,
                        color=self.colors[3], label='Median')
        line3 = ax_twin.plot(domains, ranges, '^:', linewidth=2, markersize=8,
                             color=self.colors[2], label='Range')

        # Set labels and title
        ax.set_xlabel('Number of Active Domains')
        ax.set_ylabel('Travel Time (ms)', color='black')
        ax_twin.set_ylabel('Range (ms)', color=self.colors[2])
        ax.set_title('Scalability Summary', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_path / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'scalability_analysis.png'}")

    def _create_summary_table(self, statistics: Dict, output_path: Path):
        """Create a summary table image"""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = []
        headers = ['Active Domains', 'Mean (ms)', 'Median (ms)', 'Range (ms)',
                   'Q75 (ms)', 'Q95 (ms)', 'Min (ms)', 'Max (ms)', 'Sample Size']

        for num_domains in sorted(statistics.keys()):
            overall = statistics[num_domains]['overall']
            row = [
                f"{num_domains}",
                f"{overall['mean']:.2f}",
                f"{overall['median']:.2f}",
                f"{overall['range']:.2f}",
                f"{overall['q75']:.2f}",
                f"{overall['q95']:.2f}",
                f"{overall['min']:.2f}",
                f"{overall['max']:.2f}",
                f"{overall['count']:,}"
            ]
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.11] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        title = 'Network Performance Summary Statistics'
        if self.lower_limit_ms or self.upper_limit_ms:
            limits = []
            if self.lower_limit_ms:
                limits.append(f"Lower: {self.lower_limit_ms} ms")
            if self.upper_limit_ms:
                limits.append(f"Upper: {self.upper_limit_ms} ms")
            title += f' ({", ".join(limits)})'

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'summary_table.png'}")

    def perform_statistical_tests(self, statistics: Dict):
        """Perform statistical tests to support thesis claims"""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS FOR THESIS")
        if self.lower_limit_ms or self.upper_limit_ms:
            limits = []
            if self.lower_limit_ms:
                limits.append(f"Lower: {self.lower_limit_ms} ms")
            if self.upper_limit_ms:
                limits.append(f"Upper: {self.upper_limit_ms} ms")
            print(f"Data filtered with limits - {', '.join(limits)}")
        print("=" * 80)

        # Test for significant differences between simulations
        simulation_pairs = []
        for i, num_domains_i in enumerate(sorted(statistics.keys())):
            for j, num_domains_j in enumerate(sorted(statistics.keys())):
                if i < j:
                    simulation_pairs.append((num_domains_i, num_domains_j))

        print("\nStatistical Tests for Performance Differences:")
        print("-" * 50)

        for sim1, sim2 in simulation_pairs:
            # Collect data for both simulations
            data1 = []
            data2 = []

            for domain_data in self.simulations[sim1]['data']:
                filtered_data = self._filter_data(domain_data)
                data1.extend(filtered_data)

            for domain_data in self.simulations[sim2]['data']:
                filtered_data = self._filter_data(domain_data)
                data2.extend(filtered_data)

            if len(data1) == 0 or len(data2) == 0:
                print(f"Skipping comparison {sim1} vs {sim2}: insufficient data after filtering")
                continue

            # Perform Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # Effect size using median difference
            median1, median2 = np.median(data1), np.median(data2)
            median_diff = abs(median1 - median2)

            # Mean difference for comparison
            mean1, mean2 = np.mean(data1), np.mean(data2)
            mean_diff = abs(mean1 - mean2)

            print(f"{sim1} vs {sim2} domains:")
            print(f"  Mann-Whitney U p-value: {p_value:.6f}")
            print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} difference")
            print(f"  Mean difference: {mean_diff:.2f} ms")
            print(f"  Median difference: {median_diff:.2f} ms")
            print()

        # Trend analysis
        print("Trend Analysis:")
        print("-" * 50)
        domains = sorted(statistics.keys())
        means = [statistics[d]['overall']['mean'] for d in domains]
        medians = [statistics[d]['overall']['median'] for d in domains]

        # Linear regression for mean trend
        slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = stats.linregress(domains, means)

        # Linear regression for median trend
        slope_median, intercept_median, r_value_median, p_value_median, std_err_median = stats.linregress(domains,
                                                                                                          medians)

        print(f"Linear trend in mean travel time:")
        print(f"  Slope: {slope_mean:.4f} ms per additional domain")
        print(f"  R²: {r_value_mean ** 2:.4f}")
        print(f"  P-value: {p_value_mean:.6f}")
        print(f"  Interpretation: {'Significant' if p_value_mean < 0.05 else 'No significant'} trend")

        print(f"\nLinear trend in median travel time:")
        print(f"  Slope: {slope_median:.4f} ms per additional domain")
        print(f"  R²: {r_value_median ** 2:.4f}")
        print(f"  P-value: {p_value_median:.6f}")
        print(f"  Interpretation: {'Significant' if p_value_median < 0.05 else 'No significant'} trend")

        if abs(slope_mean) < 0.1:  # Arbitrary threshold for "stable"
            print(f"  ✓ THESIS SUPPORT: Mean slope is very small ({slope_mean:.4f}), indicating stable performance")
        else:
            print(f"  ⚠ ATTENTION: Mean slope of {slope_mean:.4f} may indicate performance change")

        if abs(slope_median) < 0.1:
            print(f"  ✓ THESIS SUPPORT: Median slope is very small ({slope_median:.4f}), indicating stable performance")
        else:
            print(f"  ⚠ ATTENTION: Median slope of {slope_median:.4f} may indicate performance change")


def main():
    """Main analysis function with customizable upper and lower limits"""
    print("Enhanced Network Simulation Analysis for Thesis")
    print("=" * 50)

    # Initialize analyzer - you can set both limits here or later
    analyzer = NetworkSimulationAnalyzer("data", lower_limit_ms=None, upper_limit_ms=5)

    # Example: Set individual limits (uncomment and modify as needed)
    # analyzer.set_lower_limit(5.0)   # Only include data points >= 5ms
    # analyzer.set_upper_limit(100.0) # Only include data points <= 100ms

    # Example: Set both limits at once (uncomment and modify as needed)
    # analyzer.set_limits(10.0, 150.0)  # Only include data between 10-150ms

    # Load all simulation data
    simulations = analyzer.load_simulation_data(short="zt")

    if not simulations:
        print("No simulation data found. Please check your data directory structure.")
        return

    # Calculate statistics
    print("\nCalculating statistics...")
    statistics = analyzer.calculate_statistics()

    # Print summary
    analyzer.print_statistics_summary(statistics)

    # Perform statistical tests
    analyzer.perform_statistical_tests(statistics)

    # Create all plots
    print("\nGenerating plots...")
    analyzer.create_comparison_plots(statistics)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Generated files:")
    print("- time_series_stability.png: Shows stability over time")
    print("- boxplot_comparison.png: Distribution comparison")
    print("- distribution_comparison.png: Density plots")
    print("- scalability_analysis.png: Performance vs load analysis")
    print("- summary_table.png: Statistical summary table")
    print("\nKey improvements:")
    print("- Removed all standard deviation calculations and displays")
    print("- Added customizable upper AND lower boundary filtering")
    print("- Replaced coefficient of variation with range and percentiles")
    print("- Enhanced statistical analysis with median trends")


if __name__ == "__main__":
    main()