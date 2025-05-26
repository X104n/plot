import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import time
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple
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
    """Analyzer for network simulation data to support thesis on scalability"""

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.simulations = {}
        self.colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD']

    def load_simulation_data(self) -> Dict:
        """Load all simulation data from organized directory structure"""
        print("Loading simulation data...")

        # Expected structure: nzt1, nzt3, nzt5 directories
        simulation_dirs = ['nzt1', 'nzt3', 'nzt5']

        for sim_dir in simulation_dirs:
            sim_path = self.base_path / sim_dir
            if not sim_path.exists():
                print(f"Warning: Directory {sim_path} not found")
                continue

            # Extract number of active domains from directory name
            num_domains = int(sim_dir.replace('nzt', ''))

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

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics for each simulation"""
        stats_summary = {}

        for num_domains, sim_data in self.simulations.items():
            domain_stats = []
            combined_data = []

            for i, domain_data in enumerate(sim_data['data']):
                # Remove outliers using IQR method
                q1, q3 = np.percentile(domain_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                cleaned_data = domain_data[(domain_data >= lower_bound) & (domain_data <= upper_bound)]

                # Calculate statistics
                domain_stat = {
                    'domain_name': sim_data['domain_names'][i],
                    'count': len(cleaned_data),
                    'mean': np.mean(cleaned_data),
                    'median': np.median(cleaned_data),
                    'std': np.std(cleaned_data),
                    'min': np.min(cleaned_data),
                    'max': np.max(cleaned_data),
                    'q25': np.percentile(cleaned_data, 25),
                    'q75': np.percentile(cleaned_data, 75),
                    'outliers_removed': len(domain_data) - len(cleaned_data)
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
                        'std': np.std(combined_data),
                        'min': np.min(combined_data),
                        'max': np.max(combined_data),
                        'cv': np.std(combined_data) / np.mean(combined_data) * 100  # Coefficient of variation
                    }
                }

        return stats_summary

    def print_statistics_summary(self, statistics: Dict):
        """Print a comprehensive statistics summary"""
        print("\n" + "=" * 80)
        print("NETWORK SIMULATION STATISTICS SUMMARY")
        print("=" * 80)

        # Overall comparison table
        print(f"\n{'Domains':<8} {'Mean (ms)':<12} {'Median (ms)':<14} {'Std Dev':<12} {'CV (%)':<8} {'Count':<10}")
        print("-" * 70)

        for num_domains in sorted(statistics.keys()):
            overall = statistics[num_domains]['overall']
            print(f"{num_domains:<8} {overall['mean']:<12.2f} {overall['median']:<14.2f} "
                  f"{overall['std']:<12.2f} {overall['cv']:<8.2f} {overall['count']:<10,}")

        # Detailed breakdown for each simulation
        for num_domains in sorted(statistics.keys()):
            print(f"\n{num_domains} Active Domains - Detailed Breakdown:")
            print("-" * 50)

            for domain in statistics[num_domains]['domains']:
                print(f"  {domain['domain_name']}:")
                print(f"    Mean: {domain['mean']:.2f} ms, Median: {domain['median']:.2f} ms")
                print(f"    Count: {domain['count']:,}, Outliers removed: {domain['outliers_removed']:,}")

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
        """Create time series plot showing stability over time"""
        fig, axes = plt.subplots(len(self.simulations), 1,
                                 figsize=(14, 4 * len(self.simulations)),
                                 sharex=False)

        if len(self.simulations) == 1:
            axes = [axes]

        for idx, (num_domains, sim_data) in enumerate(sorted(self.simulations.items())):
            ax = axes[idx]

            # Combine all domain data for this simulation
            all_data = []
            all_indices = []
            current_idx = 0

            for domain_data in sim_data['data']:
                # Clean data
                q1, q3 = np.percentile(domain_data, [25, 75])
                iqr = q3 - q1
                mask = (domain_data >= q1 - 1.5 * iqr) & (domain_data <= q3 + 1.5 * iqr)
                cleaned_data = domain_data[mask]

                # Downsample for plotting
                if len(cleaned_data) > 5000:
                    step = len(cleaned_data) // 5000
                    cleaned_data = cleaned_data[::step]

                indices = np.arange(current_idx, current_idx + len(cleaned_data))
                all_data.extend(cleaned_data)
                all_indices.extend(indices)
                current_idx += len(cleaned_data)

            # Plot the data
            ax.scatter(all_indices, all_data, s=1, alpha=0.6,
                       color=self.colors[idx % len(self.colors)])

            # Add trend line
            if len(all_data) > 1:
                z = np.polyfit(all_indices, all_data, 1)
                p = np.poly1d(z)
                ax.plot(all_indices, p(all_indices), "r--", alpha=0.8, linewidth=2,
                        label=f'Trend (slope: {z[0]:.4f})')

            # Add mean line
            mean_val = np.mean(all_data)
            ax.axhline(y=mean_val, color='green', linestyle='-', alpha=0.7,
                       label=f'Mean: {mean_val:.2f} ms')

            ax.set_title(f'{num_domains} Active Domains - Travel Time Stability',
                         fontweight='bold')
            ax.set_ylabel('Travel Time (ms)')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Measurement Index')
        plt.suptitle('Network Travel Time Stability Analysis', fontsize=16, fontweight='bold')
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
                # Clean data
                q1, q3 = np.percentile(domain_data_raw, [25, 75])
                iqr = q3 - q1
                mask = (domain_data_raw >= q1 - 1.5 * iqr) & (domain_data_raw <= q3 + 1.5 * iqr)
                cleaned = domain_data_raw[mask]

                domain_data.append(cleaned)
                domain_labels.append(f"{num_domains}D-{sim_data['domain_names'][i]}")

            # Collect combined simulation data
            combined = []
            for domain_data_raw in sim_data['data']:
                q1, q3 = np.percentile(domain_data_raw, [25, 75])
                iqr = q3 - q1
                mask = (domain_data_raw >= q1 - 1.5 * iqr) & (domain_data_raw <= q3 + 1.5 * iqr)
                combined.extend(domain_data_raw[mask])

            simulation_data.append(combined)
            simulation_labels.append(f"{num_domains} Domains")

        # Individual domains box plot
        bp1 = ax1.boxplot(domain_data, labels=domain_labels, patch_artist=True)
        for i, patch in enumerate(bp1['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)

        ax1.set_title('Travel Time Distribution by Domain', fontweight='bold')
        ax1.set_ylabel('Travel Time (ms)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Simulation comparison box plot
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
                # Clean data
                q1, q3 = np.percentile(domain_data, [25, 75])
                iqr = q3 - q1
                mask = (domain_data >= q1 - 1.5 * iqr) & (domain_data <= q3 + 1.5 * iqr)
                combined_data.extend(domain_data[mask])

            # Create histogram/density plot
            ax.hist(combined_data, bins=50, alpha=0.6,
                    label=f'{num_domains} Domains (n={len(combined_data):,})',
                    color=self.colors[i], density=True)

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
        """Create scalability analysis plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data for plotting
        domains = sorted(statistics.keys())
        means = [statistics[d]['overall']['mean'] for d in domains]
        medians = [statistics[d]['overall']['median'] for d in domains]
        stds = [statistics[d]['overall']['std'] for d in domains]
        cvs = [statistics[d]['overall']['cv'] for d in domains]

        # Mean travel time vs domains
        ax1.plot(domains, means, 'o-', linewidth=2, markersize=8, color=self.colors[0])
        ax1.set_xlabel('Number of Active Domains')
        ax1.set_ylabel('Mean Travel Time (ms)')
        ax1.set_title('Mean Travel Time vs Load', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Standard deviation vs domains
        ax2.plot(domains, stds, 'o-', linewidth=2, markersize=8, color=self.colors[1])
        ax2.set_xlabel('Number of Active Domains')
        ax2.set_ylabel('Standard Deviation (ms)')
        ax2.set_title('Variability vs Load', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Coefficient of variation vs domains
        ax3.plot(domains, cvs, 'o-', linewidth=2, markersize=8, color=self.colors[2])
        ax3.set_xlabel('Number of Active Domains')
        ax3.set_ylabel('Coefficient of Variation (%)')
        ax3.set_title('Relative Variability vs Load', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Combined plot
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(domains, means, 'o-', linewidth=2, markersize=8,
                         color=self.colors[0], label='Mean')
        line2 = ax4_twin.plot(domains, cvs, 's--', linewidth=2, markersize=8,
                              color=self.colors[2], label='CV (%)')

        ax4.set_xlabel('Number of Active Domains')
        ax4.set_ylabel('Mean Travel Time (ms)', color=self.colors[0])
        ax4_twin.set_ylabel('Coefficient of Variation (%)', color=self.colors[2])
        ax4.set_title('Scalability Summary', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_path / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'scalability_analysis.png'}")

    def _create_summary_table(self, statistics: Dict, output_path: Path):
        """Create a summary table image"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = []
        headers = ['Active Domains', 'Mean (ms)', 'Median (ms)', 'Std Dev (ms)',
                   'CV (%)', 'Min (ms)', 'Max (ms)', 'Sample Size']

        for num_domains in sorted(statistics.keys()):
            overall = statistics[num_domains]['overall']
            row = [
                f"{num_domains}",
                f"{overall['mean']:.2f}",
                f"{overall['median']:.2f}",
                f"{overall['std']:.2f}",
                f"{overall['cv']:.2f}",
                f"{overall['min']:.2f}",
                f"{overall['max']:.2f}",
                f"{overall['count']:,}"
            ]
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.12] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Network Performance Summary Statistics',
                  fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'summary_table.png'}")

    def perform_statistical_tests(self, statistics: Dict):
        """Perform statistical tests to support thesis claims"""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS FOR THESIS")
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
                q1, q3 = np.percentile(domain_data, [25, 75])
                iqr = q3 - q1
                mask = (domain_data >= q1 - 1.5 * iqr) & (domain_data <= q3 + 1.5 * iqr)
                data1.extend(domain_data[mask])

            for domain_data in self.simulations[sim2]['data']:
                q1, q3 = np.percentile(domain_data, [25, 75])
                iqr = q3 - q1
                mask = (domain_data >= q1 - 1.5 * iqr) & (domain_data <= q3 + 1.5 * iqr)
                data2.extend(domain_data[mask])

            # Perform Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # Effect size (Cohen's d approximation)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std_pooled = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            cohens_d = (mean1 - mean2) / std_pooled if std_pooled > 0 else 0

            print(f"{sim1} vs {sim2} domains:")
            print(f"  Mann-Whitney U p-value: {p_value:.6f}")
            print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
            print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} difference")
            print(f"  Mean difference: {abs(mean1 - mean2):.2f} ms")
            print()

        # Trend analysis
        print("Trend Analysis:")
        print("-" * 50)
        domains = sorted(statistics.keys())
        means = [statistics[d]['overall']['mean'] for d in domains]

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(domains, means)

        print(f"Linear trend in mean travel time:")
        print(f"  Slope: {slope:.4f} ms per additional domain")
        print(f"  R²: {r_value ** 2:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'No significant'} trend")

        if abs(slope) < 0.1:  # Arbitrary threshold for "stable"
            print(f"  ✓ THESIS SUPPORT: Slope is very small ({slope:.4f}), indicating stable performance")
        else:
            print(f"  ⚠ ATTENTION: Slope of {slope:.4f} may indicate performance degradation")


def main():
    """Main analysis function"""
    print("Network Simulation Analysis for Thesis")
    print("=" * 50)

    # Initialize analyzer
    analyzer = NetworkSimulationAnalyzer("data")  # Adjust path as needed

    # Load all simulation data
    simulations = analyzer.load_simulation_data()

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
    print("\nThese plots should support your thesis that the network")
    print("implementation maintains stable performance regardless of load.")


if __name__ == "__main__":
    main()