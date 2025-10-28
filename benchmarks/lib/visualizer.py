"""Visualization of benchmark results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class BenchmarkVisualizer:
    """Create visualizations of PAML vs crabML comparisons."""

    def __init__(self, config: Dict):
        """
        Initialize visualizer.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.viz_config = config['visualization']
        self.dpi = self.viz_config['dpi']
        self.figsize = tuple(self.viz_config['figsize'])
        self.style = self.viz_config['style']
        self.colors = self.viz_config['colors']

        # Set style
        try:
            plt.style.use(self.style)
        except:
            # Fallback if style not available
            sns.set_theme(style="darkgrid")

    def plot_likelihood_comparison(
        self,
        df: pd.DataFrame,
        model: str,
        output_dir: Path
    ):
        """
        Plot PAML vs crabML log-likelihoods for a model.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        model : str
            Model name
        output_dir : Path
            Output directory for plots
        """
        model_df = df[
            (df["model"] == model) &
            (df["paml_converged"] == True) &
            (df["crabml_converged"] == True)
        ]

        if len(model_df) == 0:
            print(f"  No converged replicates for {model}")
            return

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        paml_lnL = model_df["paml_lnL"].values
        crabml_lnL = model_df["crabml_lnL"].values

        # Scatter plot
        ax.scatter(paml_lnL, crabml_lnL, alpha=0.6, s=80, edgecolors='k', linewidths=0.5)

        # Diagonal line (perfect agreement)
        lim_min = min(paml_lnL.min(), crabml_lnL.min())
        lim_max = max(paml_lnL.max(), crabml_lnL.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2,
                label='Perfect agreement')

        # Calculate correlation
        corr = np.corrcoef(paml_lnL, crabml_lnL)[0, 1]
        rmse = np.sqrt(np.mean((paml_lnL - crabml_lnL) ** 2))

        ax.set_xlabel('PAML lnL', fontsize=14)
        ax.set_ylabel('crabML lnL', fontsize=14)
        ax.set_title(f'{model}: Log-Likelihood Comparison\n' +
                    f'r = {corr:.6f}, RMSE = {rmse:.4f}',
                    fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / f"{model}_lnL.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_file}")

    def plot_parameter_comparison(
        self,
        df: pd.DataFrame,
        model: str,
        output_dir: Path
    ):
        """
        Plot parameter comparisons for a model.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        model : str
            Model name
        output_dir : Path
            Output directory for plots
        """
        model_df = df[
            (df["model"] == model) &
            (df["paml_converged"] == True) &
            (df["crabml_converged"] == True)
        ]

        if len(model_df) == 0:
            return

        # Find parameters to plot (those with true values)
        param_cols = [col for col in model_df.columns if col.startswith("true_")]
        params = [col.replace("true_", "") for col in param_cols
                 if not col.endswith(("_lnL", "_converged", "_time", "_timeout"))]

        # Filter to parameters that exist in true, paml, and crabml
        params = [p for p in params
                 if (f"true_{p}" in model_df.columns and
                     f"paml_{p}" in model_df.columns and
                     f"crabml_{p}" in model_df.columns and
                     model_df[f"true_{p}"].notna().any())]

        if len(params) == 0:
            return

        # Create figure with 3 columns per parameter
        # Col 1: PAML vs crabML (implementation agreement)
        # Col 2: True vs Estimates (accuracy)
        # Col 3: Relative error distribution
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 3, figsize=(18, 5 * n_params), dpi=self.dpi)

        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(1, -1)

        for i, param in enumerate(params):
            # Get values
            true_vals = model_df[f"true_{param}"].dropna().values
            paml_vals = model_df[f"paml_{param}"].dropna().values
            crabml_vals = model_df[f"crabml_{param}"].dropna().values

            # Ensure we have matching indices for error calculation
            valid_mask = (model_df[f"true_{param}"].notna() &
                         model_df[f"paml_{param}"].notna() &
                         model_df[f"crabml_{param}"].notna())

            true_vals_matched = model_df.loc[valid_mask, f"true_{param}"].values
            paml_vals_matched = model_df.loc[valid_mask, f"paml_{param}"].values
            crabml_vals_matched = model_df.loc[valid_mask, f"crabml_{param}"].values

            # Column 1: PAML vs crabML scatter
            ax1 = axes[i, 0]
            if len(paml_vals) > 0 and len(crabml_vals) > 0:
                ax1.scatter(paml_vals, crabml_vals, alpha=0.6, s=60,
                           edgecolors='k', linewidths=0.5, color=self.colors['paml'])

                lim_min = min(paml_vals.min(), crabml_vals.min())
                lim_max = max(paml_vals.max(), crabml_vals.max())
                ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1.5)

                if len(paml_vals) > 1:
                    corr = np.corrcoef(paml_vals, crabml_vals)[0, 1]
                    rmse = np.sqrt(np.mean((paml_vals - crabml_vals) ** 2))
                    ax1.set_title(f'{param}: PAML vs crabML\nr={corr:.5f}, RMSE={rmse:.4f}',
                                 fontsize=11, fontweight='bold')

                ax1.set_xlabel('PAML', fontsize=10)
                ax1.set_ylabel('crabML', fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Column 2: True vs Estimates
            ax2 = axes[i, 1]
            if len(true_vals_matched) > 0:
                # Plot both estimates vs true
                ax2.scatter(true_vals_matched, paml_vals_matched, alpha=0.5, s=50,
                           label='PAML', color=self.colors['paml'], edgecolors='k', linewidths=0.5)
                ax2.scatter(true_vals_matched, crabml_vals_matched, alpha=0.5, s=50,
                           label='crabML', color=self.colors['crabml'], edgecolors='k', linewidths=0.5)

                # Diagonal line (perfect estimation)
                all_vals = np.concatenate([true_vals_matched, paml_vals_matched, crabml_vals_matched])
                lim_min, lim_max = all_vals.min(), all_vals.max()
                ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1.5)

                ax2.set_xlabel(f'True {param}', fontsize=10)
                ax2.set_ylabel(f'Estimated {param}', fontsize=10)
                ax2.set_title(f'{param}: Estimation Accuracy', fontsize=11, fontweight='bold')
                ax2.legend(fontsize=9, loc='best')
                ax2.grid(True, alpha=0.3)

            # Column 3: Relative error distribution
            ax3 = axes[i, 2]
            if len(true_vals_matched) > 0:
                # Calculate relative errors: (estimated - true) / true
                paml_rel_err = (paml_vals_matched - true_vals_matched) / true_vals_matched
                crabml_rel_err = (crabml_vals_matched - true_vals_matched) / true_vals_matched

                # Create boxplot
                positions = [1, 2]
                bp = ax3.boxplot([paml_rel_err, crabml_rel_err], positions=positions,
                                widths=0.6, patch_artist=True,
                                labels=['PAML', 'crabML'])

                # Color boxes
                bp['boxes'][0].set_facecolor(self.colors['paml'])
                bp['boxes'][1].set_facecolor(self.colors['crabml'])

                # Style the median lines to be visible (dark red)
                for median in bp['medians']:
                    median.set_color('darkred')
                    median.set_linewidth(2.5)

                # Style whiskers and caps
                for whisker in bp['whiskers']:
                    whisker.set_color('black')
                    whisker.set_linewidth(1.5)
                    whisker.set_linestyle('-')

                for cap in bp['caps']:
                    cap.set_color('black')
                    cap.set_linewidth(1.5)

                # Style outlier markers
                for flier in bp['fliers']:
                    flier.set_marker('o')
                    flier.set_markerfacecolor('red')
                    flier.set_markeredgecolor('darkred')
                    flier.set_markersize(5)
                    flier.set_alpha(0.6)

                # Add zero line
                ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)

                # Calculate stats
                paml_median = np.median(paml_rel_err)
                crabml_median = np.median(crabml_rel_err)

                ax3.set_ylabel('Relative Error\n(estimated - true) / true', fontsize=10)
                ax3.set_title(f'{param}: Relative Error\nMedian: PAML={paml_median:.3f}, crabML={crabml_median:.3f}',
                             fontsize=11, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{model}: Parameter Estimation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / f"{model}_params.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_file}")

    def plot_runtime_comparison(
        self,
        df: pd.DataFrame,
        models: List[str],
        output_dir: Path
    ):
        """
        Plot runtime comparison as speedup ratios.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        models : List[str]
            Models to include
        output_dir : Path
            Output directory for plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)

        # Prepare data
        converged_df = df[
            (df["paml_converged"] == True) &
            (df["crabml_converged"] == True)
        ]

        # Left plot: Speedup ratios (PAML time / crabML time)
        speedup_data = []
        model_labels = []
        median_speedups = []

        for model in models:
            model_df = converged_df[converged_df["model"] == model]

            if len(model_df) > 0:
                paml_times = model_df["paml_time"].dropna().values
                crabml_times = model_df["crabml_time"].dropna().values

                if len(paml_times) > 0 and len(crabml_times) > 0:
                    speedups = paml_times / crabml_times
                    speedup_data.append(speedups)
                    model_labels.append(model)
                    median_speedups.append(np.median(speedups))

        if len(speedup_data) > 0:
            # Boxplot of speedup ratios
            bp = ax1.boxplot(speedup_data, labels=model_labels, patch_artist=True)

            # Color boxes
            for box in bp['boxes']:
                box.set_facecolor(self.colors['crabml'])
                box.set_alpha(0.7)

            # Style median lines
            for median in bp['medians']:
                median.set_color('darkred')
                median.set_linewidth(2.5)

            # Add horizontal line at y=1 (no speedup)
            ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='No speedup')

            ax1.set_ylabel('Speedup (PAML time / crabML time)', fontsize=12)
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_title('crabML Speedup vs PAML', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend(fontsize=10)

            # Right plot: Absolute runtimes
            positions = np.arange(len(model_labels))
            width = 0.35

            paml_medians = []
            crabml_medians = []

            for model in model_labels:
                model_df = converged_df[converged_df["model"] == model]
                paml_medians.append(np.median(model_df["paml_time"].dropna().values))
                crabml_medians.append(np.median(model_df["crabml_time"].dropna().values))

            bars1 = ax2.bar(positions - width/2, paml_medians, width, label='PAML',
                           color=self.colors['paml'], edgecolor='k', linewidth=0.5)
            bars2 = ax2.bar(positions + width/2, crabml_medians, width, label='crabML',
                           color=self.colors['crabml'], edgecolor='k', linewidth=0.5)

            ax2.set_ylabel('Median Runtime (seconds)', fontsize=12)
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_title('Absolute Runtimes', fontsize=14, fontweight='bold')
            ax2.set_xticks(positions)
            ax2.set_xticklabels(model_labels, fontsize=10)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

            # Use log scale if values span orders of magnitude
            if max(paml_medians) / min(crabml_medians) > 10:
                ax2.set_yscale('log')

            plt.tight_layout()
            output_file = output_dir / "runtime_comparison.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"  Saved {output_file}")

    def plot_aggregated_likelihood_comparison(
        self,
        df: pd.DataFrame,
        models: List[str],
        output_dir: Path
    ):
        """
        Plot aggregated likelihood comparison across all models.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        models : List[str]
            Models to include
        output_dir : Path
            Output directory for plots
        """
        # Filter to converged results
        converged_df = df[
            (df["paml_converged"] == True) &
            (df["crabml_converged"] == True)
        ]

        if len(converged_df) == 0:
            print("  No converged results for aggregated plot")
            return

        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        # Create color palette for models
        model_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        color_map = {model: color for model, color in zip(models, model_colors)}

        # Plot each model with different color
        for model in models:
            model_df = converged_df[converged_df["model"] == model]

            if len(model_df) > 0:
                paml_lnL = model_df["paml_lnL"].values
                crabml_lnL = model_df["crabml_lnL"].values

                ax.scatter(paml_lnL, crabml_lnL,
                          alpha=0.6, s=100,
                          color=color_map[model],
                          edgecolors='k', linewidths=0.5,
                          label=model)

        # Diagonal line (perfect agreement)
        all_paml = converged_df["paml_lnL"].values
        all_crabml = converged_df["crabml_lnL"].values

        lim_min = min(all_paml.min(), all_crabml.min())
        lim_max = max(all_paml.max(), all_crabml.max())

        ax.plot([lim_min, lim_max], [lim_min, lim_max],
               'k--', alpha=0.5, linewidth=2.5,
               label='Perfect agreement')

        # Calculate overall statistics
        corr = np.corrcoef(all_paml, all_crabml)[0, 1]
        rmse = np.sqrt(np.mean((all_paml - all_crabml) ** 2))

        ax.set_xlabel('PAML lnL', fontsize=14)
        ax.set_ylabel('crabML lnL', fontsize=14)
        ax.set_title(f'Aggregated Log-Likelihood Comparison\n' +
                    f'All Models (n={len(converged_df)})\n' +
                    f'r = {corr:.8f}, RMSE = {rmse:.4f}',
                    fontsize=16, fontweight='bold')

        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for square plot
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        output_file = output_dir / "aggregated_lnL_comparison.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_file}")

    def plot_convergence_rates(
        self,
        summary_stats: Dict[str, Dict],
        output_dir: Path
    ):
        """
        Plot convergence rates for PAML and crabML.

        Parameters
        ----------
        summary_stats : dict
            Summary statistics dictionary
        output_dir : Path
            Output directory for plots
        """
        models = list(summary_stats.keys())
        paml_rates = [summary_stats[m].get("paml_convergence_rate", 0) for m in models]
        crabml_rates = [summary_stats[m].get("crabml_convergence_rate", 0) for m in models]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, paml_rates, width, label='PAML',
                      color=self.colors['paml'], edgecolor='k', linewidth=0.5)
        bars2 = ax.bar(x + width/2, crabml_rates, width, label='crabML',
                      color=self.colors['crabml'], edgecolor='k', linewidth=0.5)

        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Convergence Rate', fontsize=14)
        ax.set_title('Convergence Rates: PAML vs crabML', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        output_file = output_dir / "convergence_rates.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_file}")

    def plot_all(
        self,
        df: pd.DataFrame,
        summary_stats: Dict[str, Dict],
        models: List[str],
        output_dir: Path
    ):
        """
        Generate all visualizations.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        summary_stats : dict
            Summary statistics
        models : List[str]
            Models to visualize
        output_dir : Path
            Output directory for plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        # Per-model likelihood comparisons
        print("\nLikelihood comparisons:")
        for model in models:
            self.plot_likelihood_comparison(df, model, output_dir)

        # Aggregated likelihood comparison
        print("\nAggregated likelihood comparison:")
        self.plot_aggregated_likelihood_comparison(df, models, output_dir)

        # Per-model parameter comparisons
        print("\nParameter comparisons:")
        for model in models:
            self.plot_parameter_comparison(df, model, output_dir)

        # Runtime comparison
        print("\nRuntime comparison:")
        self.plot_runtime_comparison(df, models, output_dir)

        # Convergence rates
        print("\nConvergence rates:")
        self.plot_convergence_rates(summary_stats, output_dir)

        print(f"\nAll visualizations saved to {output_dir}/")
