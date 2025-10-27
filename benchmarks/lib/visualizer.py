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

        # Find parameters to plot
        param_cols = [col for col in model_df.columns if col.startswith("paml_")]
        params = [col.replace("paml_", "") for col in param_cols
                 if not col.endswith(("_lnL", "_converged", "_time", "_timeout"))]

        # Filter to parameters that exist in both
        params = [p for p in params
                 if f"paml_{p}" in model_df.columns and f"crabml_{p}" in model_df.columns]

        if len(params) == 0:
            return

        # Create subplots
        n_params = len(params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), dpi=self.dpi)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, param in enumerate(params):
            ax = axes[i]

            paml_vals = model_df[f"paml_{param}"].dropna().values
            crabml_vals = model_df[f"crabml_{param}"].dropna().values

            if len(paml_vals) == 0 or len(crabml_vals) == 0:
                continue

            # Scatter plot
            ax.scatter(paml_vals, crabml_vals, alpha=0.6, s=80, edgecolors='k', linewidths=0.5)

            # Diagonal line
            lim_min = min(paml_vals.min(), crabml_vals.min())
            lim_max = max(paml_vals.max(), crabml_vals.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2)

            # Calculate correlation
            if len(paml_vals) == len(crabml_vals) and len(paml_vals) > 1:
                corr = np.corrcoef(paml_vals, crabml_vals)[0, 1]
                rmse = np.sqrt(np.mean((paml_vals - crabml_vals) ** 2))
                ax.set_title(f'{param}\nr = {corr:.4f}, RMSE = {rmse:.4f}', fontsize=12)
            else:
                ax.set_title(param, fontsize=12)

            ax.set_xlabel(f'PAML {param}', fontsize=11)
            ax.set_ylabel(f'crabML {param}', fontsize=11)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{model}: Parameter Comparison', fontsize=16, y=1.00)
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
        Plot runtime comparison across models.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        models : List[str]
            Models to include
        output_dir : Path
            Output directory for plots
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)

        # Prepare data for boxplot
        converged_df = df[
            (df["paml_converged"] == True) &
            (df["crabml_converged"] == True)
        ]

        data = []
        labels = []

        for model in models:
            model_df = converged_df[converged_df["model"] == model]

            if len(model_df) > 0:
                paml_times = model_df["paml_time"].dropna().values
                crabml_times = model_df["crabml_time"].dropna().values

                if len(paml_times) > 0:
                    data.append(paml_times)
                    labels.append(f"{model}\nPAML")

                if len(crabml_times) > 0:
                    data.append(crabml_times)
                    labels.append(f"{model}\ncrabML")

        if len(data) > 0:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color boxes alternately
            for i, box in enumerate(bp['boxes']):
                if i % 2 == 0:
                    box.set_facecolor(self.colors['paml'])
                else:
                    box.set_facecolor(self.colors['crabml'])

            ax.set_ylabel('Runtime (seconds)', fontsize=14)
            ax.set_title('Runtime Comparison: PAML vs crabML', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=0, fontsize=10)

            plt.tight_layout()
            output_file = output_dir / "runtime_comparison.png"
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
