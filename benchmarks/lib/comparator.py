"""Statistical comparison of PAML and crabML results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


class BenchmarkComparator:
    """Compare PAML and crabML results."""

    def __init__(self, config: Dict):
        """
        Initialize comparator.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.outlier_sigma = config['thresholds']['outlier_sigma']

    def load_results(
        self,
        data_dir: Path,
        paml_dir: Path,
        crabml_dir: Path,
        models: List[str]
    ) -> pd.DataFrame:
        """
        Load and match PAML and crabML results.

        Parameters
        ----------
        data_dir : Path
            Base directory for simulated data
        paml_dir : Path
            Directory with PAML results
        crabml_dir : Path
            Directory with crabML results
        models : List[str]
            Models to compare

        Returns
        -------
        pd.DataFrame
            Combined results table
        """
        rows = []

        for model in models:
            model_data_dir = data_dir / model
            model_paml_dir = paml_dir / model
            model_crabml_dir = crabml_dir / model

            # Find all replicates
            param_files = sorted(model_data_dir.glob("rep*.params.json"))

            for param_file in param_files:
                rep_id = int(param_file.stem.split('.')[0].replace('rep', ''))

                # Load simulation metadata
                with open(param_file, 'r') as f:
                    metadata = json.load(f)

                # Load PAML results
                paml_file = model_paml_dir / f"rep{rep_id:03d}_results.json"
                if paml_file.exists():
                    with open(paml_file, 'r') as f:
                        paml_results = json.load(f)
                else:
                    paml_results = None

                # Load crabML results
                crabml_file = model_crabml_dir / f"rep{rep_id:03d}_crabml.json"
                if crabml_file.exists():
                    with open(crabml_file, 'r') as f:
                        crabml_results = json.load(f)
                else:
                    crabml_results = None

                # Build row
                row = {
                    "model": model,
                    "replicate": rep_id,
                    "tree_id": metadata["tree_id"],
                    "seq_length": metadata["sequence_length"],
                }

                # Add true parameters
                for param, value in metadata["true_parameters"].items():
                    row[f"true_{param}"] = value

                # Add PAML results
                if paml_results:
                    row["paml_lnL"] = paml_results.get("lnL")
                    row["paml_converged"] = paml_results.get("converged", False)
                    row["paml_time"] = paml_results.get("runtime")
                    row["paml_timeout"] = paml_results.get("timeout", False)
                    for param, value in paml_results.get("parameters", {}).items():
                        row[f"paml_{param}"] = value
                else:
                    row["paml_lnL"] = None
                    row["paml_converged"] = False
                    row["paml_time"] = None
                    row["paml_timeout"] = False

                # Add crabML results
                if crabml_results:
                    row["crabml_lnL"] = crabml_results.get("lnL")
                    row["crabml_converged"] = crabml_results.get("converged", False)
                    row["crabml_time"] = crabml_results.get("runtime")
                    row["crabml_timeout"] = crabml_results.get("timeout", False)
                    for param, value in crabml_results.get("parameters", {}).items():
                        row[f"crabml_{param}"] = value
                else:
                    row["crabml_lnL"] = None
                    row["crabml_converged"] = False
                    row["crabml_time"] = None
                    row["crabml_timeout"] = False

                rows.append(row)

        return pd.DataFrame(rows)

    def calculate_statistics(
        self,
        df: pd.DataFrame,
        model: str
    ) -> Dict[str, any]:
        """
        Calculate comparison statistics for a model.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        model : str
            Model name

        Returns
        -------
        dict
            Summary statistics
        """
        model_df = df[df["model"] == model]
        n_replicates = len(model_df)

        # Filter to cases where both converged
        both_converged = model_df[
            (model_df["paml_converged"] == True) &
            (model_df["crabml_converged"] == True)
        ]
        n_converged_both = len(both_converged)

        if n_converged_both == 0:
            return {
                "n_replicates": n_replicates,
                "n_converged_both": 0,
                "error": "No replicates where both converged"
            }

        stats_dict = {
            "n_replicates": n_replicates,
            "n_converged_both": n_converged_both,
            "paml_convergence_rate": model_df["paml_converged"].sum() / n_replicates,
            "crabml_convergence_rate": model_df["crabml_converged"].sum() / n_replicates,
        }

        # Log-likelihood comparison
        paml_lnL = both_converged["paml_lnL"].values
        crabml_lnL = both_converged["crabml_lnL"].values

        if len(paml_lnL) > 1:
            stats_dict["lnL_correlation"] = np.corrcoef(paml_lnL, crabml_lnL)[0, 1]
            stats_dict["lnL_rmse"] = np.sqrt(np.mean((paml_lnL - crabml_lnL) ** 2))
            stats_dict["lnL_mean_abs_diff"] = np.mean(np.abs(paml_lnL - crabml_lnL))
            stats_dict["lnL_max_abs_diff"] = np.max(np.abs(paml_lnL - crabml_lnL))

            # Outlier detection
            diff = np.abs(paml_lnL - crabml_lnL)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            outliers = diff > (mean_diff + self.outlier_sigma * std_diff)
            stats_dict["lnL_n_outliers"] = int(np.sum(outliers))

        # Parameter comparisons
        param_cols = [col for col in both_converged.columns if col.startswith("paml_")]
        param_cols = [col.replace("paml_", "") for col in param_cols
                     if not col.endswith(("_lnL", "_converged", "_time", "_timeout"))]

        for param in param_cols:
            paml_col = f"paml_{param}"
            crabml_col = f"crabml_{param}"

            if paml_col in both_converged.columns and crabml_col in both_converged.columns:
                paml_vals = both_converged[paml_col].dropna().values
                crabml_vals = both_converged[crabml_col].dropna().values

                if len(paml_vals) > 1 and len(crabml_vals) > 1 and len(paml_vals) == len(crabml_vals):
                    corr = np.corrcoef(paml_vals, crabml_vals)[0, 1]
                    rmse = np.sqrt(np.mean((paml_vals - crabml_vals) ** 2))
                    mean_abs_diff = np.mean(np.abs(paml_vals - crabml_vals))

                    stats_dict[f"{param}_correlation"] = corr
                    stats_dict[f"{param}_rmse"] = rmse
                    stats_dict[f"{param}_mean_abs_diff"] = mean_abs_diff

        # Runtime comparison
        paml_times = both_converged["paml_time"].dropna().values
        crabml_times = both_converged["crabml_time"].dropna().values

        if len(paml_times) > 0 and len(crabml_times) > 0:
            speedups = paml_times / crabml_times
            stats_dict["crabml_speedup_median"] = float(np.median(speedups))
            stats_dict["crabml_speedup_mean"] = float(np.mean(speedups))
            stats_dict["paml_time_median"] = float(np.median(paml_times))
            stats_dict["crabml_time_median"] = float(np.median(crabml_times))

        return stats_dict

    def compare_all(
        self,
        data_dir: Path,
        paml_dir: Path,
        crabml_dir: Path,
        models: List[str],
        output_dir: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Compare all models and save results.

        Parameters
        ----------
        data_dir : Path
            Base directory for simulated data
        paml_dir : Path
            Directory with PAML results
        crabml_dir : Path
            Directory with crabML results
        models : List[str]
            Models to compare
        output_dir : Path
            Directory for output files

        Returns
        -------
        tuple
            (comparison_df, summary_stats)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all results
        print("Loading results...")
        df = self.load_results(data_dir, paml_dir, crabml_dir, models)

        # Save comparison table
        comparison_file = output_dir / "comparison.csv"
        df.to_csv(comparison_file, index=False)
        print(f"Saved comparison table to {comparison_file}")

        # Calculate statistics per model
        print("\nCalculating statistics...")
        summary_stats = {}
        for model in models:
            print(f"  {model}...")
            stats_dict = self.calculate_statistics(df, model)
            summary_stats[model] = stats_dict

        # Save summary statistics
        summary_file = output_dir / "summary_stats.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary statistics to {summary_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        for model, stats in summary_stats.items():
            print(f"\n{model}:")
            print(f"  Replicates: {stats['n_replicates']}")
            print(f"  Both converged: {stats['n_converged_both']}")

            if "lnL_correlation" in stats:
                print(f"  lnL correlation: {stats['lnL_correlation']:.6f}")
                print(f"  lnL RMSE: {stats['lnL_rmse']:.4f}")
                print(f"  lnL mean |diff|: {stats['lnL_mean_abs_diff']:.4f}")
                print(f"  lnL max |diff|: {stats['lnL_max_abs_diff']:.4f}")
                print(f"  lnL outliers: {stats['lnL_n_outliers']}")

            if "crabml_speedup_median" in stats:
                print(f"  crabML speedup (median): {stats['crabml_speedup_median']:.2f}x")

        return df, summary_stats
