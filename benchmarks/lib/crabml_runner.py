"""crabML execution and output parsing."""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


class CrabMLRunner:
    """Run crabML and parse results."""

    def __init__(self, config: Dict):
        """
        Initialize crabML runner.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.crabml_exe = config['crabml']['executable']
        self.timeout = config['crabml']['timeout']
        self.maxiter = config['crabml'].get('maxiter', 1000)

    def run_analysis(
        self,
        seq_file: Path,
        tree_file: Path,
        model: str,
        output_dir: Path,
        replicate_id: int
    ) -> Dict[str, any]:
        """
        Run crabML analysis on a dataset.

        Parameters
        ----------
        seq_file : Path
            Sequence file (FASTA format)
        tree_file : Path
            Tree file (Newick format)
        model : str
            Model name (M0, M1a, etc.)
        output_dir : Path
            Directory for output files
        replicate_id : int
            Replicate number

        Returns
        -------
        dict
            Results including lnL, parameters, runtime
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output file for crabML results
        results_file = output_dir / f"rep{replicate_id:03d}_crabml.json"

        # Build command
        cmd = [
            self.crabml_exe,
            "fit",
            "-m", model,
            "-s", str(seq_file),
            "-t", str(tree_file),
            "--format", "json",  # Request JSON output
            "--maxiter", str(self.maxiter),
            "--quiet"  # Suppress progress output
        ]

        # Run crabML
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            runtime = time.time() - start_time

            if result.returncode != 0:
                # Non-zero exit code
                parsed = {
                    "lnL": None,
                    "converged": False,
                    "parameters": {},
                    "runtime": runtime,
                    "timeout": False,
                    "error": f"Exit code {result.returncode}: {result.stderr}"
                }
            else:
                # Parse JSON output from stdout
                # Extract JSON from output (may have progress messages before it)
                try:
                    stdout = result.stdout.strip()
                    # Find the JSON block (starts with { and ends with })
                    json_start = stdout.rfind('{')
                    json_end = stdout.rfind('}')
                    if json_start != -1 and json_end != -1:
                        json_str = stdout[json_start:json_end+1]
                        output_data = json.loads(json_str)
                        parsed = self.parse_output(output_data, model, runtime)
                    else:
                        raise ValueError("No JSON found in output")
                except (json.JSONDecodeError, ValueError) as e:
                    parsed = {
                        "lnL": None,
                        "converged": False,
                        "parameters": {},
                        "runtime": runtime,
                        "timeout": False,
                        "error": f"JSON parse error: {e}\nOutput: {result.stdout[:500]}"
                    }

        except subprocess.TimeoutExpired:
            runtime = self.timeout
            parsed = {
                "lnL": None,
                "converged": False,
                "parameters": {},
                "runtime": runtime,
                "timeout": True,
                "error": "Timeout"
            }

        except Exception as e:
            runtime = time.time() - start_time
            parsed = {
                "lnL": None,
                "converged": False,
                "parameters": {},
                "runtime": runtime,
                "timeout": False,
                "error": str(e)
            }

        # Save parsed results
        with open(results_file, 'w') as f:
            json.dump(parsed, f, indent=2)

        return parsed

    def parse_output(
        self,
        output_data: Dict,
        model: str,
        runtime: float
    ) -> Dict[str, any]:
        """
        Parse crabML JSON output.

        Parameters
        ----------
        output_data : dict
            Parsed JSON from crabML stdout
        model : str
            Model name
        runtime : float
            Execution time in seconds

        Returns
        -------
        dict
            Parsed results with lnL and parameters
        """
        results = {
            "lnL": None,
            "converged": False,
            "parameters": {},
            "runtime": runtime,
            "timeout": False
        }

        # Extract log-likelihood
        if "lnL" in output_data:
            results["lnL"] = float(output_data["lnL"])
            results["converged"] = True
        elif "log_likelihood" in output_data:
            results["lnL"] = float(output_data["log_likelihood"])
            results["converged"] = True

        # Extract parameters - crabML uses "params" (not "parameters")
        # and has kappa at top level
        params = output_data.get("params", {})

        # Common parameter: kappa (at top level in crabML output)
        if "kappa" in output_data:
            results["parameters"]["kappa"] = float(output_data["kappa"])

        # Model-specific parameters
        if model == "M0":
            if "omega" in params:
                results["parameters"]["omega"] = float(params["omega"])

        elif model == "M1a":
            # p0 and omegas
            if "p0" in params:
                results["parameters"]["p0"] = float(params["p0"])
            if "omega0" in params:
                results["parameters"]["omega0"] = float(params["omega0"])
            # omega1 is fixed to 1.0, may not be in output

        elif model == "M2a":
            # Proportions and omegas
            if "p0" in params:
                results["parameters"]["p0"] = float(params["p0"])
            if "p1" in params:
                results["parameters"]["p1"] = float(params["p1"])
            if "omega0" in params:
                results["parameters"]["omega0"] = float(params["omega0"])
            if "omega2" in params:
                results["parameters"]["omega2"] = float(params["omega2"])

        elif model == "M7":
            # Beta distribution parameters
            if "p" in params:
                results["parameters"]["p"] = float(params["p"])
            if "q" in params:
                results["parameters"]["q"] = float(params["q"])

        elif model in ["M8", "M8a"]:
            # p0, beta parameters, omega_s
            if "p0" in params:
                results["parameters"]["p0"] = float(params["p0"])
            if "p" in params:
                results["parameters"]["p"] = float(params["p"])
            if "q" in params:
                results["parameters"]["q"] = float(params["q"])
            if "omega_s" in params:
                results["parameters"]["omega_s"] = float(params["omega_s"])
            # M8a should have omega_s = 1.0

        # Check for convergence flag if provided
        if "converged" in output_data:
            results["converged"] = bool(output_data["converged"])

        return results
