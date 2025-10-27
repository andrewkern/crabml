"""PAML execution and output parsing."""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class PAMLRunner:
    """Run PAML and parse results."""

    def __init__(self, config: Dict):
        """
        Initialize PAML runner.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.paml_exe = Path(config['paml']['executable']).resolve()
        self.timeout = config['paml']['timeout']

        if not self.paml_exe.exists():
            raise FileNotFoundError(f"PAML executable not found: {self.paml_exe}")

    def create_control_file(
        self,
        seq_file: Path,
        tree_file: Path,
        output_file: Path,
        model: str,
        ctl_file: Path
    ):
        """
        Create PAML control file from template.

        Parameters
        ----------
        seq_file : Path
            Sequence file path
        tree_file : Path
            Tree file path
        output_file : Path
            Output file path
        model : str
            Model name (M0, M1a, etc.)
        ctl_file : Path
            Control file path to create
        """
        # Model-specific NSsites codes
        model_codes = {
            "M0": 0,
            "M1a": 1,
            "M2a": 2,
            "M7": 7,
            "M8": 8,
            "M8a": 8,  # M8a uses NSsites=8 with fix_omega=1
        }

        nssite = model_codes.get(model, 0)

        # For M8a, we need fix_omega=1, omega=1
        fix_omega = 1 if model == "M8a" else 0
        omega_init = 1.0 if model == "M8a" else 0.4

        # Generate control file
        ctl_content = f"""      seqfile = {seq_file}
     treefile = {tree_file}
      outfile = {output_file}

        noisy = 0
      verbose = 0
      runmode = 0

      seqtype = 1
    CodonFreq = 2
        model = 0
      NSsites = {nssite}

        icode = 0
    fix_kappa = 0
        kappa = 2
    fix_omega = {fix_omega}
        omega = {omega_init}

    fix_alpha = 1
        alpha = 0
       Malpha = 0
        ncatG = 10

        clock = 0

        getSE = 0
 RateAncestor = 0

   Small_Diff = .5e-6
    cleandata = 0
       method = 0
"""

        with open(ctl_file, 'w') as f:
            f.write(ctl_content)

    def parse_output(self, output_file: Path, model: str) -> Dict[str, any]:
        """
        Parse PAML output file.

        Parameters
        ----------
        output_file : Path
            PAML output file
        model : str
            Model name

        Returns
        -------
        dict
            Parsed results with lnL and parameters
        """
        if not output_file.exists():
            return {"error": "Output file not found", "converged": False}

        with open(output_file, 'r') as f:
            content = f.read()

        results = {
            "lnL": None,
            "converged": False,
            "parameters": {},
        }

        # Parse log-likelihood
        lnl_match = re.search(r'lnL\(.*?\):\s+([-\d.]+)', content)
        if lnl_match:
            results["lnL"] = float(lnl_match.group(1))
            results["converged"] = True

        # Parse kappa
        kappa_match = re.search(r'kappa.*?=\s+([\d.]+)', content)
        if kappa_match:
            results["parameters"]["kappa"] = float(kappa_match.group(1))

        # Model-specific parameter parsing
        if model == "M0":
            omega_match = re.search(r'omega.*?=\s+([\d.]+)', content)
            if omega_match:
                results["parameters"]["omega"] = float(omega_match.group(1))

        elif model == "M1a":
            # Parse p0 and omegas
            # Look for site class proportions and omegas in the output
            prop_match = re.search(r'proportion\s+([\d.]+)\s+([\d.]+)', content)
            if prop_match:
                results["parameters"]["p0"] = float(prop_match.group(1))

            # Find omega values
            omega_matches = re.findall(r'w:\s+([\d.]+)', content)
            if len(omega_matches) >= 2:
                results["parameters"]["omega0"] = float(omega_matches[0])
                # omega1 should be 1.0 for M1a

        elif model == "M2a":
            # Parse proportions and omegas
            prop_match = re.search(r'proportion\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
            if prop_match:
                results["parameters"]["p0"] = float(prop_match.group(1))
                results["parameters"]["p1"] = float(prop_match.group(2))

            omega_matches = re.findall(r'w:\s+([\d.]+)', content)
            if len(omega_matches) >= 3:
                results["parameters"]["omega0"] = float(omega_matches[0])
                results["parameters"]["omega2"] = float(omega_matches[2])

        elif model == "M7":
            # Parse beta parameters p and q
            beta_match = re.search(r'p:\s+([\d.]+)\s+q:\s+([\d.]+)', content)
            if beta_match:
                results["parameters"]["p"] = float(beta_match.group(1))
                results["parameters"]["q"] = float(beta_match.group(2))

        elif model in ["M8", "M8a"]:
            # Parse p0, beta parameters, and omega_s
            prop_match = re.search(r'p0=\s+([\d.]+)', content)
            if prop_match:
                results["parameters"]["p0"] = float(prop_match.group(1))

            beta_match = re.search(r'p:\s+([\d.]+)\s+q:\s+([\d.]+)', content)
            if beta_match:
                results["parameters"]["p"] = float(beta_match.group(1))
                results["parameters"]["q"] = float(beta_match.group(2))

            # Find the additional omega class
            # Look for w: values, the last one should be omega_s
            omega_matches = re.findall(r'w:\s+([\d.]+)', content)
            if omega_matches:
                results["parameters"]["omega_s"] = float(omega_matches[-1])

        return results

    def run_analysis(
        self,
        seq_file: Path,
        tree_file: Path,
        model: str,
        output_dir: Path,
        replicate_id: int
    ) -> Dict[str, any]:
        """
        Run PAML analysis on a dataset.

        Parameters
        ----------
        seq_file : Path
            Sequence file
        tree_file : Path
            Tree file
        model : str
            Model name
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

        # Create control file
        ctl_file = output_dir / f"rep{replicate_id:03d}.ctl"
        mlc_file = output_dir / f"rep{replicate_id:03d}.mlc"

        self.create_control_file(seq_file, tree_file, mlc_file, model, ctl_file)

        # Run PAML
        start_time = time.time()
        try:
            # Run in the output directory to capture auxiliary files
            result = subprocess.run(
                [str(self.paml_exe), str(ctl_file.name)],
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            runtime = time.time() - start_time

            # Parse output
            parsed = self.parse_output(mlc_file, model)
            parsed["runtime"] = runtime
            parsed["timeout"] = False

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
        results_file = output_dir / f"rep{replicate_id:03d}_results.json"
        with open(results_file, 'w') as f:
            json.dump(parsed, f, indent=2)

        return parsed
