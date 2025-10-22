"""
Result objects for hypothesis tests.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from .lrt import calculate_lrt

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class LRTResult:
    """
    Results from a likelihood ratio test for positive selection.

    Attributes
    ----------
    test_name : str
        Name of the test (e.g., "M1a vs M2a")
    null_model : str
        Name of the null model
    alt_model : str
        Name of the alternative model
    lnL_null : float
        Log-likelihood of the null model
    lnL_alt : float
        Log-likelihood of the alternative model
    df : int
        Degrees of freedom (difference in number of parameters)
    null_params : dict
        Optimized parameters from null model
    alt_params : dict
        Optimized parameters from alternative model
    null_optimization_success : bool
        Whether null model optimization converged
    alt_optimization_success : bool
        Whether alternative model optimization converged
    """

    test_name: str
    null_model: str
    alt_model: str
    lnL_null: float
    lnL_alt: float
    df: int
    null_params: Dict[str, Any]
    alt_params: Dict[str, Any]
    null_optimization_success: bool = True
    alt_optimization_success: bool = True
    _override_lrt: Optional[float] = None
    _override_pvalue: Optional[float] = None

    @property
    def LRT(self) -> float:
        """Likelihood ratio test statistic: 2 * (lnL_alt - lnL_null)"""
        if self._override_lrt is not None:
            return self._override_lrt
        return 2 * (self.lnL_alt - self.lnL_null)

    @property
    def pvalue(self) -> float:
        """P-value from chi-square distribution (or mixture distribution if overridden)"""
        if self._override_pvalue is not None:
            return self._override_pvalue
        _, pval = calculate_lrt(self.lnL_null, self.lnL_alt, self.df)
        return pval

    def significant(self, alpha: float = 0.05) -> bool:
        """
        Test if result is statistically significant.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        bool
            True if p-value < alpha
        """
        return self.pvalue < alpha

    @property
    def omega_positive(self) -> Optional[float]:
        """
        The omega value for the positive selection class (if present).

        Returns
        -------
        float or None
            Omega > 1 value from alternative model, or None if not applicable
        """
        # Extract omega values from alternative model parameters
        if 'omega_2' in self.alt_params:
            return self.alt_params['omega_2']
        elif 'omega_s' in self.alt_params:
            return self.alt_params['omega_s']
        return None

    @property
    def proportion_positive(self) -> Optional[float]:
        """
        Proportion of sites under positive selection.

        Returns
        -------
        float or None
            Proportion of sites in omega > 1 class, or None if not applicable
        """
        # Extract proportion from alternative model parameters
        if 'p2' in self.alt_params:
            # M2a model
            return self.alt_params['p2']
        elif 'p0' in self.alt_params:
            # M8 model: (1 - p0) is the proportion in omega > 1 class
            return 1.0 - self.alt_params['p0']
        return None

    def summary(self) -> str:
        """
        Generate a formatted summary of the test results.

        Returns
        -------
        str
            Multi-line formatted summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Likelihood Ratio Test for Positive Selection")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Test: {self.test_name}")
        lines.append("")

        # Convergence warnings
        if not self.null_optimization_success or not self.alt_optimization_success:
            lines.append("⚠ WARNING: Optimization may not have converged properly")
            if not self.null_optimization_success:
                lines.append(f"  - {self.null_model} optimization did not converge")
            if not self.alt_optimization_success:
                lines.append(f"  - {self.alt_model} optimization did not converge")
            lines.append("")

        # Null model
        lines.append(f"NULL MODEL ({self.null_model}):")
        lines.append(f"  Log-likelihood: {self.lnL_null:.6f}")
        lines.append("  Parameters:")
        for key, value in self.null_params.items():
            if isinstance(value, float):
                lines.append(f"    {key} = {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"    {key}:")
                for k, v in value.items():
                    lines.append(f"      {k} = {v:.4f}")
        lines.append("")

        # Alternative model
        lines.append(f"ALTERNATIVE MODEL ({self.alt_model}):")
        lines.append(f"  Log-likelihood: {self.lnL_alt:.6f}")
        lines.append("  Parameters:")
        for key, value in self.alt_params.items():
            if isinstance(value, float):
                lines.append(f"    {key} = {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"      {key}:")
                for k, v in value.items():
                    lines.append(f"      {k} = {v:.4f}")
        lines.append("")

        # Test statistics
        lines.append("LIKELIHOOD RATIO TEST:")
        lines.append(f"  LRT statistic: {self.LRT:.4f}")
        lines.append(f"  Degrees of freedom: {self.df}")
        lines.append(f"  P-value: {self.pvalue:.4f}")
        lines.append("")

        # Conclusion
        if self.significant(0.05):
            lines.append("CONCLUSION:")
            lines.append("  ✓ Significant evidence for positive selection (α = 0.05)")
        elif self.significant(0.10):
            lines.append("CONCLUSION:")
            lines.append("  ~ Marginal evidence for positive selection (0.05 < p < 0.10)")
        else:
            lines.append("CONCLUSION:")
            lines.append("  ✗ No significant evidence for positive selection (α = 0.05)")

        lines.append("")

        # Interpretation
        if self.omega_positive is not None and self.proportion_positive is not None:
            lines.append("INTERPRETATION:")
            if self.significant(0.05):
                lines.append(
                    f"  The alternative model ({self.alt_model}) fits significantly better "
                    f"than the null ({self.null_model})."
                )
                lines.append(
                    f"  Approximately {self.proportion_positive*100:.1f}% of sites show "
                    f"ω = {self.omega_positive:.2f} > 1, indicating positive selection."
                )
            else:
                lines.append(
                    f"  Although {self.proportion_positive*100:.1f}% of sites show "
                    f"ω = {self.omega_positive:.2f} > 1,"
                )
                lines.append(
                    f"  this improvement over the null model is not statistically significant."
                )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export results as a dictionary.

        Returns
        -------
        dict
            Dictionary containing all test results
        """
        return {
            'test_name': self.test_name,
            'null_model': self.null_model,
            'alt_model': self.alt_model,
            'lnL_null': float(self.lnL_null),
            'lnL_alt': float(self.lnL_alt),
            'LRT': float(self.LRT),
            'df': int(self.df),
            'pvalue': float(self.pvalue),
            'significant_0.05': bool(self.significant(0.05)),
            'significant_0.01': bool(self.significant(0.01)),
            'omega_positive': float(self.omega_positive) if self.omega_positive is not None else None,
            'proportion_positive': float(self.proportion_positive) if self.proportion_positive is not None else None,
            'null_params': self.null_params,
            'alt_params': self.alt_params,
            'null_optimization_success': bool(self.null_optimization_success),
            'alt_optimization_success': bool(self.alt_optimization_success),
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Export results as JSON.

        Parameters
        ----------
        filepath : str, optional
            If provided, write JSON to this file
        indent : int, default=2
            Indentation level for pretty printing

        Returns
        -------
        str
            JSON string representation

        Examples
        --------
        >>> result.to_json('results.json')
        >>> json_str = result.to_json()
        """
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, indent=indent)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def to_markdown_table(self) -> str:
        """
        Export results as a markdown table suitable for papers.

        Returns
        -------
        str
            Markdown formatted table

        Examples
        --------
        >>> print(result.to_markdown_table())
        | Model | Log-likelihood | Parameters | LRT | P-value | Significant |
        |-------|----------------|------------|-----|---------|-------------|
        | M1a   | -902.504       | ...        | ... | ...     | ...         |
        """
        lines = []
        lines.append(f"### {self.test_name}")
        lines.append("")
        lines.append("| Model | Log-likelihood | LRT | df | P-value | Significant (α=0.05) |")
        lines.append("|-------|----------------|-----|----|---------|--------------------|")

        # Null model row
        lines.append(f"| {self.null_model} (null) | {self.lnL_null:.6f} | — | — | — | — |")

        # Alternative model row
        sig_symbol = "✓" if self.significant(0.05) else "✗"
        lines.append(
            f"| {self.alt_model} (alternative) | {self.lnL_alt:.6f} | "
            f"{self.LRT:.4f} | {self.df} | {self.pvalue:.4f} | {sig_symbol} |"
        )

        lines.append("")

        # Add parameter summary if available
        if self.omega_positive is not None and self.proportion_positive is not None:
            lines.append(f"**Positive selection class:** ω = {self.omega_positive:.4f}, "
                        f"proportion = {self.proportion_positive:.2%}")

        return "\n".join(lines)

    def to_csv_row(self, include_header: bool = False) -> str:
        """
        Export results as CSV row(s).

        Parameters
        ----------
        include_header : bool, default=False
            If True, include header row

        Returns
        -------
        str
            CSV formatted string

        Examples
        --------
        >>> print(result.to_csv_row(include_header=True))
        test_name,null_model,alt_model,lnL_null,lnL_alt,LRT,df,pvalue,significant
        M1a vs M2a,M1a,M2a,-902.504,-899.999,5.011,2,0.0817,False
        """
        result_dict = self.to_dict()

        # Select key fields for CSV
        fields = [
            'test_name', 'null_model', 'alt_model',
            'lnL_null', 'lnL_alt', 'LRT', 'df', 'pvalue',
            'significant_0.05', 'omega_positive', 'proportion_positive'
        ]

        lines = []
        if include_header:
            lines.append(','.join(fields))

        values = [str(result_dict.get(f, '')) for f in fields]
        lines.append(','.join(values))

        return '\n'.join(lines)

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Export results as a pandas DataFrame (single row).

        Returns
        -------
        pd.DataFrame
            DataFrame with one row containing all test results

        Raises
        ------
        ImportError
            If pandas is not installed

        Examples
        --------
        >>> df = result.to_dataframe()
        >>> df.to_csv('results.csv', index=False)
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        """String representation shows summary"""
        return self.summary()


def compare_results(results: List[LRTResult], format: str = 'table') -> str:
    """
    Compare multiple LRT results side-by-side.

    Parameters
    ----------
    results : list of LRTResult
        Results to compare
    format : str, default='table'
        Output format: 'table', 'markdown', or 'csv'

    Returns
    -------
    str
        Formatted comparison table

    Examples
    --------
    >>> from crabml.analysis import test_positive_selection
    >>> results = test_positive_selection('data.fa', 'tree.nwk', test='both')
    >>> from crabml.analysis.results import compare_results
    >>> print(compare_results([results['M1a_vs_M2a'], results['M7_vs_M8']]))
    """
    if format == 'markdown':
        return _compare_results_markdown(results)
    elif format == 'csv':
        return _compare_results_csv(results)
    else:
        return _compare_results_table(results)


def _compare_results_table(results: List[LRTResult]) -> str:
    """Format comparison as plain text table."""
    lines = []
    lines.append("=" * 100)
    lines.append("COMPARISON OF POSITIVE SELECTION TESTS")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(f"{'Test':<15} {'Null lnL':>12} {'Alt lnL':>12} {'LRT':>8} {'df':>4} {'P-value':>10} {'Sig?':>6}")
    lines.append("-" * 100)

    # Data rows
    for result in results:
        sig = "✓" if result.significant(0.05) else "✗"
        lines.append(
            f"{result.test_name:<15} "
            f"{result.lnL_null:>12.6f} "
            f"{result.lnL_alt:>12.6f} "
            f"{result.LRT:>8.4f} "
            f"{result.df:>4d} "
            f"{result.pvalue:>10.6f} "
            f"{sig:>6}"
        )

    lines.append("")

    # Summary
    sig_count = sum(1 for r in results if r.significant(0.05))
    lines.append(f"Summary: {sig_count}/{len(results)} tests significant at α=0.05")

    if sig_count == len(results):
        lines.append("Conclusion: Strong evidence for positive selection (all tests significant)")
    elif sig_count > 0:
        lines.append("Conclusion: Moderate evidence for positive selection (some tests significant)")
    else:
        lines.append("Conclusion: No significant evidence for positive selection")

    lines.append("=" * 100)

    return "\n".join(lines)


def _compare_results_markdown(results: List[LRTResult]) -> str:
    """Format comparison as markdown table."""
    lines = []
    lines.append("## Comparison of Positive Selection Tests")
    lines.append("")
    lines.append("| Test | Null lnL | Alt lnL | LRT | df | P-value | Significant (α=0.05) |")
    lines.append("|------|----------|---------|-----|----|---------|--------------------|")

    for result in results:
        sig = "✓" if result.significant(0.05) else "✗"
        lines.append(
            f"| {result.test_name} | {result.lnL_null:.6f} | {result.lnL_alt:.6f} | "
            f"{result.LRT:.4f} | {result.df} | {result.pvalue:.6f} | {sig} |"
        )

    lines.append("")

    # Add summary
    sig_count = sum(1 for r in results if r.significant(0.05))
    lines.append(f"**Summary:** {sig_count}/{len(results)} tests significant at α=0.05")

    return "\n".join(lines)


def _compare_results_csv(results: List[LRTResult]) -> str:
    """Format comparison as CSV."""
    lines = []

    # Header
    lines.append("test_name,lnL_null,lnL_alt,LRT,df,pvalue,significant_0.05")

    # Data
    for result in results:
        lines.append(
            f"{result.test_name},"
            f"{result.lnL_null},"
            f"{result.lnL_alt},"
            f"{result.LRT},"
            f"{result.df},"
            f"{result.pvalue},"
            f"{result.significant(0.05)}"
        )

    return "\n".join(lines)
