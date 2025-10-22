"""
Visualization tools for Bayes Empirical Bayes (BEB) results.

Provides publication-quality plots for exploring BEB analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from .beb import BEBResult

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class BEBVisualizer:
    """
    Visualization tools for BEB results.

    Examples
    --------
    >>> from crabml.analysis.beb_visualizer import BEBVisualizer
    >>> viz = BEBVisualizer()
    >>> fig = viz.plot_posterior_probabilities(beb_result)
    >>> fig.savefig('beb_posteriors.png', dpi=300, bbox_inches='tight')
    """

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer.

        Parameters
        ----------
        style : str, default='seaborn-v0_8-whitegrid'
            Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass

    def plot_posterior_probabilities(
        self,
        beb_result: BEBResult,
        threshold_95: float = 0.95,
        threshold_99: float = 0.99,
        figsize: Tuple[float, float] = (14, 6),
        highlight_sites: Optional[List[int]] = None
    ) -> plt.Figure:
        """
        Bar plot of posterior probabilities for positive selection.

        Parameters
        ----------
        beb_result : BEBResult
            BEB analysis results
        threshold_95 : float, default=0.95
            Threshold for marking sites (*)
        threshold_99 : float, default=0.99
            Threshold for marking sites (**)
        figsize : tuple, default=(14, 6)
            Figure size in inches
        highlight_sites : List[int], optional
            Specific sites to highlight

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        x = beb_result.site_numbers
        y = beb_result.posterior_probs[:, -1]  # Positive selection class

        # Color code by significance
        colors = np.array(['lightgray'] * len(y))
        colors[y >= threshold_95] = 'orange'
        colors[y >= threshold_99] = 'red'

        # Create bar plot
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Highlight specific sites if requested
        if highlight_sites:
            for site in highlight_sites:
                idx = np.where(beb_result.site_numbers == site)[0]
                if len(idx) > 0:
                    bars[idx[0]].set_edgecolor('blue')
                    bars[idx[0]].set_linewidth(2)

        # Add threshold lines
        ax.axhline(threshold_95, color='orange', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'P > {threshold_95} (*)')
        ax.axhline(threshold_99, color='red', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'P > {threshold_99} (**)')

        # Labels and title
        ax.set_xlabel('Site Position', fontweight='bold')
        ax.set_ylabel('Posterior Probability (ω > 1)', fontweight='bold')
        ax.set_title(
            f'Bayes Empirical Bayes: Positively Selected Sites ({beb_result.model_name})',
            fontweight='bold'
        )

        # Legend
        ax.legend(loc='upper right', framealpha=0.9)

        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_omega_distribution(
        self,
        beb_result: BEBResult,
        figsize: Tuple[float, float] = (12, 5),
        bins: int = 30
    ) -> plt.Figure:
        """
        Distribution of posterior mean ω values across sites.

        Parameters
        ----------
        beb_result : BEBResult
            BEB analysis results
        figsize : tuple, default=(12, 5)
            Figure size
        bins : int, default=30
            Number of histogram bins

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        omega = beb_result.posterior_omega

        # Left panel: Histogram
        ax1.hist(omega, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(1.0, color='red', linestyle='--', linewidth=2,
                    label='ω = 1 (neutral)')
        ax1.set_xlabel('Posterior Mean ω', fontweight='bold')
        ax1.set_ylabel('Number of Sites', fontweight='bold')
        ax1.set_title('Distribution of Posterior ω', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Right panel: Violin plot by class
        # Group sites by most likely class
        class_assignments = np.argmax(beb_result.posterior_probs, axis=1)

        data_by_class = []
        labels = []
        positions = []
        for i in range(len(beb_result.site_classes)):
            mask = class_assignments == i
            if np.any(mask):
                data_by_class.append(omega[mask])
                labels.append(beb_result.site_classes[i])
                positions.append(len(data_by_class) - 1)

        if len(data_by_class) > 0:
            parts = ax2.violinplot(data_by_class, positions=positions,
                                    showmeans=True, showmedians=True)

            # Color by omega value
            for i, pc in enumerate(parts['bodies']):
                class_omega = beb_result.class_omegas[i]
                if class_omega > 1:
                    pc.set_facecolor('red')
                    pc.set_alpha(0.5)
                elif class_omega < 1:
                    pc.set_facecolor('blue')
                    pc.set_alpha(0.5)
                else:
                    pc.set_facecolor('gray')
                    pc.set_alpha(0.5)

            ax2.set_xticks(positions)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_ylabel('Posterior Mean ω', fontweight='bold')
            ax2.set_title('ω by Most Likely Class', fontweight='bold')
            ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_site_heatmap(
        self,
        beb_result: BEBResult,
        figsize: Tuple[float, float] = (10, 12),
        cmap: str = 'YlOrRd',
        show_yticklabels: bool = False
    ) -> plt.Figure:
        """
        Heatmap of posterior probabilities across sites and classes.

        Parameters
        ----------
        beb_result : BEBResult
            BEB analysis results
        figsize : tuple, default=(10, 12)
            Figure size
        cmap : str, default='YlOrRd'
            Colormap name
        show_yticklabels : bool, default=False
            Whether to show site numbers on y-axis (can be crowded)

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            beb_result.posterior_probs,
            cmap=cmap,
            vmin=0,
            vmax=1,
            xticklabels=beb_result.site_classes,
            yticklabels=beb_result.site_numbers if show_yticklabels else False,
            cbar_kws={'label': 'Posterior Probability'},
            ax=ax
        )

        ax.set_xlabel('Site Class', fontweight='bold')
        ax.set_ylabel('Site Position', fontweight='bold')
        ax.set_title(
            f'BEB Posterior Probabilities Heatmap ({beb_result.model_name})',
            fontweight='bold'
        )

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_sequence_view(
        self,
        beb_result: BEBResult,
        alignment,
        reference_seq_idx: int = 0,
        threshold: float = 0.95,
        figsize: Tuple[float, float] = (16, 8)
    ) -> plt.Figure:
        """
        Sequence alignment view with BEB results overlay.

        Shows the amino acid sequence with sites colored by
        posterior probability of positive selection.

        Parameters
        ----------
        beb_result : BEBResult
            BEB analysis results
        alignment : Alignment
            Sequence alignment
        reference_seq_idx : int, default=0
            Index of reference sequence to display
        threshold : float, default=0.95
            Threshold for highlighting sites
        figsize : tuple, default=(16, 8)
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        from ..io.sequences import INDEX_TO_CODON, GENETIC_CODE

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=figsize,
            height_ratios=[1, 2],
            sharex=True
        )

        # Top panel: Posterior probabilities
        x = beb_result.site_numbers
        y = beb_result.posterior_probs[:, -1]

        colors = ['red' if p > threshold else 'gray' for p in y]
        ax1.bar(x, y, color=colors, alpha=0.6, width=1.0, edgecolor='none')
        ax1.axhline(threshold, color='black', linestyle='--', linewidth=1)
        ax1.set_ylabel('P(ω > 1)', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_title('Sequence View with BEB Results', fontweight='bold')

        # Bottom panel: Amino acid sequence
        # Convert codon indices to amino acids
        ref_seq = alignment.sequences[reference_seq_idx, :]
        amino_acids = []

        for codon_idx in ref_seq:
            if codon_idx >= 0:
                codon = INDEX_TO_CODON[codon_idx]
                aa = GENETIC_CODE[codon]
                amino_acids.append(aa)
            else:
                amino_acids.append('-')

        # Plot amino acids as text
        for i, aa in enumerate(amino_acids):
            site_num = i + 1
            prob = beb_result.posterior_probs[i, -1]

            # Color by posterior probability
            if prob > threshold:
                color = 'red'
                weight = 'bold'
                size = 12
            else:
                color = 'black'
                weight = 'normal'
                size = 10

            ax2.text(
                site_num, 0.5, aa,
                ha='center', va='center',
                color=color,
                fontweight=weight,
                fontsize=size,
                family='monospace'
            )

        ax2.set_xlim(0.5, len(amino_acids) + 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Site Position', fontweight='bold')
        ax2.set_ylabel('Amino Acid', fontweight='bold')
        ax2.set_yticks([])
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Add gridlines at every 10th site
        for i in range(10, len(amino_acids) + 1, 10):
            ax2.axvline(i + 0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        beb_results: List[BEBResult],
        labels: List[str],
        figsize: Tuple[float, float] = (14, 6)
    ) -> plt.Figure:
        """
        Compare BEB results from multiple models.

        Parameters
        ----------
        beb_results : List[BEBResult]
            List of BEB results to compare
        labels : List[str]
            Labels for each result
        figsize : tuple, default=(14, 6)
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(beb_results)
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        for i, (result, label) in enumerate(zip(beb_results, labels)):
            # Plot as line
            x = result.site_numbers
            y = result.posterior_probs[:, -1]

            ax.plot(x, y, label=label, alpha=0.7, linewidth=1.5, color=colors[i])

        ax.axhline(0.95, color='red', linestyle='--', linewidth=1,
                   alpha=0.5, label='P = 0.95')
        ax.set_xlabel('Site Position', fontweight='bold')
        ax.set_ylabel('Posterior Probability (ω > 1)', fontweight='bold')
        ax.set_title('BEB Comparison Across Models', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        return fig
