"""
Example of using the positive selection analysis API.

This demonstrates the new convenience functions for testing positive selection
using M1a vs M2a and M7 vs M8 likelihood ratio tests.
"""

from crabml.analysis import test_positive_selection

# Path to data files
FASTA = "tests/data/lysozyme.fasta"
TREE = "tests/data/lysozyme.tree"

def main():
    print("=" * 80)
    print("Testing for Positive Selection in Lysozyme")
    print("=" * 80)
    print()

    # Run both tests
    print("Running both M1a vs M2a and M7 vs M8 tests...")
    print()

    results = test_positive_selection(
        alignment=FASTA,
        tree=TREE,
        test='both',
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    print("M1a vs M2a Test:")
    print(f"  LRT statistic: {results['M1a_vs_M2a'].LRT:.4f}")
    print(f"  P-value: {results['M1a_vs_M2a'].pvalue:.6f}")
    print(f"  Significant (α=0.05): {results['M1a_vs_M2a'].significant(0.05)}")

    if results['M1a_vs_M2a'].omega_positive:
        print(f"  ω for positive selection: {results['M1a_vs_M2a'].omega_positive:.4f}")
        print(f"  Proportion of sites: {results['M1a_vs_M2a'].proportion_positive:.2%}")
    print()

    print("M7 vs M8 Test:")
    print(f"  LRT statistic: {results['M7_vs_M8'].LRT:.4f}")
    print(f"  P-value: {results['M7_vs_M8'].pvalue:.6f}")
    print(f"  Significant (α=0.05): {results['M7_vs_M8'].significant(0.05)}")

    if results['M7_vs_M8'].omega_positive:
        print(f"  ω for positive selection: {results['M7_vs_M8'].omega_positive:.4f}")
        print(f"  Proportion of sites: {results['M7_vs_M8'].proportion_positive:.2%}")
    print()

    # Overall conclusion
    print("Overall Conclusion:")
    m1a_m2a_sig = results['M1a_vs_M2a'].significant(0.05)
    m7_m8_sig = results['M7_vs_M8'].significant(0.05)

    if m1a_m2a_sig and m7_m8_sig:
        print("  ✓ Strong evidence for positive selection (both tests significant)")
    elif m1a_m2a_sig or m7_m8_sig:
        print("  ~ Moderate evidence for positive selection (one test significant)")
    else:
        print("  ✗ No significant evidence for positive selection")
    print()

    # Export results
    print("Exporting results to dictionary...")
    result_dict = results['M1a_vs_M2a'].to_dict()
    print(f"  Keys: {list(result_dict.keys())}")
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
