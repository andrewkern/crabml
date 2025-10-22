"""
Demonstration of all export formats available in the analysis module.

This shows how to export LRT results in various formats suitable for
publications, presentations, and further analysis.
"""

from crabml.analysis import test_positive_selection, compare_results

# Path to data files
FASTA = "tests/data/lysozyme.fasta"
TREE = "tests/data/lysozyme.tree"

def main():
    print("Running positive selection tests...")
    results = test_positive_selection(
        alignment=FASTA,
        tree=TREE,
        test='both',
        verbose=False
    )

    m1a_m2a = results['M1a_vs_M2a']
    m7_m8 = results['M7_vs_M8']

    print("\n" + "=" * 80)
    print("EXPORT FORMAT DEMONSTRATIONS")
    print("=" * 80)

    # 1. Formatted summary (default)
    print("\n1. FORMATTED SUMMARY (console output)")
    print("-" * 80)
    print(m1a_m2a.summary())

    # 2. JSON export
    print("\n2. JSON EXPORT")
    print("-" * 80)
    json_str = m1a_m2a.to_json()
    print(json_str[:500] + "...")  # Print first 500 chars
    print("\n# Save to file:")
    print("result.to_json('m1a_vs_m2a.json')")

    # 3. Markdown table
    print("\n3. MARKDOWN TABLE (for papers)")
    print("-" * 80)
    print(m1a_m2a.to_markdown_table())

    # 4. CSV export
    print("\n4. CSV EXPORT (single result)")
    print("-" * 80)
    print(m1a_m2a.to_csv_row(include_header=True))

    # 5. Dictionary export
    print("\n5. DICTIONARY EXPORT (for programmatic access)")
    print("-" * 80)
    result_dict = m1a_m2a.to_dict()
    print(f"Keys: {list(result_dict.keys())}")
    print(f"P-value: {result_dict['pvalue']:.6f}")
    print(f"Significant: {result_dict['significant_0.05']}")

    # 6. Comparison table (both tests)
    print("\n6. COMPARISON TABLE (multiple tests)")
    print("-" * 80)
    comparison = compare_results([m1a_m2a, m7_m8], format='table')
    print(comparison)

    # 7. Comparison markdown
    print("\n7. COMPARISON MARKDOWN (for papers)")
    print("-" * 80)
    comparison_md = compare_results([m1a_m2a, m7_m8], format='markdown')
    print(comparison_md)

    # 8. Comparison CSV
    print("\n8. COMPARISON CSV (for spreadsheets)")
    print("-" * 80)
    comparison_csv = compare_results([m1a_m2a, m7_m8], format='csv')
    print(comparison_csv)

    # 9. Pandas DataFrame (if pandas installed)
    print("\n9. PANDAS DATAFRAME")
    print("-" * 80)
    try:
        df = m1a_m2a.to_dataframe()
        print("DataFrame created successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        print("\n# Save to CSV:")
        print("df.to_csv('results.csv', index=False)")
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")

    # 10. Programmatic access to key values
    print("\n10. PROGRAMMATIC ACCESS")
    print("-" * 80)
    print(f"Test: {m1a_m2a.test_name}")
    print(f"LRT statistic: {m1a_m2a.LRT:.4f}")
    print(f"P-value: {m1a_m2a.pvalue:.6f}")
    print(f"Significant at α=0.05: {m1a_m2a.significant(0.05)}")
    print(f"Significant at α=0.01: {m1a_m2a.significant(0.01)}")
    if m1a_m2a.omega_positive:
        print(f"ω for positive selection: {m1a_m2a.omega_positive:.4f}")
        print(f"Proportion of sites: {m1a_m2a.proportion_positive:.2%}")

    print("\n" + "=" * 80)
    print("All export formats demonstrated!")
    print("=" * 80)

if __name__ == "__main__":
    main()
