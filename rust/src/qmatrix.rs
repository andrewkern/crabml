/// Fast Q matrix construction using pre-computed codon graph
///
/// This module provides high-performance codon Q matrix construction
/// by using the pre-computed codon relationship graph.

use ndarray::{Array1, Array2};
use crate::codon::codon_graph;

/// Build a codon Q matrix for given kappa, omega, and pi
///
/// This is the hot path for optimization - called hundreds of times.
/// Uses the pre-computed codon graph for maximum performance.
///
/// # Arguments
/// * `kappa` - Transition/transversion ratio
/// * `omega` - dN/dS ratio (non-synonymous/synonymous rate ratio)
/// * `pi` - Codon equilibrium frequencies (length 61)
/// * `normalization_factor` - Optional pre-computed normalization factor
///
/// # Returns
/// Q matrix (61 x 61)
pub fn build_codon_q_matrix(
    kappa: f64,
    omega: f64,
    pi: &Array1<f64>,
    normalization_factor: Option<f64>,
) -> Array2<f64> {
    let n = 61;
    let mut q = Array2::zeros((n, n));

    let graph = codon_graph();

    // Build off-diagonal elements using pre-computed graph
    for i in 0..n {
        for edge in graph.neighbors(i) {
            let j = edge.to_codon;

            // Start with base rate of 1.0
            let mut rate = 1.0;

            // Multiply by kappa if transition
            if edge.is_transition {
                rate *= kappa;
            }

            // Multiply by omega if non-synonymous
            if !edge.is_synonymous {
                rate *= omega;
            }

            // Apply reversibility: Q[i,j] = rate * pi[j]
            q[[i, j]] = rate * pi[j];
        }
    }

    // Set diagonal elements: Q[i,i] = -sum(Q[i,j] for j ≠ i)
    for i in 0..n {
        let row_sum: f64 = q.row(i).sum();
        q[[i, i]] = -row_sum;
    }

    // Normalize
    if let Some(factor) = normalization_factor {
        // Use provided normalization factor (for site class models)
        q / factor
    } else {
        // Compute normalization factor from this Q matrix
        // Expected rate = -sum(pi[i] * Q[i,i])
        let expected_rate: f64 = -(0..n)
            .map(|i| pi[i] * q[[i, i]])
            .sum::<f64>();

        if expected_rate > 0.0 {
            q / expected_rate
        } else {
            // Edge case: avoid division by zero
            q
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_q_matrix_shape() {
        let pi = Array1::from_elem(61, 1.0 / 61.0);
        let q = build_codon_q_matrix(2.0, 0.5, &pi, None);

        assert_eq!(q.shape(), &[61, 61]);
    }

    #[test]
    fn test_q_matrix_row_sums_zero() {
        // Each row should sum to zero (within numerical precision)
        let pi = Array1::from_elem(61, 1.0 / 61.0);
        let q = build_codon_q_matrix(2.0, 0.5, &pi, None);

        for i in 0..61 {
            let row_sum: f64 = q.row(i).sum();
            assert_abs_diff_eq!(row_sum, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_q_matrix_normalization() {
        // Without normalization factor, expected rate should be 1.0
        let pi = Array1::from_elem(61, 1.0 / 61.0);
        let q = build_codon_q_matrix(2.0, 0.5, &pi, None);

        // Expected rate = -sum(pi[i] * Q[i,i])
        let expected_rate: f64 = -(0..61)
            .map(|i| pi[i] * q[[i, i]])
            .sum::<f64>();

        assert_abs_diff_eq!(expected_rate, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_q_matrix_detailed_balance() {
        // Test reversibility: pi[i] * Q[i,j] = pi[j] * Q[j,i]
        let pi = Array1::from_vec(
            (0..61).map(|i| (i + 1) as f64 / (61.0 * 62.0 / 2.0)).collect()
        );
        let q = build_codon_q_matrix(2.0, 0.5, &pi, None);

        for i in 0..61 {
            for j in 0..61 {
                if i != j && q[[i, j]] > 0.0 {
                    let lhs = pi[i] * q[[i, j]];
                    let rhs = pi[j] * q[[j, i]];
                    assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_q_matrix_kappa_effect() {
        // Higher kappa should increase transition rates relative to transversion rates
        let pi = Array1::from_elem(61, 1.0 / 61.0);
        let q1 = build_codon_q_matrix(1.0, 1.0, &pi, Some(1.0));
        let q2 = build_codon_q_matrix(3.0, 1.0, &pi, Some(1.0));

        // With kappa=3, transitions should be 3x higher than transversions
        // But the difference should be visible in off-diagonal elements
        let mut found_difference = false;
        for i in 0..61 {
            for j in 0..61 {
                if i != j && q1[[i, j]] > 0.0 {
                    // q2 should generally be >= q1 (transitions get boosted)
                    if q2[[i, j]] > q1[[i, j]] * 1.5 {
                        found_difference = true;
                    }
                }
            }
        }
        assert!(found_difference, "Kappa should affect some substitution rates");
    }

    #[test]
    fn test_q_matrix_omega_effect() {
        // omega affects non-synonymous substitutions
        let pi = Array1::from_elem(61, 1.0 / 61.0);
        let q1 = build_codon_q_matrix(2.0, 0.1, &pi, None);
        let q2 = build_codon_q_matrix(2.0, 2.0, &pi, None);

        // With omega=2.0, non-synonymous rates should be higher
        // This affects the overall scale, so we can't compare individual elements directly
        // But the diagonal elements should differ
        let mut diff_count = 0;
        for i in 0..61 {
            if (q2[[i, i]] - q1[[i, i]]).abs() > 1e-6 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 30, "Omega should affect many diagonal elements");
    }
}
