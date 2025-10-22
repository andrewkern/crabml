/// crabML Rust Backend
///
/// High-performance likelihood calculations for phylogenetic codon models.
/// Uses BLAS/LAPACK for matrix operations and Rayon for parallelization.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

mod matrix;
mod likelihood;

use matrix::CachedQMatrix;
use likelihood::{LikelihoodCalculator, Tree};

/// Cache Q matrix eigendecomposition for fast matrix exponentials
///
/// This dramatically speeds up likelihood calculations by computing
/// eigendecomposition once and reusing it for all branch lengths.
///
/// Args:
///     q: Rate matrix (n_states × n_states), must be reversible
///     pi: Stationary frequencies (n_states,)
///
/// Returns:
///     Opaque handle to cached Q matrix
#[pyfunction]
fn cache_q_matrix(
    q: PyReadonlyArray2<f64>,
    pi: PyReadonlyArray1<f64>,
) -> PyResult<usize> {
    let q_array = q.as_array();
    let pi_array = pi.as_array();

    let _cached = CachedQMatrix::new(q_array, pi_array)
        .map_err(|e| PyValueError::new_err(e))?;

    // Store in thread-local storage and return handle
    // TODO: Implement proper handle management
    // For now, return dummy handle
    Ok(0)
}

/// Compute matrix exponential exp(Q * t)
///
/// Uses cached eigendecomposition for maximum performance.
/// Expected speedup: 10-100x vs scipy.linalg.expm
///
/// Args:
///     q: Rate matrix (n_states × n_states)
///     pi: Stationary frequencies (n_states,)
///     t: Branch length
///
/// Returns:
///     Transition probability matrix P(t) = exp(Qt)
#[pyfunction]
fn matrix_exponential<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    pi: PyReadonlyArray1<'py, f64>,
    t: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q_array = q.as_array();
    let pi_array = pi.as_array();

    // Create cached Q matrix
    let cached = CachedQMatrix::new(q_array, pi_array)
        .map_err(|e| PyValueError::new_err(e))?;

    // Compute matrix exponential
    let result = cached.expm(t);

    // Convert to numpy array
    Ok(PyArray2::from_array_bound(py, &result))
}

/// Compute phylogenetic log-likelihood (single Q matrix)
///
/// Fast implementation of Felsenstein pruning algorithm using BLAS.
/// Expected speedup: 15-25x vs Python
///
/// Args:
///     q: Rate matrix (n_states × n_states)
///     pi: Stationary frequencies (n_states,)
///     tree_structure: List of (node_id, parent_id) tuples
///     branch_lengths: Branch length for each node (to parent)
///     leaf_names: Names of leaf nodes
///     sequences: Encoded sequences (n_leaves × n_sites)
///     leaf_node_ids: Node IDs that are leaves (for correct indexing)
///
/// Returns:
///     Log-likelihood value
#[pyfunction]
#[pyo3(signature = (q, pi, tree_structure, branch_lengths, leaf_names, sequences, leaf_node_ids))]
fn compute_log_likelihood(
    q: PyReadonlyArray2<f64>,
    pi: PyReadonlyArray1<f64>,
    tree_structure: Vec<(usize, Option<usize>)>,
    branch_lengths: Vec<f64>,
    leaf_names: Vec<String>,
    sequences: PyReadonlyArray2<i32>,
    leaf_node_ids: Vec<usize>,
) -> PyResult<f64> {
    let q_array = q.as_array();
    let pi_array = pi.as_array();
    let seq_array = sequences.as_array();

    // Create cached Q matrix
    let cached_q = CachedQMatrix::new(q_array, pi_array)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create tree
    let tree = Tree::from_structure(tree_structure, branch_lengths, leaf_names, leaf_node_ids)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create calculator
    let n_sites = seq_array.ncols();
    let n_states = pi_array.len();
    let mut calc = LikelihoodCalculator::new(tree, n_sites, n_states);

    // Compute likelihood
    Ok(calc.compute_log_likelihood(&cached_q, pi_array, seq_array))
}

/// Compute phylogenetic log-likelihood for site class models (M1a, M2a, M3)
///
/// PARALLELIZED: Site classes computed in parallel with Rayon.
/// Expected speedup: 20-30x vs Python
///
/// Args:
///     q_matrices: List of rate matrices, one per site class
///     proportions: Proportion of sites in each class
///     pi: Stationary frequencies
///     tree_structure: List of (node_id, parent_id) tuples
///     branch_lengths: Branch length for each node
///     leaf_names: Names of leaf nodes
///     sequences: Encoded sequences (n_leaves × n_sites)
///     leaf_node_ids: Node IDs that are leaves
///
/// Returns:
///     Log-likelihood value
#[pyfunction]
#[pyo3(signature = (q_matrices, proportions, pi, tree_structure, branch_lengths, leaf_names, sequences, leaf_node_ids))]
fn compute_site_class_log_likelihood(
    py: Python<'_>,
    q_matrices: Vec<PyReadonlyArray2<f64>>,
    proportions: Vec<f64>,
    pi: PyReadonlyArray1<f64>,
    tree_structure: Vec<(usize, Option<usize>)>,
    branch_lengths: Vec<f64>,
    leaf_names: Vec<String>,
    sequences: PyReadonlyArray2<i32>,
    leaf_node_ids: Vec<usize>,
) -> PyResult<f64> {
    let pi_array = pi.as_array();
    let seq_array = sequences.as_array();

    // Cache all Q matrices
    let cached_qs: Vec<CachedQMatrix> = q_matrices.iter()
        .map(|q| {
            let q_array = q.as_array();
            CachedQMatrix::new(q_array, pi_array)
                .map_err(|e| PyValueError::new_err(e))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Create tree
    let tree = Tree::from_structure(tree_structure, branch_lengths, leaf_names, leaf_node_ids)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create calculator
    let n_sites = seq_array.ncols();
    let n_states = pi_array.len();
    let mut calc = LikelihoodCalculator::new(tree, n_sites, n_states);

    // Release GIL during computation
    let result = py.allow_threads(|| {
        calc.compute_site_class_log_likelihood(
            &cached_qs,
            &proportions,
            pi_array,
            seq_array,
        )
    });

    Ok(result)
}

/// Compute site-specific log-likelihoods for each class (for BEB analysis)
///
/// Returns a 2D array of shape [n_sites, n_classes] where each entry [i, k]
/// is the log-likelihood of site i given class k.
///
/// This is used by Bayes Empirical Bayes to compute posterior probabilities
/// for each site belonging to each class.
///
/// PARALLELIZED: Site classes computed in parallel with Rayon.
///
/// Args:
///     q_matrices: List of rate matrices, one per site class
///     pi: Stationary frequencies
///     tree_structure: List of (node_id, parent_id) tuples
///     branch_lengths: Branch length for each node
///     leaf_names: Names of leaf nodes
///     sequences: Encoded sequences (n_leaves × n_sites)
///     leaf_node_ids: Node IDs that are leaves
///
/// Returns:
///     2D numpy array of shape (n_sites, n_classes) with log-likelihoods
#[pyfunction]
#[pyo3(signature = (q_matrices, pi, tree_structure, branch_lengths, leaf_names, sequences, leaf_node_ids))]
fn compute_site_log_likelihoods_by_class<'py>(
    py: Python<'py>,
    q_matrices: Vec<PyReadonlyArray2<f64>>,
    pi: PyReadonlyArray1<f64>,
    tree_structure: Vec<(usize, Option<usize>)>,
    branch_lengths: Vec<f64>,
    leaf_names: Vec<String>,
    sequences: PyReadonlyArray2<i32>,
    leaf_node_ids: Vec<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pi_array = pi.as_array();
    let seq_array = sequences.as_array();

    // Cache all Q matrices
    let cached_qs: Vec<CachedQMatrix> = q_matrices.iter()
        .map(|q| {
            let q_array = q.as_array();
            CachedQMatrix::new(q_array, pi_array)
                .map_err(|e| PyValueError::new_err(e))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Create tree
    let tree = Tree::from_structure(tree_structure, branch_lengths, leaf_names, leaf_node_ids)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create calculator
    let n_sites = seq_array.ncols();
    let n_states = pi_array.len();
    let mut calc = LikelihoodCalculator::new(tree, n_sites, n_states);

    // Release GIL during computation
    let result = py.allow_threads(|| {
        calc.compute_site_log_likelihoods_by_class(
            &cached_qs,
            pi_array,
            seq_array,
        )
    });

    // Convert to numpy array
    Ok(PyArray2::from_array_bound(py, &result))
}

/// crabML Rust Backend Module
///
/// High-performance phylogenetic likelihood calculations.
#[pymodule]
fn crabml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_exponential, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(compute_site_class_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(compute_site_log_likelihoods_by_class, m)?)?;
    Ok(())
}
