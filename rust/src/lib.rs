/// crabML Rust Backend
///
/// High-performance likelihood calculations for phylogenetic codon models.
/// Uses BLAS/LAPACK for matrix operations and Rayon for parallelization.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

mod matrix;
mod likelihood;
mod branch_site_likelihood;

use matrix::CachedQMatrix;
use likelihood::{LikelihoodCalculator, Tree};
use branch_site_likelihood::BranchSiteLikelihoodCalculator;

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

/// Compute branch-site model log-likelihood (Model A)
///
/// Branch-site models allow different omega values on foreground vs background branches.
/// Each branch+site_class combination uses a different omega value.
///
/// PARALLELIZED: Site classes computed in parallel with Rayon.
/// Expected speedup: 20-30x vs Python
///
/// Args:
///     q_omega0: Q matrix with omega0 (purifying selection)
///     q_omega1: Q matrix with omega=1 (neutral)
///     q_omega2: Q matrix with omega2 (positive selection)
///     qfactor_back: Q normalization factor for background branches
///     qfactor_fore: Q normalization factor for foreground branches
///     omega0: Purifying selection omega value
///     omega2: Positive selection omega value
///     site_class_freqs: Frequencies of 4 site classes [p0, p1, p2a, p2b]
///     pi: Stationary frequencies
///     tree_structure: List of (node_id, parent_id) tuples
///     branch_lengths: Branch length for each node
///     branch_labels: Branch labels (0=background, 1=foreground)
///     leaf_names: Names of leaf nodes
///     sequences: Encoded sequences (n_leaves × n_sites)
///     leaf_node_ids: Node IDs that are leaves
///
/// Returns:
///     Log-likelihood value
#[pyfunction]
#[pyo3(signature = (q_omega0, q_omega1, q_omega2, qfactor_back, qfactor_fore, omega0, omega2, site_class_freqs, pi, tree_structure, branch_lengths, branch_labels, leaf_names, sequences, leaf_node_ids))]
fn compute_branch_site_log_likelihood(
    py: Python<'_>,
    q_omega0: PyReadonlyArray2<f64>,
    q_omega1: PyReadonlyArray2<f64>,
    q_omega2: PyReadonlyArray2<f64>,
    qfactor_back: f64,
    qfactor_fore: f64,
    omega0: f64,
    omega2: f64,
    site_class_freqs: [f64; 4],
    pi: PyReadonlyArray1<f64>,
    tree_structure: Vec<(usize, Option<usize>)>,
    branch_lengths: Vec<f64>,
    branch_labels: Vec<u8>,
    leaf_names: Vec<String>,
    sequences: PyReadonlyArray2<i32>,
    leaf_node_ids: Vec<usize>,
) -> PyResult<f64> {
    let pi_array = pi.as_array();
    let seq_array = sequences.as_array();

    // Cache all Q matrices
    let cached_q0 = CachedQMatrix::new(q_omega0.as_array(), pi_array)
        .map_err(|e| PyValueError::new_err(e))?;
    let cached_q1 = CachedQMatrix::new(q_omega1.as_array(), pi_array)
        .map_err(|e| PyValueError::new_err(e))?;
    let cached_q2 = CachedQMatrix::new(q_omega2.as_array(), pi_array)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create tree
    let tree = Tree::from_structure(tree_structure, branch_lengths, leaf_names, leaf_node_ids)
        .map_err(|e| PyValueError::new_err(e))?;

    // Create calculator
    let n_sites = seq_array.ncols();
    let n_states = pi_array.len();
    let mut calc = BranchSiteLikelihoodCalculator::new(tree, n_sites, n_states, branch_labels)
        .map_err(|e| PyValueError::new_err(e))?;

    // Release GIL during computation
    let result = py.allow_threads(|| {
        calc.compute_log_likelihood(
            &cached_q0,
            &cached_q1,
            &cached_q2,
            qfactor_back,
            qfactor_fore,
            omega0,
            omega2,
            &site_class_freqs,
            pi_array,
            seq_array,
        )
    });

    Ok(result)
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
    m.add_function(wrap_pyfunction!(compute_branch_site_log_likelihood, m)?)?;
    Ok(())
}
