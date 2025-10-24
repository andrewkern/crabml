/// Codon utilities for building Q matrices
///
/// This module provides fast codon encoding/decoding and relationship checking
/// that is used to build codon substitution models.

use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Nucleotide encoding (matches PAML convention)
/// T=0, C=1, A=2, G=3
const NUCLEOTIDES: [char; 4] = ['T', 'C', 'A', 'G'];

/// Standard genetic code (64 codons -> amino acids or stop)
/// Indexed by codon as: n0*16 + n1*4 + n2 where ni are nucleotide indices
static GENETIC_CODE: Lazy<HashMap<String, char>> = Lazy::new(|| {
    let mut code = HashMap::new();

    // Standard genetic code
    let codons_aa = [
        // TTx
        ("TTT", 'F'), ("TTC", 'F'), ("TTA", 'L'), ("TTG", 'L'),
        // TCx
        ("TCT", 'S'), ("TCC", 'S'), ("TCA", 'S'), ("TCG", 'S'),
        // TAx
        ("TAT", 'Y'), ("TAC", 'Y'), ("TAA", '*'), ("TAG", '*'),
        // TGx
        ("TGT", 'C'), ("TGC", 'C'), ("TGA", '*'), ("TGG", 'W'),
        // CTx
        ("CTT", 'L'), ("CTC", 'L'), ("CTA", 'L'), ("CTG", 'L'),
        // CCx
        ("CCT", 'P'), ("CCC", 'P'), ("CCA", 'P'), ("CCG", 'P'),
        // CAx
        ("CAT", 'H'), ("CAC", 'H'), ("CAA", 'Q'), ("CAG", 'Q'),
        // CGx
        ("CGT", 'R'), ("CGC", 'R'), ("CGA", 'R'), ("CGG", 'R'),
        // ATx
        ("ATT", 'I'), ("ATC", 'I'), ("ATA", 'I'), ("ATG", 'M'),
        // ACx
        ("ACT", 'T'), ("ACC", 'T'), ("ACA", 'T'), ("ACG", 'T'),
        // AAx
        ("AAT", 'N'), ("AAC", 'N'), ("AAA", 'K'), ("AAG", 'K'),
        // AGx
        ("AGT", 'S'), ("AGC", 'S'), ("AGA", 'R'), ("AGG", 'R'),
        // GTx
        ("GTT", 'V'), ("GTC", 'V'), ("GTA", 'V'), ("GTG", 'V'),
        // GCx
        ("GCT", 'A'), ("GCC", 'A'), ("GCA", 'A'), ("GCG", 'A'),
        // GAx
        ("GAT", 'D'), ("GAC", 'D'), ("GAA", 'E'), ("GAG", 'E'),
        // GGx
        ("GGT", 'G'), ("GGC", 'G'), ("GGA", 'G'), ("GGG", 'G'),
    ];

    for (codon, aa) in codons_aa {
        code.insert(codon.to_string(), aa);
    }

    code
});

/// List of 61 sense codons (excluding stop codons)
/// Order matches PAML's FROM61 array for compatibility
static CODONS: Lazy<Vec<String>> = Lazy::new(|| {
    let mut codons = Vec::new();

    // Generate all 64 codons in PAML order
    for i in 0..64 {
        let n0 = i / 16;
        let n1 = (i / 4) % 4;
        let n2 = i % 4;

        let codon = format!(
            "{}{}{}",
            NUCLEOTIDES[n0],
            NUCLEOTIDES[n1],
            NUCLEOTIDES[n2]
        );

        // Skip stop codons (TAA, TAG, TGA)
        if GENETIC_CODE.get(&codon) != Some(&'*') {
            codons.push(codon);
        }
    }

    assert_eq!(codons.len(), 61, "Expected 61 sense codons");
    codons
});

/// Map from codon string to index (0-60)
static CODON_TO_INDEX: Lazy<HashMap<String, usize>> = Lazy::new(|| {
    CODONS.iter()
        .enumerate()
        .map(|(i, codon)| (codon.clone(), i))
        .collect()
});

/// Get codon string from index (0-60)
#[inline]
pub fn index_to_codon(index: usize) -> &'static str {
    &CODONS[index]
}

/// Get index (0-60) from codon string
#[inline]
pub fn codon_to_index(codon: &str) -> Option<usize> {
    CODON_TO_INDEX.get(codon).copied()
}

/// Check if a nucleotide substitution is a transition (A<->G or C<->T)
#[inline]
pub fn is_transition(nuc1: char, nuc2: char) -> bool {
    matches!(
        (nuc1, nuc2),
        ('A', 'G') | ('G', 'A') | ('C', 'T') | ('T', 'C')
    )
}

/// Check if two codons code for the same amino acid (synonymous substitution)
#[inline]
pub fn is_synonymous(codon1: &str, codon2: &str) -> bool {
    GENETIC_CODE.get(codon1) == GENETIC_CODE.get(codon2)
}

/// Get the number of nucleotide differences between two codons
#[inline]
pub fn nucleotide_differences(codon1: &str, codon2: &str) -> usize {
    codon1.chars()
        .zip(codon2.chars())
        .filter(|(c1, c2)| c1 != c2)
        .count()
}

/// Find the position (0-2) where two codons differ
/// Returns None if they don't differ at exactly one position
#[inline]
pub fn find_diff_position(codon1: &str, codon2: &str) -> Option<usize> {
    let diffs: Vec<usize> = codon1.chars()
        .zip(codon2.chars())
        .enumerate()
        .filter(|(_, (c1, c2))| c1 != c2)
        .map(|(i, _)| i)
        .collect();

    if diffs.len() == 1 {
        Some(diffs[0])
    } else {
        None
    }
}

/// Information about a single-nucleotide codon substitution
#[derive(Clone, Copy, Debug)]
pub struct CodonEdge {
    /// Target codon index (0-60)
    pub to_codon: usize,
    /// Is this a transition (A<->G or C<->T)?
    pub is_transition: bool,
    /// Is this synonymous (same amino acid)?
    pub is_synonymous: bool,
}

/// Pre-computed graph of all valid single-nucleotide codon substitutions
///
/// This is computed once at module initialization and reused for all
/// Q matrix constructions, eliminating the need to repeatedly check
/// codon relationships.
pub struct CodonGraph {
    /// For each codon i, edges[i] contains all single-nucleotide neighbors
    pub edges: Vec<Vec<CodonEdge>>,
}

impl CodonGraph {
    /// Build the codon graph (run once at initialization)
    fn new() -> Self {
        let n_codons = CODONS.len();
        let mut edges = vec![Vec::new(); n_codons];

        for i in 0..n_codons {
            let codon_i = index_to_codon(i);

            for j in 0..n_codons {
                if i == j {
                    continue;
                }

                let codon_j = index_to_codon(j);

                // Check if they differ at exactly one position
                if nucleotide_differences(codon_i, codon_j) != 1 {
                    continue;
                }

                // Find the differing position
                let diff_pos = find_diff_position(codon_i, codon_j).unwrap();
                let chars_i: Vec<char> = codon_i.chars().collect();
                let chars_j: Vec<char> = codon_j.chars().collect();
                let nuc_i = chars_i[diff_pos];
                let nuc_j = chars_j[diff_pos];

                // Record the edge
                edges[i].push(CodonEdge {
                    to_codon: j,
                    is_transition: is_transition(nuc_i, nuc_j),
                    is_synonymous: is_synonymous(codon_i, codon_j),
                });
            }
        }

        CodonGraph { edges }
    }

    /// Get the neighbors of a codon
    #[inline]
    pub fn neighbors(&self, codon_index: usize) -> &[CodonEdge] {
        &self.edges[codon_index]
    }
}

/// Global codon graph (computed once at module load)
static CODON_GRAPH: Lazy<CodonGraph> = Lazy::new(|| CodonGraph::new());

/// Get the global codon graph
#[inline]
pub fn codon_graph() -> &'static CodonGraph {
    &CODON_GRAPH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codon_count() {
        assert_eq!(CODONS.len(), 61, "Should have 61 sense codons");
    }

    #[test]
    fn test_no_stop_codons() {
        let stop_codons = ["TAA", "TAG", "TGA"];
        for stop in &stop_codons {
            assert!(!CODONS.contains(&stop.to_string()),
                    "Stop codon {} should not be in CODONS", stop);
        }
    }

    #[test]
    fn test_codon_indexing() {
        // Test round-trip conversion
        for (i, codon) in CODONS.iter().enumerate() {
            assert_eq!(codon_to_index(codon), Some(i));
            assert_eq!(index_to_codon(i), codon);
        }
    }

    #[test]
    fn test_is_transition() {
        // Transitions
        assert!(is_transition('A', 'G'));
        assert!(is_transition('G', 'A'));
        assert!(is_transition('C', 'T'));
        assert!(is_transition('T', 'C'));

        // Transversions
        assert!(!is_transition('A', 'T'));
        assert!(!is_transition('A', 'C'));
        assert!(!is_transition('G', 'T'));
        assert!(!is_transition('G', 'C'));
        assert!(!is_transition('C', 'A'));
        assert!(!is_transition('C', 'G'));
    }

    #[test]
    fn test_is_synonymous() {
        // Synonymous (both code for Leucine)
        assert!(is_synonymous("TTG", "TTA"));
        assert!(is_synonymous("CTT", "CTC"));

        // Synonymous (both code for Phenylalanine)
        assert!(is_synonymous("TTT", "TTC"));

        // Non-synonymous
        assert!(!is_synonymous("ATG", "ATA")); // Met vs Ile
        assert!(!is_synonymous("TTT", "TTA")); // Phe vs Leu
    }

    #[test]
    fn test_synonymous_correct() {
        // TTT and TTC both code for Phenylalanine
        assert!(is_synonymous("TTT", "TTC"));

        // ATG (Met) vs ATA (Ile) are non-synonymous
        assert!(!is_synonymous("ATG", "ATA"));

        // All Leucine codons
        let leu_codons = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"];
        for &c1 in &leu_codons {
            for &c2 in &leu_codons {
                assert!(is_synonymous(c1, c2), "{} and {} should both be Leucine", c1, c2);
            }
        }
    }

    #[test]
    fn test_nucleotide_differences() {
        assert_eq!(nucleotide_differences("ATG", "ATG"), 0);
        assert_eq!(nucleotide_differences("ATG", "ATA"), 1);
        assert_eq!(nucleotide_differences("ATG", "AAA"), 2);
        assert_eq!(nucleotide_differences("ATG", "CCC"), 3);
    }

    #[test]
    fn test_find_diff_position() {
        // Single difference
        assert_eq!(find_diff_position("ATG", "ATA"), Some(2));
        assert_eq!(find_diff_position("ATG", "ACG"), Some(1));
        assert_eq!(find_diff_position("ATG", "CTG"), Some(0));

        // No difference
        assert_eq!(find_diff_position("ATG", "ATG"), None);

        // Multiple differences
        assert_eq!(find_diff_position("ATG", "AAA"), None);
        assert_eq!(find_diff_position("ATG", "CCC"), None);
    }

    #[test]
    fn test_genetic_code() {
        // Start codon
        assert_eq!(GENETIC_CODE.get("ATG"), Some(&'M'));

        // Stop codons
        assert_eq!(GENETIC_CODE.get("TAA"), Some(&'*'));
        assert_eq!(GENETIC_CODE.get("TAG"), Some(&'*'));
        assert_eq!(GENETIC_CODE.get("TGA"), Some(&'*'));

        // Common amino acids
        assert_eq!(GENETIC_CODE.get("GGG"), Some(&'G')); // Glycine
        assert_eq!(GENETIC_CODE.get("AAA"), Some(&'K')); // Lysine
        assert_eq!(GENETIC_CODE.get("TTT"), Some(&'F')); // Phenylalanine
    }

    #[test]
    fn test_codon_graph_structure() {
        // Graph should have 61 entries (one per codon)
        assert_eq!(CODON_GRAPH.edges.len(), 61);

        // Each codon should have exactly 9 neighbors:
        // 3 positions × 3 alternative nucleotides = 9
        // (some may be slightly less due to stop codons)
        for (i, neighbors) in CODON_GRAPH.edges.iter().enumerate() {
            assert!(neighbors.len() <= 9,
                   "Codon {} ({}) has {} neighbors, expected <= 9",
                   i, index_to_codon(i), neighbors.len());
            assert!(neighbors.len() >= 6,
                   "Codon {} ({}) has only {} neighbors, expected >= 6",
                   i, index_to_codon(i), neighbors.len());
        }
    }

    #[test]
    fn test_codon_graph_edges() {
        // Test a specific example: TTT (Phe)
        let ttt_idx = codon_to_index("TTT").unwrap();
        let neighbors = CODON_GRAPH.neighbors(ttt_idx);

        // TTT can change to:
        // Position 0: CTT, ATT, GTT
        // Position 1: TCT, TAT, TGT
        // Position 2: TTC, TTA, TTG
        assert_eq!(neighbors.len(), 9);

        // Find TTC edge (should be synonymous and IS a transition T->C)
        let ttc_idx = codon_to_index("TTC").unwrap();
        let ttc_edge = neighbors.iter()
            .find(|e| e.to_codon == ttc_idx)
            .expect("Should have edge to TTC");

        assert!(ttc_edge.is_synonymous, "TTT->TTC should be synonymous (both Phe)");
        assert!(ttc_edge.is_transition, "T->C IS a transition");
    }

    #[test]
    fn test_codon_graph_transition_transversion() {
        // ATG (Met) can change to:
        // Position 0: TTG, CTG, GTG (all transversions except GTG which is transition)
        // Position 1: AAG, ACG, AGG (AAG and AGG are transitions)
        // Position 2: ATA, ATC, ATT (none are transitions)

        let atg_idx = codon_to_index("ATG").unwrap();
        let neighbors = CODON_GRAPH.neighbors(atg_idx);

        // Check GTG edge (A->G transition at position 0)
        let gtg_idx = codon_to_index("GTG").unwrap();
        let gtg_edge = neighbors.iter()
            .find(|e| e.to_codon == gtg_idx)
            .unwrap();
        assert!(gtg_edge.is_transition, "A->G should be transition");

        // Check AAG edge (T->A transversion at position 1) - wait, T->A is not a transition
        let aag_idx = codon_to_index("AAG").unwrap();
        let aag_edge = neighbors.iter()
            .find(|e| e.to_codon == aag_idx)
            .unwrap();
        // ATG -> AAG is T->A which is a transversion
        assert!(!aag_edge.is_transition, "T->A should be transversion");
    }

    #[test]
    fn test_codon_graph_symmetry() {
        // If there's an edge i->j, there should be an edge j->i
        for i in 0..61 {
            for edge in CODON_GRAPH.neighbors(i) {
                let j = edge.to_codon;

                // Find reverse edge
                let reverse_edge = CODON_GRAPH.neighbors(j).iter()
                    .find(|e| e.to_codon == i)
                    .expect(&format!("No reverse edge from {} to {}", j, i));

                // Properties should match
                assert_eq!(edge.is_transition, reverse_edge.is_transition);
                assert_eq!(edge.is_synonymous, reverse_edge.is_synonymous);
            }
        }
    }
}
