      seqfile = ../paml_examples/lysozyme/lysozymeSmall.txt
     treefile = ../paml_examples/lysozyme/lysozymeSmall.trees
      outfile = lysozyme_m7_out.txt

        noisy = 3
      verbose = 1
      runmode = 0

      seqtype = 1   * 1:codons
    CodonFreq = 2   * 0:1/61 each, 1:F1X4, 2:F3X4
        clock = 0
        model = 0   * 0: M0 (one dN/dS ratio for all branches)

      NSsites = 7   * 7: M7 (beta distribution)
        icode = 0   * 0:standard genetic code

    fix_kappa = 0   * 0: estimate kappa
        kappa = 2   * initial kappa
    fix_omega = 0   * 0: estimate omega
        omega = 0.4 * initial omega

    fix_alpha = 1   * 1: fix alpha (no rate variation)
        alpha = 0.  * 0: constant rate
       Malpha = 0
        ncatG = 10  * number of categories for discretizing beta

        getSE = 0
 RateAncestor = 0

  fix_blength = 0  * 0: estimate branch lengths
       method = 0
