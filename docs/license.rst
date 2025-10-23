License
=======

crabML is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).

Full License Text
-----------------

.. code-block:: text

   GNU GENERAL PUBLIC LICENSE
   Version 3, 29 June 2007

   Copyright (C) 2025 Andrew Kern

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.

What This Means
---------------

GPL-3.0-or-later is a copyleft license that:

**Allows:**

* Commercial use
* Modification
* Distribution
* Patent use
* Private use

**Requires:**

* Disclose source code
* License and copyright notice
* State changes
* Same license (copyleft)

**Forbids:**

* Liability
* Warranty

Why GPL?
--------

crabML reimplements PAML's codeml, which is itself GPL-licensed. To ensure
compatibility and to keep this software free and open for the scientific
community, we use the GPL license.

Academic Use
------------

crabML is free to use for academic research. If you use crabML in your
research, please cite:

.. code-block:: text

   Kern, A. D. (2025). crabML: High-performance reimplementation of
   PAML's codeml in Python and Rust. https://github.com/adkern/crabml

Commercial Use
--------------

Commercial use is permitted under GPL-3.0-or-later, provided that:

1. You comply with all GPL requirements (see above)
2. Any derivative works are also GPL-licensed
3. You provide source code to recipients

For alternative licensing arrangements, contact: adkern@uoregon.edu

Third-Party Licenses
--------------------

crabML depends on several open-source libraries:

**Python:**

* NumPy (BSD-3-Clause)
* SciPy (BSD-3-Clause)
* BioPython (Biopython License)
* matplotlib (PSF-based)

**Rust:**

* PyO3 (Apache-2.0 OR MIT)
* ndarray (Apache-2.0 OR MIT)
* Rayon (Apache-2.0 OR MIT)

See ``LICENSE`` file in the repository for complete details.
