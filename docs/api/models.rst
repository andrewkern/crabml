Models
======

Codon substitution model classes.

.. currentmodule:: crabml.models

Codon Models
------------

.. currentmodule:: crabml.models.codon

M0CodonModel
~~~~~~~~~~~~

.. autoclass:: M0CodonModel
   :members:
   :special-members: __init__

Site-Class Codon Models
~~~~~~~~~~~~~~~~~~~~~~~

Additional model classes:

* ``M1aCodonModel`` - Nearly neutral model
* ``M2aCodonModel`` - Positive selection model
* ``M3CodonModel`` - Discrete omegas
* ``M7CodonModel`` - Beta distribution
* ``M8CodonModel`` - Beta + omega
* ``M8aCodonModel`` - Beta + omega=1
* ``M4CodonModel`` - Fixed omegas with variable frequencies
* ``M5CodonModel`` - Gamma distribution
* ``M6CodonModel`` - 2Gamma distribution
* ``M9CodonModel`` - Beta & Gamma

All models follow similar interfaces with model-specific parameters.
