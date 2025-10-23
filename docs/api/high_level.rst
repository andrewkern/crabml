High-Level API
==============

The high-level API provides simple, user-friendly functions for common analyses.

.. currentmodule:: crabml

Model Optimization Functions
-----------------------------

optimize_model
~~~~~~~~~~~~~~

.. autofunction:: optimize_model

optimize_branch_model
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: optimize_branch_model

optimize_branch_site_model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: optimize_branch_site_model

Result Classes
--------------

SiteModelResult
~~~~~~~~~~~~~~~

.. autoclass:: SiteModelResult
   :members:
   :inherited-members:
   :special-members: __init__

BranchModelResult
~~~~~~~~~~~~~~~~~

.. autoclass:: BranchModelResult
   :members:
   :inherited-members:
   :special-members: __init__

BranchSiteModelResult
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BranchSiteModelResult
   :members:
   :inherited-members:
   :special-members: __init__

ModelResultBase
~~~~~~~~~~~~~~~

.. autoclass:: ModelResultBase
   :members:
   :special-members: __init__

ModelResult (Deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: ModelResult

   Alias for :class:`SiteModelResult` for backwards compatibility.

   .. deprecated:: 0.2.1
      Use :class:`SiteModelResult` instead.
