.. _api_materials:

=============================
:mod:`gensec.materials`
=============================

Constitutive material laws and property tables.

The abstract base class :class:`~gensec.materials.base.Material` defines
the interface that all materials must implement.  Three concrete
implementations are provided — :class:`Concrete`,
:class:`Steel`, and :class:`TabulatedMaterial` —
plus bridge modules for automatic property lookup from European standards.


Abstract base
-------------

.. automodule:: gensec.materials.base
   :members:
   :show-inheritance:


Concrete — parabola-rectangle
-------------------------------

.. automodule:: gensec.materials.concrete
   :members:
   :show-inheritance:


Reinforcing steel — elastic-plastic
-------------------------------------

.. automodule:: gensec.materials.steel
   :members:
   :show-inheritance:


Tabulated material
-------------------

.. automodule:: gensec.materials.tabulated
   :members:
   :show-inheritance:


EC2 property tables
--------------------

Full implementation of EN 1992-1-1 Table 3.1 parameters, including
high-strength concrete (:math:`f_{ck} > 50\;\text{MPa}`) and the French
National Annex.

.. automodule:: gensec.materials.ec2_properties
   :members:
   :show-inheritance:


EN 10025-2 structural steel
-----------------------------

Thickness-dependent yield and ultimate strength for carbon structural
steels (S235, S275, S355).

.. automodule:: gensec.materials.en10025_properties
   :members:
   :show-inheritance:


Bridge / factory functions
---------------------------

Factory functions that create :class:`Concrete` and :class:`Steel`
objects from the EC2 and EN 10025 property classes, extracting all
constitutive parameters automatically.

.. automodule:: gensec.materials.ec2_bridge
   :members:
   :show-inheritance:


Package-level exports
----------------------

.. automodule:: gensec.materials
   :members:
   :show-inheritance:
