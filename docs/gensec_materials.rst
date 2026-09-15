.. _api_materials:

=============================
:mod:`gensec.materials`
=============================

Constitutive material laws and property tables.

The abstract base class :class:`~gensec.materials.base.Material` defines
the interface that all materials must implement.  Four concrete
implementations are provided — :class:`Concrete`,
:class:`ServiceConcrete`, :class:`Steel`, and
:class:`TabulatedMaterial` — plus bridge modules for automatic
property lookup from European standards.

:class:`Concrete` and :class:`ServiceConcrete` are not two flavours of
the same thing.  The first is a **design envelope**, calibrated so that
its integral gives the right resultant and lever arm at collapse; its
slope at the origin carries :math:`\gamma_c` and :math:`\alpha_{cc}`
and has no physical meaning.  The second is what a concrete **does**:
expression (3.14) in compression, a linear branch to a cut at
:math:`f_{ct,\mathrm{eff}}` in tension, no partial factors anywhere.
A model declares which one it means; neither is a default.


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


Concrete — service law, EN 1992-1-1 (3.14)
-------------------------------------------

.. automodule:: gensec.materials.service_concrete
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
   :noindex:
   :show-inheritance:
