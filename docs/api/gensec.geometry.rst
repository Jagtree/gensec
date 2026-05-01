.. _api_geometry:

===========================
:mod:`gensec.geometry`
===========================

Section definition, fiber meshing, parametric shape factories, and
ideal_gross geometric-property computation.

A cross-section in GenSec is represented as a collection of **bulk
fibers** (from the meshed polygon) and **point fibers** (rebars,
tendons, FRP strips).  The :class:`~gensec.geometry.geometry.GenericSection`
class is the primary section object; :class:`~gensec.geometry.section.RectSection`
is a backward-compatible wrapper.

The *ideal_gross* geometric properties — area, centroid, centroidal and
principal second-moments, central inertia ellipse, kern — are
computed exactly on the polygon (independently of the fiber mesh)
by the module :mod:`gensec.geometry.properties`.  See the
dedicated page :doc:`ideal_gross_properties` for the mathematical
formulation and usage.


Point fibers (rebars)
----------------------

.. automodule:: gensec.geometry.fiber
   :members:
   :show-inheritance:


Generic section with polygon meshing
--------------------------------------

.. automodule:: gensec.geometry.geometry
   :members:
   :show-inheritance:


Parametric section primitives
------------------------------

Factory functions that return :class:`shapely.geometry.Polygon` objects
for common section shapes.  Each polygon can be passed directly to
:class:`~gensec.geometry.geometry.GenericSection`.

.. automodule:: gensec.geometry.primitives
   :members:


ideal_gross geometric properties
---------------------------------

Exact polygon integrals (Green's theorem), centroidal and
principal second-moments, central inertia ellipse, kern.
Documented in detail in :doc:`ideal_gross_properties`.

.. automodule:: gensec.geometry.properties
   :members:
   :show-inheritance:


Legacy rectangular section
---------------------------

.. automodule:: gensec.geometry.section
   :members:
   :show-inheritance:
