.. image:: https://img.shields.io/pypi/v/druhg.svg
    :target: https://pypi.python.org/pypi/druhg/
    :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/druhg.svg
    :target: https://github.com/artamono1/druhg/blob/master/LICENSE
    :alt: License

=====
DRUHG
=====

DRUHG — Dialectical Reflection Universal Hierarchical Grouping (друг).

Density-based clustering that catches global outliers and lets you navigate the cluster hierarchy visually. It does not require clustering hyperparameters. The space metric (e.g. Euclidean) is the only real choice.

Optional knobs:

- ``size_range`` — filter cluster size; use ``[1, 1]`` for genuine outliers.
- ``fix_outliers`` — assign outliers to their closest clusters along MST edges.
- ``max_ranking`` — neighbor-search depth; trade speed for precision.

-------------
Basic Concept
-------------

The algorithm applies **the universal society rule: treat others as you want to be treated**.

Point A inspects the surroundings of point B and converts that view into its own. Each pair ``A, B`` produces a **dialectical distance** ``max( r/(R-∩) d(r); d(R) )``, where ``r`` and ``R`` are the numbers of points inside the balls from A to B and from B to A. The closest dialectical distance wins and crystallizes into an edge. The process repeats.

This orders outliers last and equal densities first. It is a strong EDA (exploratory data analysis) method, a replacement for (H)DBSCAN, and a global-outlier detector.

Evaluating all ``O(n²)`` pairs is unnecessary; only a small number of nearest neighbors matters. Control speed vs precision with ``max_ranking`` — after some ``k`` the result converges.

**The cluster.** The coloring formula is easiest to see through graphs and the nature of mathematical objects. Points *are*, edges *connect*, and **the dictionary of key–value pairs** (point-to-edge) "*colors*". When two graphs connect, the two sets of points can be linked to the connecting edge:

1. Both graphs clusterize to the same edge; In case of mutual clusterisation it is seen as one cluster.
2. One graph clusterizes; its points link to the connecting edge (a regular cluster).
3. No clusterisation. Everything aggregates.

Each graph reflects in its rival and solves the inequality:

``D N₂ L₁ ∑₁ 1 / dᵢ > N₁ (L₁ + L₂)``

where:

- ``D`` — dialectical distance of the connecting edge
- ``N₁``, ``N₂`` — own and rival sides of a graph
- ``L₁``, ``L₂`` — unique linked edges
- ``∑₁ 1 / dᵢ `` — sum of reciprocals of unique linked edges

A newly formed cluster resists reclusterisation with its internal high ``dᵢ`` and low ``L₁``. Outliers bring ``1`` as ``N₂``, contribute ``1`` to ``L₂``. Eventually a huge external ``D``, ``N₂``, or dilution of ``L₁`` will clusterize anything.

This is drastically different from the usual overcome-xyz coefficient.

----------------
How to use DRUHG
----------------

.. code:: python

    import sklearn.datasets as datasets
    from sklearn.metrics import adjusted_rand_score
    import druhg

    iris = datasets.load_iris()
    XX = iris['data']

    clusterer = druhg.DRUHG(max_ranking=50)
    labels = clusterer.fit(XX).labels_

This builds the tree and labels the points. You can then reshape clusters by relabeling:

.. code:: python

    labels = clusterer.relabel(exclude=[7749, 100], size_range=[0.2, 2242], fix_outliers=True)
    ari = adjusted_rand_score(iris['target'], labels)
    print('iris ari', ari)

Relabeling is cheap:

- ``exclude`` — break clusters by label number
- ``size_range`` — restrict cluster size by fraction (values ``< 1``) or by absolute count
- ``fix_outliers`` — color outliers by connectivity

Draw the MST with DRUHG edges:

.. code:: python

    clusterer.plot(labels)

Or open interactive sliders for exploration:

.. code:: python

    clusterer.plot()

.. image:: https://raw.githubusercontent.com/artamono1/druhg/master/docs/source/pics/chameleon-sliders.png
   :width: 300px
   :align: center
   :height: 200px
   :alt: chameleon-sliders

-----------
Performance
-----------

It can be slow on highly structured data. Lower ``max_ranking`` for better performance.

.. image:: https://raw.githubusercontent.com/artamono1/druhg/master/docs/source/pics/comparison_ver.png
    :width: 300px
    :align: center
    :height: 200px
    :alt: comparison

----------
Installing
----------

PyPI install, assuming an up-to-date pip:

.. code:: bash

    pip install druhg

-----------------
Running the Tests
-----------------

After installation:

.. code:: bash

    pytest druhg/tests -k "test_name"

The tests may fail.

.. code:: bash

    pytest druhg/tests -k "test_name" -v

For a verbose logging


.. code:: bash

    pytest druhg/tests -k "test_name" -log=DEBUG

For a deep dive

--------------
Python Version
--------------

DRUHG supports Python 3.

------------
Contributing
------------

Contributions in any form are welcome. Help with documentation, especially tutorials, is always useful. Fork the project, make your changes, and submit a pull request:

https://github.com/artamono1/druhg

---------
Licensing
---------

The druhg package is 3-clause BSD licensed.
