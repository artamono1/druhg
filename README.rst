.. image:: https://img.shields.io/pypi/v/druhg.svg
    :target: https://pypi.python.org/pypi/druhg/
    :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/druhg.svg
    :target: https://github.com/artamono1/druhg/blob/master/LICENSE
    :alt: License

=====
DRUHG
=====

| DRUHG - Dialectical Reflection Universal Hierarchical Grouping (друг).
| Performs clustering based on densities and builds a minimum spanning tree.
| **Does not require parameters.** *(The parameter is space metric, e.x. euclidean)*
| The user can filter the size of the clusters with ``size_range``, for genuine result and genuine outliers set to [1,1].
| Parameter ``fix_outliers`` allows to label outliers to their closest clusters via mstree edges.

-------------
Basic Concept
-------------

| There are some optional tuning parameters but the actual algorithm requires none and is universal.
| It works by applying **the universal society rule: treat others how you want to be treated**.
| The core of the algorithm is to rank the subject's closest subjective similarities and amalgamate them accordingly.
| Parameter ``max_ranking`` controls precision vs productivity balance, after some value the precision and the result would not change.
| todo: Parameter ``algorithm`` can be set to 'slow' to further enhance the precision.
|
|
| The **dialectical distance** reflects the opposite density.
| Max( r/R d(r); d(R) ), where r and R are ranks from A to B and from B to A.
| This orders outliers last and equal densities first.
| It's great **replacement for DBSCAN** and **global outliers detection**.
|
| Those ordered connections become trees. Two trees reflect of each other in their totality and can transfrom into cluster.
| D N₂ K₁/(K₁+K₂) sum 1/dᵢ > N₁ - 1, where N is size of a tree, K is number of clusters in a tree.
| This allows newly formed clusters to resist the reshaping.


----------------
How to use DRUHG
----------------
.. code:: python

             import sklearn.datasets as datasets
             import druhg

             iris = datasets.load_iris()
             XX = iris['data']

             clusterer = druhg.DRUHG(max_ranking=50)
             labels = clusterer.fit(XX).labels_

It will build the tree and label the points. Now you can manipulate clusters by relabeling.

.. code:: python

             labels = dr.relabel(exclude=[7749, 100], size_range==[0.2, 2242], fix_outliers=1)
             ari = adjusted_rand_score(iris['target'], labels)
             print ('iris ari', ari)

Relabeling is cheap.
 - ``exclude`` breaks clusters by label number,
 - ``size_range`` restricts cluster size by percent or by absolute number,
 - ``fix_outliers`` colors outliers by connectivity.

.. code:: python

            clusterer.plot(labels)

It will draw mstree with druhg-edges.

.. code:: python

            clusterer.plot()

It will provide interactive sliders for an exploration.

.. image:: https://raw.githubusercontent.com/artamono1/druhg/master/docs/source/pics/chameleon-sliders.png
    :width: 300px
    :align: center
    :height: 200px
    :alt: chameleon-sliders

-----------
Performance
-----------
| It can be slow on a highly structural data.
| There is a parameters ``max_ranking`` that can be used to decrease for a better performance.

.. image:: https://raw.githubusercontent.com/artamono1/druhg/master/docs/source/pics/comparison_ver.png
    :width: 300px
    :align: center
    :height: 200px
    :alt: comparison

----------
Installing
----------

PyPI install, presuming you have an up to date pip:

.. code:: bash

    pip install druhg


-----------------
Running the Tests
-----------------

The package tests can be run after installation using the command:

.. code:: bash

    pytest -k "test_name"


The tests may fail :-D

--------------
Python Version
--------------

The druhg library supports Python 3.


------------
Contributing
------------

We welcome contributions in any form! Assistance with documentation, particularly expanding tutorials,
is always welcome. To contribute please `fork the project <https://github.com/artamono1/druhg/issues#fork-destination-box>`_
make your changes and submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

---------
Licensing
---------

The druhg package is 3-clause BSD licensed.
