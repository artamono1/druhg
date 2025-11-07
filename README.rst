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
| Performs clustering based on densities, catches global outliers and allows visual cluster hierarchy navigation.
| **Does not require parameters.** *(The parameter is metric of space, e.x. euclidean)*
| The user can filter the size of the clusters with ``size_range``, for genuine result/outliers set to [1,1].
| Parameter ``fix_outliers`` allows to label outliers to their closest clusters via mstree edges.

-------------
Basic Concept
-------------

| The algorithm works by applying **the universal society rule: treat others how you want to be treated**.
| The A point checks surrondings of the point B and converts it in it's own point of view.
| Each pair A,B produces **dialectical distance** max( r/R d(r); d(R) ), 
| where r and R are amounts of points inside of balls from A to B and from B to A.
| The closest distance wins and crystalizes into edge. Process repeats.

| This orders outliers last and equal densities first. The best EDA *Exploratory Data Analysis*.
| It's great **replacement for (H)DBSCAN** and **global outliers detection**.
| The expensiveness O(n*n) of all points pairs evaluation is not needed, only low k-neighbors matters.
| Therefore you can control productivity vs precision with ``max_ranking`` parameter, after some k the result converges.

| **The cluster**.
| The formula behind clusters' coloring best explain through graphs and the nature of maths objects.
| The points *are*, the edges *connects*, and **the dictionary of key-value pairs** point-to-edge "*colors*".
| And when two graphs connects, then two sets of points can be linked to the connecting edge.
| 1. Both graphs clusterize to same edge, for a future connections it will be one cluster.
| 2. One graph clusterizes, its' points link to the connecting edge. E.g. regular cluster.
| 3. No clusterisation. Everything aggregates. The connecting edge is not pointed by any point, and it doesn't have a color.

| Each graph reflects in it's rival and solves mathematical inequality:
| D N₂ L₁ sum₁ (nᵢ-1')/nᵢdᵢ > l₁(L₁+L₂), where D - dialectical distance of connecting edge;
| N₂ - rival's points; L₁, L₂ - unique linked edges;
| dᵢ - dialectical distance of the pointed edge (sum is iterated over unique pointed edges L₁, not their dd values);
| nᵢ-1' - amount of edges of one color (usually edges = points - 1, except outliers they have 1edge and 1 point);
| thus (nᵢ-1)/nᵢdᵢ when nᵢ>1 and 1/dᵢ when nᵢ=1;
| l₁ - amount of "colored" edges, every edge of a linked subgraph is counted, edges from no clusterisation example are not.

| Newly formed cluster resists reclusterisation with it's internal high dᵢ, high l₁ and low L₁.
| Outliers bring 1 as N₂, contribute 1 to L₂ and they are easily countered with l₁.
| External eventually huge D or N₂ or dillution of L₁ will clusterize anything.
| This approach is drastically different from an usual overcome 2. coefficient.



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
