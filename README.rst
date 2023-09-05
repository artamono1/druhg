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
| **Does not require parameters.** *(The parameter is metric)*
| The user can filter the size of the clusters with ``limit1`` and ``limit2``.
| To get the genuine result and genuine outliers set ``limit1`` to 1 and ``limit2`` to sample size.
| Parameter ``fix_outliers`` allows to label outliers to their closest clusters via mstree edges.

-------------
Basic Concept
-------------

| There are some optional tuning parameters but the actual algorithm requires none and is universal.
| It works by applying **the universal society rule: treat others how you want to be treated**.
| The core of the algorithm is to rank the subject's closest subjective similarities and amalgamate them accordingly.
| Parameter ``max_ranking`` controls precision vs productivity balance, after some value the precision and the result would not change.
| Parameter ``algorithm`` can be set to 'slow' to further enhance the precision.

| The relationship of two objects sets two local densities, and distorts the distance between them.
| That **dialectical distance** is the reflection - one objects adjusts it's density to fit it's counterpart.
| This allows to arrange all of the relationships into minimal spanning tree.
| Mutual closeness is preferential.

| At the start, unconnected objects amalgamate into Universal and these contradictions define what amalgamation is the cluster.
| The amalgamation has to reflect in the other to emerge as a cluster. The more sizeable adversary the more probable is the change.
| After formation big cluster resists the outliers. This makes it a great algorithm for **outlier detection**.

| *Cluster is a mutually-close reflections.*
| To come up with this universal solution philosophy of dialectical materialism was used.
| You can read more in papers folder

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

             labels = dr.relabel(limit1=1, limit2=len(XX)/2, fix_outliers=1)
             ari = adjusted_rand_score(iris['target'], labels)
             print ('iris ari', ari)

It will relabel the clusters, by restricting their size.

.. code:: python

            from druhg import DRUHG
            import matplotlib.pyplot as plt
            import pandas as pd, numpy as np

            XX = pd.read_csv('chameleon.csv', sep='\t', header=None)
            XX = np.array(XX)
            clusterer = DRUHG(max_ranking=200)
            clusterer.fit(XX)

            plt.figure(figsize=(30,16))
            clusterer.minimum_spanning_tree_.plot(node_size=200)

It will draw mstree with druhg-edges.

.. image:: https://raw.githubusercontent.com/artamono1/druhg/master/docs/source/pics/chameleon.jpg
    :width: 300px
    :align: center
    :height: 200px
    :alt: chameleon

-----------
Performance
-----------
| It can be slow on a highly structural data.
| There is a parameters ``max_ranking`` that can be used to decrease for a better performance.

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

    pytest -s druhg

or

.. code:: bash

    python -m pytest -s druhg

The tests may fail :-D

--------------
Python Version
--------------

The druhg library supports both Python 2 and Python 3. 


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
