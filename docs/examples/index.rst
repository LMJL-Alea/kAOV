Examples
========

The tutorials below are Jupyter notebooks stored in the
`docs/examples <https://github.com/LMJL-Alea/kAOV/tree/main/docs/examples>`_
folder of the GitHub repository.

- :doc:`tutorial_kaov` downloads its data from GitHub, so it only requires
  kAOV to be installed (see :doc:`../installation`) and an internet
  connection.
- :doc:`rabbits_anndata_tutorial` reads a large single-cell dataset
  (``Data/rabbits_ct.h5ad``, about 180 MB) that is not distributed with the
  package. Running it requires a local copy of the repository, as explained
  below.

Running the tutorials from a local copy of the repository
---------------------------------------------------------

1. Install `Git LFS <https://git-lfs.com>`_, which is needed to download
   the large data files, and activate it (once per user account):

   .. code-block:: console

      $ git lfs install

2. Clone the repository and move into it:

   .. code-block:: console

      $ git clone https://github.com/LMJL-Alea/kAOV.git
      $ cd kAOV

   If Git LFS was installed after cloning, download the data files with
   ``git lfs pull``. ``Data/rabbits_ct.h5ad`` must be about 180 MB. A file of
   a few bytes means that the data has not been downloaded.

3. Create and activate a virtual environment:

   .. code-block:: console

      $ python3 -m venv .venv
      $ source .venv/bin/activate

   On Windows, activate it with ``.venv\Scripts\activate`` instead.

4. Install kAOV from the local copy, together with the ``datasets`` extra
   (which provides `scanpy <https://scanpy.readthedocs.io>`_, used by the
   single-cell tutorial) and JupyterLab:

   .. code-block:: console

      $ pip install ".[datasets]" jupyterlab

5. Launch JupyterLab from the folder containing the notebooks (the
   single-cell tutorial loads its data with a path relative to this
   folder):

   .. code-block:: console

      $ cd docs/examples
      $ jupyter lab

6. Open a notebook and run all its cells (*Run* > *Run All Cells*). The
   single-cell tutorial can take 15 minutes or more.

.. toctree::
   :maxdepth: 1

   tutorial_kaov
   rabbits_anndata_tutorial
