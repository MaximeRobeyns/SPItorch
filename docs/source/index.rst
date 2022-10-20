.. AGN Finder documentation master file


SPItorch documentation
======================

-------------------------------------------------------------------------------

If you're looking for installation instructions, please see the `installation
<installation.html>`_ page.

Here is a link to the `GitHub Repo
<https://github.com/MaximeRobeyns/SPItorch>`_.

-------------------------------------------------------------------------------

This is a fork of the original SPItorch project by Mike Walmsey and
collaborators, which aims to use (conditional) generative modelling techniques to
map from photometry to distributions over physical parameters :math:`\theta`.

There are broadly two components:

1. **Simulation and Dataset Creation**

   The first section is for *simulation*, where you can create a dataset
   of simulated :math:`(\text{galaxy parmeter}, \text{photometry})` pairs.
   This uses `Prospector
   <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_ behind the
   scenes, and should be the first step if you are installing and running this
   repository for the first time.

   See the `Photometry Sampling </sampling.html>`_ section for more details about
   this stage.

2. **Inference**

   In this second section of the codebase we deal with the task of estimating
   the distribution over physical galaxy parameters, which we denote
   :math:`\mathbf{y}`, given photometric observations :math:`\mathbf{x}`; that
   is, estimating
   :math:`p(\mathbf{y} \vert \mathbf{x})`.

   This is a standard supervised machine learning setup. There are several
   candidate models in the codebase, and the code has been structured such that
   it is easy to implement new models, and compare them to the existing methods.

   To see how to do this, and for an overview of the existing models, please
   read the `Inference </inference.html>`_ page.

Configurations
~~~~~~~~~~~~~~

Configurations for the project are set in ``spt/config.py``, not through command
line arguments.


Miscellaneous
~~~~~~~~~~~~~

We use `type hints <https://www.python.org/dev/peps/pep-0484/>`_ throughout the
code to allow for static type checking using `mypy <http://mypy-lang.org/>`_. In
general this helps to catch bugs, and makes it clearer to folks unfamiliar with
the code what an object is, or what a function is doing.

Tests are written using `pytest <https://pytest.org>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :numbered:

   installation
   basic_usage
   sampling
   inference
   san_inference

..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
