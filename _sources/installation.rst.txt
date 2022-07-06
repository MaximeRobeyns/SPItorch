.. _installation:

Installation Guide
##################

Quickstart
----------

To install the project, run::

    git clone https://github.com/maximerobeyns/spitorch
    cd spitorch
    make install

Please note that if you wish to install the library within a virtual
environment, you should activate it before running `make install` above.

Then, to make sure everything works, run::

    make test

If everything worked well, you can now take a look at the `basic usage guide
</basic_usage.html>`_.

If something went wrong, consult the `Troubleshooting`_ section below for common
issues. If things went *very* wrong or you're installing on an unconventional
setup, see the `Manual Installation`_ guide.

Troubleshooting
===============

Some common issues

1. **Python 3.9**

   The project needs Python 3.9 or later to run.

   You can check your currently installed version by running::

       python --version

   Follow the instructions in the `Building Python 3.9`_ section for help if
   you have an older version.


2 **Specific Python Path**

  If the standard python executable (i.e. the result of running ``which
  python``) is different from the Python >= 3.9 executable that you want to
  use, then update the ``PYTHON`` variable in the ``Makefile`` to point to
  your desired python path, before running ``make install``::

    # Makefile, around line 28
    PYTHON = /absolute/path/to/your/python3.9

Optional Steps
==============

Shell Setup
~~~~~~~~~~~

Every time you run the project, or want to develop using linters and type
checkers, you should activate the virtual environment and set some necessary
environment variables (such as ``SPS_HOME``).

This will be done automatically for you when running targets through the
``Makefile``, but if you're running things directly, then you can easily
setup your shell by running::

    source setup.sh

To deactivate the virtual environment, either exit your terminal, or type
``deactivate``.

Full tests
~~~~~~~~~~

Running ``make test`` after installation only runs a subset of the tests which
don't take too long to run. If you want to run the slow running tests too, then
you can do this with ``make alltest``.


Manual Installation
-------------------

We first need to setup two data files. The smaller of the two is bundled with
the repository as a tarball, and should be extracted using::

    tar xzf ./data/cpz_paper_sample_week3.parquet.tar.gz
    mv cpz_paper_sample_week3.parquet ./data

The second file is much larger and needs to be downloaded separately. It is (for
now) kindly hosted by the good folks behind the `clumpy
<https://www.clumpy.org/>`_ project, and you can download it with::

	wget -O ./data/clumpy_models_201410_tvavg.hdf5 \
            https://www.clumpy.org/downloads/clumpy_models_201410_tvavg.hdf5

In order to install ``python-fsps`` (which the project uses), we need to export
an ``SPS_HOME`` environment variable to point a directory where `FSPS
<https://github.com/cconroy20/fsps>`_'s source code lies. In this case, we will
install it in the ``./deps`` directory::

    export SPS_HOME=$(pwd)/deps/fsps

We then get the ``FSPS`` source code by cloning the repository into the
``./deps`` directory::

    git clone https://github.com/cconroy20/fsps.git ./deps/fsps

We now create virtual environment called ``spitorch``::

    python3.9 -m venv spitorch

If you prefer to use conda, or install the dependencies somewhere else (for
instance if the current directory is mounted on an NFS), then you can make this
change here.

We then activate the virtual environment, upgrade pip, and install the
dependencies in ``requirements.txt``::

    source spitorch/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Note that the ``python`` executable above should be the one in the virtual
environment.

We now need to copy some custom sedpy filters into sedpy's filter directory
(which should be a subdirectory of the ``spitorch``). To find the location of
this filter directory, drop into a python shell (with the venv activated), and
run::

    >>> import sedpy
    >>> print(sedpy.__file__)
    /path/to/spitorch/spitorch/lib/python3.9/site-packages/sedpy/__init__.py

This points us to sedpy's installation directory; we want to copy the filters in
``./filters`` to the ``<sedpy-base>/data/filters/`` directory. That is::

    cp -n ./filters/* \
        /path/to/spitorch/spitorch/lib/python3.9/site-packages/sedpy/data/filters/

Now we can install ``spitorch`` itself, by running the following from the root
of the repository::

    pip install -e .

Writing Documentation
---------------------

The documentation for this project is written in `sphinx
<https://www.sphinx-doc.org/en/master/>`_, inside a Docker container.

To write documentation, you should install the additional dependencies with::

    pip install -e .[docs]

You can then run::

    make docs

which will compile the HTML documentation, open it in your browser, and watch
both the documentation source files (in ``/docs/``) as well as the source files
in ``/spt`` for changes.


Building Python 3.9
-------------------

This is an optional step if you do not have Python 3.9 available on the system
you intend to run ``spitorch`` on. Here we will assume that you do not have
root privileges.

First, download a Python>=3.9 source code release in some convenient directory.
You could choose to work in ``/tmp``, or any other directory (ideally on your
target machine / architecture). At the time of writing, the latest release can
be downloaded with::

    wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz

Extract this and go into the source directory::

    tar xzf Python-3.9.6.tgz
    cd Python-3.9.6

We now follow a fairly standard ``./configure && make && make install`` build
procedure. Since we assume that we don't have root privileges, we will
explicitly specify the desired installation prefix during the configuration
stage, as well as providing some other python-specific options::

    ./configure --enable-optimizations --with-ensurepip=install --prefix=$HOME

If you wish to install to another prefix (for instance, you don't want the
resulting executables on some NFS), then replace ``$HOME`` with an appropriate
alternative for your system.

Building and installing is now straightforward::

    make -j<nprocs>
    make install

where ``<nprocs>`` is the number of processes that you are happy to run
concurrently. If compiling on a login node, remember be mindful of other users!

Anaconda
--------

Perhaps an easier way to install Python 3.9 on a machine that does not have it
is to use a package manager like `Anaconda <https://www.anaconda.com/>`_ with
packages written to your home directory.

If you do not have root access, set up a ``.condarc`` file to set the prefix to
your conda environments to a folder in your home directory::

  # file ~/.condarc

  envs_dirs:
    -  ~/path/to/conda_envs

You can now create a virtual environment with Python 3.9::

  conda create spitorch_env Python=3.9

You can then activate the environment with::

  conda activate spitorch_env
