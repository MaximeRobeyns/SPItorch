.. _sampling:

Photometry Sampling
###################

This section repeatedly runs a forward model on various physical parameters
(redshift, agn_mass, inclination, tage etc.) to simulate photometric
observations, which we use in the `inference <inference.html>`_ section to train a model to
produce the *reverse* mapping: from photometric observations back to
distributions over physical parameters.

The code for the sampling procedure is in the `spt/modelling` directory.

Forward Model Configuration
---------------------------

Before running the sampling, we need to configure a forward model so that we can
simulate observations. The forward model configuration is found under the
``ForwardModelParams`` class in the ``config.py`` file.

.. py:class:: ForwardModelParams(FilterCheck, ParamConfig, ConfigClass)

   A self-contained description of the parameters to use with a Prospector
   forward model.

   :param list[Filter] filters: the filter library to use
   :param build_obs_fn_t build_obs_fn: A function to turn an observation
                                       (``pd.Series``) into an observation dictionary.

   ----------------------------------------------------------------------------

   :param list[str] model_param_templates: Any standard model parameter
                                           templates to apply.
   :param list[Parameter] model_params: A list of model parameter descriptions.
   :param build_model_fn_t build_model_fn: A function to initialise the
                                           ``SedModel`` from the parameters.

   ----------------------------------------------------------------------------

   :param dict[str, Any] sps_kwargs: keyword arguments for building the sps.
   :param build_sps_fn_t build_sps_fn: function to build the ``sps`` object, returning an ``SSPBasis`` object.

   :Example:

     >>> class ForwardModelParams(FilterCheck, ParamConfig, ConfigClass):
     ...
     ...     # Observations:
     ...
     ...     filters: list[Filter] = FilterLibrary['des']
     ...     build_obs_fn: build_obs_fn_t = spt.modelling.build_obs
     ...
     ...     # Model parameters:
     ...
     ...     model_param_templates: list[str] = ['parametric_sfh']
     ...     model_params: list[Parameter] = [
     ...         Parameter('zred', 0., 0.1, 4., units='redshift, $z$'),
     ...         Parameter('mass', 6, 8, 10, priors.LogUniform, units='mass',
     ...                   log_scale=True, disp_floor=6.),
     ...         Parameter('logzsol', -2, -0.5, 0.19, units='$\\log (Z/Z_\\odot)$'),
     ...         Parameter('dust2', 0., 0.05, 2., units='optical depth at 5500AA'),
     ...         Parameter('tage', 0.001, 13., 13.8, units='Age, Gyr', disp_floor=1.),
     ...         Parameter('tau', 0.1, 1, 100, priors.LogUniform, units='Gyr^{-1}'),
     ...     ]
     ...     build_model_fn: build_model_fn_t = spt.modelling.build_model
     ...
     ...     # SPS parameters:
     ...
     ...     sps_kwargs: dict[str, Any] = {'zcontinuous': 1}
     ...     build_sps_fn: build_sps_fn_t = spt.modelling.build_sps


The forward model parameter descriptions are instances of the ``Parameter``
class, which accepts the following parameters:

- ``name`` is the name of this parameter, which must be recognized by FSPS. See
  <https://dfm.io/python-fsps/current/stellarpop_api/> for these.
- ``range_min`` is the minimum value that you expect this parameter to take.
  Some prior distributions used in Prospector (e.g. the ``ClippedNormal``) will
  reject samples drawn outside this range.
- ``init`` is the initial value to set this parameter to (i.e. the starting
  point for optimisation metohds)
- ``range_max`` is analagous to ``range_min``, and specifies the highest value
  you expect this parameter to take.
- ``prior`` is a priors distribution class from ``prospect.models.priors``.
- ``prior_kwargs`` are keyword arguments provided while initialising the ``prior``
  distribution above.
- ``log_scale`` this is a boolean indicating whether the values follow a
  logarithmic scale.
- ``model_this`` is a boolean parameter which indicates whether or not to model
  this parameter (otherwise, it is treated as a 'fixed' parameter). ``True`` by
  default.
- ``units`` is a readable description of the units for this parameter. Note that
  if this string begins with ``log*`` (e.g. ``log_mass``), then the
  ``range_min``, ``init``, ``range_max`` and ``disp_floor`` are exponentiated
  (base 10`) before use. Note however that the ``prior_kwargs``, if provided,
  are `not` modified: it is up to you to transform them appropirately.
- ``disp_floor`` sets the initial dispertion to use when using clouds of EMCEE
  walkers (only for MCMC sampling with Prospector).



Running the Sampling Procedure
------------------------------


Running ``make sim`` will run the simulation code in the ``__main__`` section of
``agnfinder/inference/inference.py``, using the configurations in
``agnfinder/config.py``.

This is a multi-threaded program which runs on CPUs only. We use processes
rather than threads due to Python's GIL. With :math:`P` processes used to
produce :math:`N` mock photometric observations from using parameters drawn
uniformly at random from the parameter space, each process works to produce
:math:`\frac{N}{P}` 'observations', with the parameter space partitioned among
processes along the redshift dimension.

Each process writes its samples to a 'partial' results file before exiting. All
of these partial results are finally combined (and shuffled) into a final hdf5
file, ready for use in training a model. Generating 1M samples will result in
approximatly 125Mb of data. Since this combination step is run in-memory, do
bear the memory requirements in mind if you are simulating large catalogues
(e.g. 100M samples).

Free Parameter Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *free parameters* are the physical parameters that are allowed to vary. The
default configuration is as follows::

    class FreeParams(FreeParameters):
        redshift: tuple[float, float] = (0., 6.)
        log_mass: tuple[float, float] = (8, 12)
        log_agn_mass: tuple[float, float] = (-7, math.log10(15))
        log_agn_torus_mass: tuple[float, float] = (-7, math.log10(15))
        dust2: tuple[float, float] = (0., 2.)
        tage: tuple[float, float] = (0.001, 13.8)
        log_tau: tuple[float, float] = (math.log10(.1), math.log10(30))
        agn_eb_v: tuple[float, float] = (0., 0.5)
        inclination: tuple[float, float] = (0., 90.)

This defines our prior assumptions about the reasonable range of values that
these parameters may take, and hence the (in this case) 9-dimensional hyper-cube
from which parameters are uniformly sampled during the simulation process.

Parameters whose names begin with ``log_`` are exponentiated before use. For
example, above the ``log_agn_mass`` parameter constrains the brightness of the
galaxy, from :math:`10^{-7}` to :math:`15`.


Sampling Parameters
~~~~~~~~~~~~~~~~~~~

These are parameters relating to the sampling or simulation procedure itself::

    class SamplingParams(ConfigClass):
        n_samples: int = 40000000
        concurrency: int = os.cpu_count()
        save_dir: str = './data/cubes/my_simulation_results'
        noise: bool = False  # deprecated; ignore
        filters: FilterSet = Filters.DES  # {Euclid, DES, Reliable, All}
        shuffle: bool = True  # whether to shuffle final samples

The ``n_samples`` determines the number of samples generated across all the
processes. The ``concurrency`` parameter is best set according to how much RAM
you have on a particular host: since we use *processes* and not *threads*, the
entire model is re-created for each worker and must be stored in memory.

The ``filters`` parameter selects between pre-defined sets of filters, defined
in ``agnfinder/prospector/load_photometry``. To use a different set of filters,
you should edit this file, and add a new filter to the ``Filters`` class in
``agnfinder/types.py``.

You can find various other parameters relating to the model used to simulate the
photometric observations in the ``agnfinder/config.py`` file.

