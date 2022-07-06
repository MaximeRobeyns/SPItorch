.. _basic_usage:

.. warning:: This section may be out of date. If in doubt, look at the `tutorial
   notebooks
   <https://github.com/MaximeRobeyns/SPItorch/tree/master/tutorial_notebooks>`_
   instead.

Basic Usage
###########

This section gives a series of 'recipes' for doing common tasks. To see more
details about a specific task, find the corresponding section in the sidebar.

In general, most tasks can be accomplished by

1. Updating the relevant parameter class in ``spt/config.py``.
2. Running ``make <task>``. (See ``make help`` for a list of targets.)

Generating a Training Dataset
-----------------------------

The relevant parameter class for this task is ``SamplingParams``, whose default
values may be similar to the following::

    class SamplingParams(ConfigClass):

        n_samples: int = int(10e6)
        concurrency: int = 10  # set this to os.cpu_count() (or slightly less)
        observation: bool = False
        save_dir: str = './data/dsets/example/'
        combine_samples: bool = True  # combine partial samples into one big file?
        cmethod: ConcurrencyMethod = ConcurrencyMethod.MPI  # how to multithread


Now, the sampling (or *simulation*) procedure can be run as configured with
``make sim``. To understand what this is doing, a single worker runs something
like the following---you could run this yourself in a notebook::

    from agnfinder import config as cfg
    from agnfinder.simulation import Simulator

    # Configure the logger (defaults to INFO-level logs)
    cfg.configure_logging()

    # Load the sampling parameters defined in config.py
    sp = cfg.SamplingParams()

    # Initialise a `Simulator` object
    sim = Simulator(rshift_min=sp.redshift_min, rshift_max=sp.redshift_max)

    # Use Latin hypercube sampling for the galaxy parameters
    sim.sample_theta()

    # Create the forward model using Prospector
    sim.create_forward_model()

    # Simulate the photometry
    sim.run()

    # Save the sample results to disk (hdf5 file in sp.save_dir)
    sim.save_samples()

To see more information about the sampling procedure, please consult the
`sampling section </sampling.html>`_.


Training a Model (SAN)
----------------------

The relevant configuration classes are ``SANParams`` as well as
``InferenceParams``. The variable names in the configuration classes are
hopefully descriptive of the configuration option---if not, see the `inference
overview </inference.html>`_ for an overview of the inference process in
``agnfinder``, and in particular see the `SAN </san_inference.html>`_ page for a
description of our architecture.

You can run the training from the command line with ``make san``. Alternatively,
you could run the following code from a notebook::


    import agnfinder.config as cfg
    import agnfinder.nbutils as nbu

    from torchvision import transforms
    from agnfinder.inference import SAN
    from agnfinder.inference import utils

    # Configure the logger (defaults to INFO-level logs)
    cfg.configure_logging()

    # Initialise the inference, and model parameters; defined in config.py
    ip = cfg.InferenceParams()
    sp = cfg.SANParams()

    # Get the dataloaders for training and testing
    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        normalise_phot=(lambda x: np.log(x)),
        transforms=[transforms.ToTensor()])
    logging.info('Created data loaders')

    # Initialise the model
    model = SAN(sp)
    logging.info('Initialised SAN model')

    # Run the training procedure
    model.trainmodel(train_loader, ip)
    logging.info('Trained SAN model')

    # (Example: use the model for something)
    x, _ = nbu.new_sample(test_loader)
    posterior_samples = model.sample(x, n_samples=1000)
    logging.info('Successfully sampled from model')


Inferring Parameters
--------------------

.. TODO:: Write example.
