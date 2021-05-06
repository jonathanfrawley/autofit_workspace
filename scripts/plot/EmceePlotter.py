"""
Plots: EmceePlotter
=====================

This example illustrates how to plot visualization summarizing the `Emcee` of a non-linear search
using a `EmceePlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autofit.plot as aplt
import model as m
import analysis as a

"""
First, lets create a simple `Emcee` object by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(m.Gaussian)

analysis = a.Analysis(data=data, noise_map=noise_map)

emcee = af.Emcee(
    path_prefix=path.join("plot", "EmceePlotter"),
    name="Emcee",
    nwalkers=100,
    nsteps=10000,
)

result = emcee.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `EmceePlotter`.
"""
emcee_plotter = aplt.EmceePlotter(samples=samples)

"""
The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
"""
emcee_plotter.corner()

"""
We can use the `kwargs` of this function to pass in any of the input parameters, according the API docs 
of `corner.py`:

 - https://corner.readthedocs.io/en/latest/api.html
"""
emcee_plotter.corner(
    bins=20,
    range=None,
    color='k',
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt='.2f',
    title_kwargs=None,
    truths=None,
    truth_color='#4682b4',
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    group='posterior',
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
)


"""
Finish.
"""
