"""
Plots: SamplesPlotter
=====================

This example illustrates how to plot visualization summarizing the `Samples` of a non-linear search
using a `SamplesPlotter`.
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
First, lets create a simple `Samples` object by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(m.Gaussian)

analysis = a.Analysis(data=data, noise_map=noise_map)

# emcee = af.Emcee(
#     path_prefix=path.join("plot", "SamplesPlotter"),
#     name="Emcee",
#     nwalkers=100,
#     nsteps=10000,
# )
#
# result = emcee.fit(model=model, analysis=analysis)

dynesty = af.DynestyStatic(path_prefix=path.join("plot"), name="DynestyPlotter2")

result = dynesty.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `SamplesPlotter` and call various `figure_*` methods to plot different plots.
"""
dynesty_plotter = aplt.DynestyPlotter(samples=samples)

"""
The plotter wraps the `cornerplot` method of the inbuilt Dynesty visualization:

 - https://dynesty.readthedocs.io/en/latest/quickstart.html
 -  - https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.plotting
"""
dynesty_plotter.cornerplot()

"""
We can use the `kwargs` of this function to pass in any of the input parameters, according the API docs 
of `dynesty`:

 - https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.plotting
"""
dynesty_plotter.cornerplot(
    dims=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color='black',
    smooth=0.02,
    quantiles_2d=None,
    hist_kwargs=None,
    hist2d_kwargs=None,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color='red',
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
)


"""
Finish.
"""
