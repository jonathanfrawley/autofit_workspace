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

dynesty = af.DynestyStatic(
    path_prefix=path.join("plot", "SamplesPlotter"), name="DynestyStatic"
)

result = dynesty.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `SamplesPlotter` and call various `figure_*` methods to plot different plots.
"""
samples_plotter = aplt.SamplesPlotter(samples=samples)

"""
The plotter includes corner plots, using the library corner.py, which summarize the posterior of the results.
"""
samples_plotter.figure_corner(triangle=True)
stop

"""
There are various `figure_*` methods to plot different plots that summarize the quality, speed and results of the 
model-fit.
"""
samples_plotter.figures_1d(progress=True)

"""
Finish.
"""
