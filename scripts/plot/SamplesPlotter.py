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

import autofit as af
import autofit.plot as aplt
from autofit.non_linear import samples as samp

"""
First, lets create a simple `Samples` object.
"""
sample_0 = samp.Sample(log_likelihood=-1000.0, log_prior=2.0, weights=0.25)
sample_1 = samp.Sample(log_likelihood=1.0, log_prior=5.0, weights=0.75)

samples = af.PDFSamples(model=af.Mapper(), samples=[sample_0, sample_1])

"""
We now pass the samples to a `SamplesPlotter` and call various `figure_*` methods to plot different plots that 
summarize the quality, speed and results of the model-fit.
"""
imaging_plotter = aplt.SamplesPlotter(samples=samples)
imaging_plotter.figures_1d(progress=True)

"""
Finish.
"""
