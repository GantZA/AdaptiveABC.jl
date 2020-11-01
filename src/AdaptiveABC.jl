module AdaptiveABC

using Distributions
using ProgressMeter
using StatsBase: mean, quantile, autocor, Weights, ecdf, pacf
using Printf
using Distances: KLDivergence
using DelimitedFiles
using Random
using Statistics: median, var
using Base.Threads
using LinearAlgebra: norm
using RecipesBase
using PlotUtils
using Measures
using IPMeasures: mmd, GaussianKernel

import Base.show, Base.copy, Base.rand
import Distributions.pdf
import HypothesisTests.ksstats


include("bootstrap_weight.jl")
include("distances.jl")
include("abc_types.jl")

include("abc_pmc.jl")
include("abc_rejection.jl")
include("plots.jl")
include("summary.jl")


export 
    WeightedEuclidean, # Distance Measure Datatypes
    WeightedBootstrap,
    ABCInput, ABCOutput, # ABC Input and Output Types
    ABC_PMC, # ABC Algorithms
    ABCRejection, # ABC Algorithms
    Prior,  # Prior struct for parameter prior distributions
    get_summary_stats,
    BlockBootstrapWeightMatrix


end # module
