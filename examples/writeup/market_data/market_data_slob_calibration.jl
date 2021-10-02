using Distributions:Random
using AdaptiveABC
using SequentialLOB
using Random
using Distributions
using Optim
using NLopt
using JLD2
using Plots
using DelimitedFiles

# Set Simulator Parameters and Seed
parameter_names = ["D", "σ", "nu", "μ", "α"]
n_replications = 10
Random.seed!(1717617000301)
# Constants
num_paths = 1; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200 ; λ = 1.0

# Read in Observed Market Data

observed_price_path = round.(readdlm("$(dirname(@__FILE__))/Original_Price_Bars_2300.csv", ',', Float64, '\n')[:,1], digits=2)
observed_log_returns = diff(log.(observed_price_path[:,1]))
observed_summary_stats = get_summary_stats(observed_log_returns)

plt = plot(
    1:T + 1, observed_price_path[:,1], legend=false,
    xlab="Time", ylab="Mid-Price",
)

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters, n_summary_stats, n_replications)
    try
        D, σ, nu, μ, α_slob = parameters
        model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α_slob, SourceTerm(λ, μ))
        summary_stats = Array{Float64,2}(undef, n_summary_stats, n_replications)
        for i in 1:n_replications
            sim_price_path = model()
            sim_obs = diff(log.(sim_price_path[:, 1]))
            summary_stats[:, i] = get_summary_stats(observed_log_returns, sim_obs)    
        end
        return true, summary_stats
    catch e
        return false, zeros(n_summary_stats, n_replications)
    end
end

# Setup for Calibration of Parameters through AdaptiveABC
# Define Prior distribution using Prior struct
prior = Prior([
    Uniform(0.5, 3.0), # D
    Uniform(0.1, 3.0), # σ
    Uniform(0.05, 0.95), # nu
    Uniform(0.1, 3.0), # μ
    Uniform(50.0, 200.0) # α_slob
])

#####
# Calibration Technique: SLOB ABC-PMC BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(9201292, vcat([0], observed_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], observed_log_returns), get_summary_stats, boot_weight_matrix)

abc_input_wb = ABCInput(
    prior,
    parameter_names,
    observed_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_pmc_out_wb = ABC_PMC(
    abc_input_wb,
    100,
    n_replications,
    0.25,
    10_000,
    10;
    parallel=true,
    batch_size=250,
    seed=18291002012543
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_wb)

println()
println("ABC-PMC BBWM: D, σ, nu, μ, α_slob")
print(round.(posterior_means[:,end], digits=3))
println()

JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_wb.jld2", abc_pmc_out_wb)

# ABC-PMC BBWM: D, σ, nu, μ, α_slob 
# [1.957, 1.973, 0.688, 1.648, 13.338]

# param_inds = [1,2,3,4,5]
# plt = plot(abc_pmc_out_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1,5,10], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_wb.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])
# plt
# savefig(plt, "$(dirname(@__FILE__))/abc_pmc_wb.pdf")  


#####
# Calibration Technique: ABC-PMC MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    observed_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(observed_log_returns, get_summary_stats, ones(17), "MAD")
)

abc_pmc_out_we = ABC_PMC(
    abc_input_we,
    100,
    n_replications,
    0.15,
    10_000,
    10;
    parallel=true,
    batch_size=250,
    seed=77261511
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_we)
println("")
println("ABC-PMC MADWE: D, σ, nu, μ, α")
print(round.(posterior_means[:,end], digits=3))
println("")
# ABC-PMC MADWE: D, σ, nu, μ, α_slob
# [1.79, 1.837, 0.693, 1.561, 13.391]

JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_we.jld2", abc_pmc_out_we)

# param_inds = [1,2,3,4,5]
# plt = plot(abc_pmc_out_we, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1,abc_pmc_out_we.n_iterations], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_we.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])
# savefig(plt, "$(dirname(@__FILE__))/abc_pmc_we.pdf")  
# plt



