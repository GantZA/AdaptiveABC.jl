using AdaptiveABC
using QuantEcon
using Plots
using Random
using Distributions

# Set Simulator Parameters and Seed
Random.seed!(1623)
ϕ = 0.7
θ = 0.3
σ = 0.1
parameter_true = [ϕ, θ, σ]
n = 1000 
parameter_names = ["phi", "theta", "sigma"]

# Create model, simulate to obtain observation vector and then calculate summary statistics
true_arma = ARMA(ϕ, θ, σ)
obs = simulation(true_arma, ts_length=n)
obs_summary_stats = get_summary_stats(obs)

# Setup for Calibration of Parameters through AdaptiveABC
# Define Prior distribution using Prior struct
prior = Prior([
    Uniform(0.05, 0.95),
    Uniform(0.05, 0.95),
    Uniform(0.05, 0.2),
])

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters)
    success = true
    model = ARMA(parameters...)
    sim_obs = simulation(model, ts_length=n)
    summary_stats = get_summary_stats(obs, sim_obs)
    return success, summary_stats
end

a, b = abc_input.summary_fn([0.5, 0.5, 0.1])
b
# Create ABCInput object. Holds the prior, parameters names, observed summary statistics, summary function, the dimension of
# the summary statistics and the datatype for the distance measure. 
abc_input = ABCInput(
    prior,
    parameter_names,
    obs_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean
)

# Perform the ABC Rejection Algorithm (non-parallel) where particles are accepted if they are below
# the distance threshold of 3.0 with a maximum of 20 000 simulations.
@time abc_reject_out_non_parallel = ABCRejection(
    abc_input,
    30000,
    5.0,
    parallel=false
)

# Plot the output of the ABC Rejection using the define plot recipe.
plot(
    abc_reject_out_non_parallel; iteration_colours=cgrad(:blues, 2, categorical=true), 
    iterations=[1], params_true=parameter_true, prior_dists=abc_input.prior.distribution
    )


# Perform the ABC Rejection Algorithm (parallel) where particles are accepted if they are below
# the distance threshold of 2.0 with a maximum of 100 000 simulations.
@time abc_reject_out_parallel = ABCRejection(
    abc_input,
    50000,
    3.5,
    parallel=true
)

plot(
    abc_reject_out_parallel; iteration_colours=cgrad(:blues, 2, categorical=true), 
    iterations=[1], params_true=parameter_true, prior_dists=abc_input.prior.distribution
    )


abc_pmc_out_parallel = ABC_PMC(
    abc_input,
    1000,
    0.5,
    300000,
    20;
    parallel=true,
    batch_size=50000
)


plot(abc_pmc_out_parallel; iteration_colours=cgrad(:blues, 5, categorical=true), 
    iterations=[1, 5, 10, 15, 16], params_true=parameter_true, prior_dists=abc_input.prior.distribution
    )
