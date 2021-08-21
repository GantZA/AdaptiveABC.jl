using Distributions:Random
using AdaptiveABC
using QuantEcon
using Plots
using Plots.PlotMeasures
using Random
using Distributions
using Optim
using NLopt

# Set Simulator Parameters and Seed
Random.seed!(37162)
ϕ = 0.7
θ = 0.3
σ = 0.01
parameter_true = [ϕ, θ, σ]
n = 1000
p0 = 100
parameter_names = ["phi", "theta", "sigma"]

# Create model, simulate to obtain observation vector and then calculate summary statistics
true_arma = ARMA(ϕ, θ, σ)
obs = simulation(true_arma, ts_length=n)
obs_summary_stats = get_summary_stats(obs)

prices = zeros(n + 1)
prices[1] = p0
for i in 1:n
    prices[i + 1] = exp(obs[i]) * prices[i]
end

# l = @layout [a b]

# p1 = plot(1:n, obs, legend=false,
#     xlab="Time", ylab="ARMA Log Returns"
# );

# p2 = plot(1:n + 1, prices, legend=false,
# xlab="Time", ylab="ARMA Mid-Price"
# );
# p3 = plot(p1, p2, layout=l, tickfontsize=6, guidefontsize=8,
#     titlefontsize=10, right_margin=5mm, size=(1000, 400)) ;

# savefig(p3, "examples/arma/arma_lr_mp_plot.pdf")

#### Calibration Assessment

#####
# Utility Functions
#####

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters, n_summary_stats, n_replications)
    success = true
    model = ARMA(parameters...)
    sim_obs = Array{Float64,2}(undef, n, n_replications)
    summary_stats = Array{Float64,2}(undef, n_summary_stats, n_replications)
    for i in 1:n_replications
        sim_obs[:, i] = simulation(model, ts_length=n)
        summary_stats[:, i] = get_summary_stats(obs, sim_obs[:, i])
    end
    return success, summary_stats
end

prior = Prior([
    Uniform(0.05, 0.95),
    Uniform(0.05, 0.95),
    Uniform(0.005, 0.1),
])

#####
# Calibration Technique: ABC Rejection BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(1312, obs, get_summary_stats, 100, 10000)
weighted_bootstrap = WeightedBootstrap(obs, get_summary_stats, boot_weight_matrix)


abc_input_wb = ABCInput(
    prior,
    parameter_names,
    obs_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_reject_wb = ABCRejection(
    abc_input_wb,
    100_000,
    10,
    250.0,
    parallel=true,
    seed=817221
)
posterior_means = AdaptiveABC.parameter_means(abc_reject_wb)
println()
println("ABC rejection BBWM: ϕ, θ, σ")
print(round.(posterior_means[:,end], digits=3))
println()

#####
# Calibration Technique: ABC Rejection MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    obs_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(obs, get_summary_stats, ones(17), "MAD")
)

@time abc_reject_we = ABCRejection(
    abc_input_we,
    100_000,
    10,
    12.0,
    parallel=true,
    seed=2125179095
)
posterior_means = AdaptiveABC.parameter_means(abc_reject_we)
println()
println("ABC rejection MADWE: ϕ, θ, σ")
print(round.(posterior_means[:,end], digits=3))
println()
#####
# Calibration Technique: ABC-PMC BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(1312, obs, get_summary_stats, 100, 10000)
weighted_bootstrap = WeightedBootstrap(obs, get_summary_stats, boot_weight_matrix)


abc_input_wb = ABCInput(
    prior,
    parameter_names,
    obs_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_pmc_out_wb = ABC_PMC(
    abc_input_wb,
    500,
    10,
    0.15,
    10000,
    9;
    parallel=true,
    batch_size=250,
    seed=2077971236
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_wb)
print(round.(posterior_means[:,end], digits=3))

#####
# Calibration Technique: ABC-PMC MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    obs_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(obs, get_summary_stats, ones(17), "MAD")
)

abc_pmc_out_we = ABC_PMC(
    abc_input_we,
    100,
    10,
    0.15,
    100_000,
    9;
    parallel=true,
    batch_size=250,
    seed=589181349
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_we)
println("")
println("ABC-PMC MADWE: ϕ, θ, σ")
print(round.(posterior_means[:,end], digits=3))
println("")

#####
# Calibration Technique: Nelder-Mead BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(1312, obs, get_summary_stats, 100, 10000)
weighted_bootstrap = WeightedBootstrap(obs, get_summary_stats, boot_weight_matrix)

function f(x, weight, reps)
    success, sum_stats = summary_fn(x, 17, reps)
    return weight(sum_stats)
end

f_wb(x, grad) = f(x, weighted_bootstrap, 50)
lower = [0.05, 0.05, 0.005]
upper = [0.95, 0.95, 0.1]

nm_reps = 3
results = zeros(3, nm_reps)
Random.seed!(831838178)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, 3)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.xtol_rel = 1e-6
    opt.min_objective = f_wb
    init_params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)

    (minf, minx, ret) = NLopt.optimize(opt, init_params)

    results[:,i] = minx
end
println()
println("Nelder-Mead BBWM: ")
round.(mean(results, dims=2), digits=3)
results
println()

#####
# Calibration Technique: Nelder-Mead MADWE
#####

weighted_mad = WeightedEuclidean(obs, get_summary_stats, ones(17), "MAD")
sim_sum_stats = zeros(17, 1000, 10)
Random.seed!(156271221)
for i in 1:1000
    params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)
    success, sum_stats = summary_fn(params, 17, 10)
    sim_sum_stats[:,i,:] = sum_stats
end

inv_weights = [AdaptiveABC.MAD(sim_sum_stats[i, :, :]) for i in 1:17]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
weights = 1.0 ./ inv_weights

weighted_mad.weights = weights

f_mad(x, grad) = f(x, weighted_mad, 50)
lower = [0.05, 0.05, 0.005]
upper = [0.95, 0.95, 0.1]

nm_reps = 3
results = zeros(3, nm_reps)
Random.seed!(11019388531)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, 3)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.xtol_rel = 1e-6
    opt.min_objective = f_mad
    init_params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)

    (minf, minx, ret) = NLopt.optimize(opt, init_params)

    results[:,i] = minx
end
println()
println("Nelder-Mead BBWM: ")
round.(mean(results, dims=2), digits=3)
results
println()

###################################################################################################################

