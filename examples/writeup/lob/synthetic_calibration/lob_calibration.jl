using Distributions:Random
using AdaptiveABC
using SequentialLOB
using Random
using Distributions
using Optim
using NLopt
using JLD2
using Plots

# Set Simulator Parameters and Seed
D = 1.0
σ = 1.5
nu = 2.0
μ = 1.0
parameter_true = [D, σ, nu, μ]
parameter_names = ["D", "σ", "nu", "μ"]
Random.seed!(8929200)
# Constants
num_paths = 1; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200
Δx = L / M ; λ = 1.0 ; Δt = (Δx^2) / (2.0 * D) ; α_lob = 0.0
println("Δx = $Δx and Δt = $Δt")



# Create model, simulate to obtain observation vector and then calculate summary statistics
true_lob_model = SLOB(num_paths,
    T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))
true_lob_price_path = true_lob_model(7136)
true_lob_log_returns = diff(log.(true_lob_price_path[:,1]))
true_lob_summary_stats = get_summary_stats(true_lob_log_returns)

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters, n_summary_stats, n_replications)
    try
        D, σ, nu, μ = parameters
        model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))
        sim_price_path = model()
        sim_obs = diff(log.(sim_price_path[:, 1]))
        summary_stats = Array{Float64, 2}(undef, n_summary_stats, n_replications)
        summary_stats[:, 1] = get_summary_stats(true_lob_log_returns, sim_obs)
        return true, summary_stats
    catch e
        return false, zeros(17)
    end
end

# Setup for Calibration of Parameters through AdaptiveABC
# Define Prior distribution using Prior struct
prior = Prior([
    Uniform(0.5, 3.0), # D
    Uniform(0.1, 3.0), # σ
    Uniform(0.5, 5.0), # nu
    Uniform(0.1, 3.0) # μ
])

#####
# Calibration Technique: ABC Rejection BBWM
#####

# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(122477, vcat([0], true_lob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_lob_log_returns), get_summary_stats, boot_weight_matrix)



abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_reject_wb = ABCRejection(
    abc_input_wb,
    3_000,
    1,
    100.0,
    parallel=true
)

posterior_means = AdaptiveABC.parameter_means(abc_reject_wb)
println()
println("ABC rejection BBWM: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

param_inds = [1,2,3,4]
plt = plot(abc_reject_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_wb.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=parameter_names[param_inds])

savefig(plt, "$(dirname(@__FILE__))/abc_reject_wb.pdf")


#####
# Calibration Technique: ABC Rejection MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(true_lob_log_returns, get_summary_stats, ones(17), "MAD")
)

@time abc_reject_we = ABCRejection(
    abc_input_we,
    3_000,
    1,
    5.0,
    parallel=true,
    seed=738173189
)
posterior_means = AdaptiveABC.parameter_means(abc_reject_we)
println()
println("ABC rejection MADWE: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

abc_reject_we

param_inds = [1,2,3,4]
plt = plot(abc_reject_we, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_we.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=parameter_names[param_inds])

savefig(plt, "$(dirname(@__FILE__))/abc_reject_we.pdf")

#####
# Calibration Technique: ABC-PMC BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(122477, vcat([0], true_lob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_lob_log_returns), get_summary_stats, boot_weight_matrix)

abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_pmc_out_wb = ABC_PMC(
    abc_input_wb,
    10,
    1,
    0.5,
    10_000,
    3;
    parallel=true,
    batch_size=10_000,
    seed=918731553
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_wb)

println()
println("ABC-PMC BBWM: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

param_inds = [1,2,3,4]
plt = plot(abc_pmc_out_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_we.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=parameter_names[param_inds])