using Base.Threads
using SequentialLOB
using Plots
pyplot()
using PyPlot
pygui(true)

using AdaptiveABC
using QuantEcon
using Random
using Distributions
using ProgressMeter



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
n_summary_stats = size(obs_summary_stats, 1)

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

weighted_mad = WeightedEuclidean(obs, get_summary_stats, ones(n_summary_stats), "MAD")
sim_sum_stats = zeros(n_summary_stats, 10000, 1)


prog = Progress(10000, 1)

seeds = Int.(rand(MersenneTwister(651411613), UInt32, 10000))
for i in 1:10000
    params = rand(seeds[i], prior)
    success, sum_stats = summary_fn(params, n_summary_stats, 1)
    sim_sum_stats[:,i,:] = sum_stats
    next!(prog)
end

inv_weights = [AdaptiveABC.MAD(sim_sum_stats[i, :, :]) for i in 1:n_summary_stats]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
weights = 1.0 ./ inv_weights

weighted_mad.weights = weights


function f(x, weight, reps)
    success, sum_stats = summary_fn(x, n_summary_stats, reps)
    return weight(sum_stats)
end

f_mad(x, grad) = f(x, weighted_mad, 1000)

# phi (x) vs theta (y)

num_points = 20
x_s = repeat(collect(range(0.6, stop=0.8, length=num_points)), inner=num_points)
y_s = repeat(collect(range(0.2, stop=0.4, length=num_points)), outer=num_points)

z_s = zeros(num_points^2)

prog = Progress(num_points^2, 1)

Threads.@threads for i in 1:num_points^2
    z_s[i] = f_mad([x_s[i], y_s[i], 0.01], [0])
    next!(prog)
end

param_true_vals = [ϕ, θ]

z_true = 0
for i in 1:size(x_s, 1)
    if x_s[i] >= param_true_vals[1] && y_s[i] >= param_true_vals[2]
        z_true = z_s[i]
        break
    end
end
z_range = maximum(z_s) - minimum(z_s)
true_plot_z = 0.01 * z_range

plt = Plots.plot(
    x_s, y_s, z_s, st=:surface,
    size=(1500, 800),
        xlabel="Phi (ϕ)" , ylabel="Theta (θ)",
        zlabel="Distance Value", camera=(-39, 15),
        xguidefontsize=14,yguidefontsize=14,zguidefontsize=14,
        xtickfontsize=8,ytickfontsize=8,ztickfontsize=8,
        dpi=200, zguidefontrotation=90, formatter=:plain, legend=:none
)


Plots.plot!(
    plt, 
    [param_true_vals[1],param_true_vals[1]], 
    [param_true_vals[2],param_true_vals[2]], 
    [z_true - true_plot_z,z_true + true_plot_z],
    label="True Parameter Values", 
    camera=(-39, 15),
    st=:line, c=:red, linewidth=2, formatter=:plain,);


Plots.savefig(plt, joinpath(@__DIR__, "phi_theta_surface.pdf"))
