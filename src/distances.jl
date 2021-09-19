abstract type ABCDistance end

mutable struct WeightedEuclidean <: ABCDistance
    obs_returns::Array{Float64,1}
    get_summary_stats::Function
    weights::Array{Float64,1} 
    scale_type::AbstractString
end

function copy(we::WeightedEuclidean)
    return WeightedEuclidean(we.obs_returns, we.get_summary_stats, we.weights, we.scale_type)
end
# function WeightedEuclidean()
#     return WeightedEuclidean(Array{Float64, 1}(undef, 0), identity, Array{Float64, 1}(undef, 0), "MAD")
# end

# function WeightedEuclidean(obs_returns::Array{Float64, 1})
#     WeightedEuclidean(obs_returns, identity, Array{Float64}(0), "MAD")
# end

# function WeightedEuclidean(obs_returns::Array{Float64, 1}, summary_stats::Array{Float64, 2})
#     return WeightedEuclidean(obs_returns, identity, "MAD", summary_stats)
# end

# function WeightedEuclidean(obs_returns::Array{Float64, 1}, get_summary_stats::Function, summary_stats::Array{Float64, 2})
#     n_summary_stats, n_particles = size(summary_stats)
#     if n_particles == 0
#         sigma = ones(n_summary_stats)
#         return WeightedEuclidean(obs_returns, get_summary_stats, sigma, "MAD")
#     else
#         inv_weights = [MAD(summary_stats[i, :]) for i in 1:n_summary_stats]  # For each statistic, calculate the MAD across all n_sim values        
#         return WeightedEuclidean(obs_returns, get_summary_stats, 1.0./inv_weights, scale_type)
#     end
# end

function (we::WeightedEuclidean)(
        init_sum_stats::Array{Float64,3},
        sim_sum_stats::Array{Float64,3}
    )::Array{Float64,1}
    n_summary_stats, n_particles, n_replications = size(sim_sum_stats)

    # Calculate Weighted Differences
    obs_sum_stats_vec = we.get_summary_stats(we.obs_returns)
    obs_sum_stats = repeat(obs_sum_stats_vec, outer=[1, n_particles, n_replications])
    weighted_diff = repeat(we.weights, outer=[1, n_particles, n_replications]) .* abs.(obs_sum_stats - sim_sum_stats)  # Array of Weighted Differences (stats x sims x reps) 
    vec_norm_weighted_diffs = mapslices(x -> norm(x, 2), weighted_diff, dims=[1, 3])[1,:,1]
    
    # Update/Calculate weights using init_sum_stats
    if n_particles == 0
        weights = ones(n_summary_stats)
    else
        inv_weights = [MAD(init_sum_stats[i, :, :]) for i in 1:n_summary_stats]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
        weights = 1.0 ./ inv_weights
    end
    we.weights = weights
    return vec_norm_weighted_diffs # Vec of Normalised Weighted Differences (sims)
end


function (we::WeightedEuclidean)(
        sim_sum_stats::Array{Float64,3}
    )::Array{Float64,1}
    n_summary_stats, n_particles, n_replications = size(sim_sum_stats)

    # Calculate weights using sim_sum_stats
    if n_particles == 0
        weights = ones(n_summary_stats)
    else
        inv_weights = [MAD(sim_sum_stats[i, :, :]) for i in 1:n_summary_stats]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
        weights = 1.0 ./ inv_weights
    end
    we.weights = weights

    # Calculate Weighted Differences
    obs_sum_stats_vec = we.get_summary_stats(we.obs_returns)
    obs_sum_stats = repeat(obs_sum_stats_vec, outer=[1, n_particles, n_replications])
    weighted_diff = repeat(we.weights, outer=[1, n_particles, n_replications]) .* abs.(obs_sum_stats - sim_sum_stats)  # Array of Weighted Differences (stats x sims x reps) 
    vec_norm_weighted_diffs = mapslices(x -> norm(x, 2), weighted_diff, dims=[1, 3])[1,:,1]
    return vec_norm_weighted_diffs # Vec of Normalised Weighted Differences (sims)
end


function (we::WeightedEuclidean)(sim_sum_stats::Array{Float64,1})
    # Calculate Weighted Differences
    obs_sum_stats = we.get_summary_stats(we.obs_returns)
    weighted_diff = we.weights .* abs.(obs_sum_stats .- sim_sum_stats)  # Vector of Weighted Differences (stats x 1) 
    return norm(weighted_diff, 2) # Scalar of Normalised Weighted Differences (1 x 1)
end

function (we::WeightedEuclidean)(sim_sum_stats::Array{Float64,2})
    # Given 1 particle of n_rep and n_sum_stats, output the distance
    n_summary_stats, n_replications = size(sim_sum_stats)
    obs_sum_stats = repeat(we.get_summary_stats(we.obs_returns), outer=[1, n_replications])
    weighted_diff = we.weights .* abs.(obs_sum_stats .- sim_sum_stats)  # 2D Array of Weighted Differences (stats x 1)
    return norm(weighted_diff, 2) # Scalar of Normalised Weighted Differences (1 x 1)
end


# Median Absolute Deviation
function MAD(x::Array{Float64,2})::Float64
    # Calculate Median Absolute Deviation of Array (n_particles x n_replications)
    median_per_particle = median(x, dims=2) # Vec of size n_particles
    x_median = median(median_per_particle)
    return median(abs.(x .- x_median))
end

# Median Absolute Deviation
function MAD(x::Array{Float64,1})::Float64
    # Calculate Median Absolute Deviation of Array (n_particles x 1)
    x_median = median(x)
    return median(abs.(x .- x_median))
end


# Winker Objective Function

mutable struct WeightedBootstrap <: ABCDistance
    obs_returns::Array{Float64,1}
    get_summary_stats::Function 
    bootstrap_weight_matrix::BlockBootstrapWeightMatrix
end

function copy(wb::WeightedBootstrap)
    return WeightedBootstrap(wb.obs_returns, wb.get_summary_stats, wb.bootstrap_weight_matrix)
end


function (wb::WeightedBootstrap)(
        init_sum_stats::Array{Float64,3},
        sim_sum_stats::Array{Float64,3}
    )::Array{Float64,1}
    n_summary_stats, n_particles, n_replications = size(sim_sum_stats)
    # Calculate Weighted Differences
    obs_sum_stats_vec = wb.get_summary_stats(wb.obs_returns)
    obs_sum_stats = repeat(obs_sum_stats_vec, outer=[1, n_particles, n_replications])

    summary_diffs = mean(obs_sum_stats .- sim_sum_stats, dims=3)[:, :, 1]
    weighted_diffs = mapslices(x -> x' * wb.bootstrap_weight_matrix.weight_matrix * x, summary_diffs, dims=[1])
    return weighted_diffs[1, :]
end

function (wb::WeightedBootstrap)(
    sim_sum_stats::Array{Float64,3}
    )::Array{Float64,1}

    return wb(zeros(1, 1, 1), sim_sum_stats)
end

function (wb::WeightedBootstrap)(sim_sum_stats::Array{Float64,1})
    # Calculate Weighted Differences
    obs_sum_stats = wb.get_summary_stats(wb.obs_returns)
    summary_diffs = obs_sum_stats .- sim_sum_stats
    weighted_diff = summary_diffs' * wb.bootstrap_weight_matrix.weight_matrix * summary_diffs
    return weighted_diff
end

function (wb::WeightedBootstrap)(sim_sum_stats::Array{Float64,2})
    # Given 1 particle of n_rep and n_sum_stats, output the distance
    n_summary_stats, n_replications = size(sim_sum_stats)
    obs_sum_stats = repeat(wb.get_summary_stats(wb.obs_returns), outer=[1, n_replications])
    summary_diffs = mean(obs_sum_stats .- sim_sum_stats, dims=2)[:, 1]
    weighted_diff = summary_diffs' * wb.bootstrap_weight_matrix.weight_matrix * summary_diffs
    return weighted_diff
end

# Maximum Mean Discrepancy

# mutable struct MaximumMeanDiscrepancy <: ABCDistance
#     obs_returns::Array{Float64, 1}
#     get_summary_stats::Function
# end

# function copy(MMD::MaximumMeanDiscrepancy)
#     return MaximumMeanDiscrepancy(MMD.obs_returns, MMD.get_summary_stats)
# end


# function (MMD::MaximumMeanDiscrepancy)(sim_sum_stats::Array{Float64, 3})::Array{Float64, 1}
#     n_summary_stats, n_particles, n_replications = size(sim_sum_stats)
#     # Calculate Weighted Differences
#     obs_array = repeat(obs, outer=[1, n_particles, n_replications])
#     max_mean_discrepancies = mmd(GaussianKernel(1.0), reshape(obs_array, (n_particles,:)), reshape(simulation(ARMA(0.5, 0.5, 0.09), ts_length=n), (1,:)))
#     return weighted_diffs[1, :]
# end

# function (MMD::MaximumMeanDiscrepancy)(sim_sum_stats::Array{Float64, 1})
#     # Calculate Weighted Differences
#     obs_sum_stats = MMD.get_summary_stats(MMD.obs_returns)
#     summary_diffs = obs_sum_stats .- sim_sum_stats

# end
# mmd(GaussianKernel(1.0), reshape(obs, (1,:)), reshape(simulation(ARMA(0.5, 0.5, 0.09), ts_length=n), (1,:)))
# function (MMD::MaximumMeanDiscrepancy)(sim_sum_stats::Array{Float64, 2})
#     # Given 1 particle of n_rep and n_sum_stats, output the distance
#     n_summary_stats, n_replications = size(sim_sum_stats)
#     obs_sum_stats = repeat(MMD.get_summary_stats(MMD.obs_returns), outer=[1, n_replications])
#     summary_diffs = mean(obs_sum_stats .- sim_sum_stats, dims=2)[:, 1]

# end

# function 