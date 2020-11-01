abstract type ABCDistance end

mutable struct WeightedEuclidean <: ABCDistance
    obs_returns::Array{Float64, 1}
    get_summary_stats::Function
    weights::Array{Float64, 1} 
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

function (we::WeightedEuclidean)(sim_sum_stats::Array{Float64, 2})
    n_summary_stats, n_particles = size(sim_sum_stats)

    # Calculate weights
    if n_particles == 0
        weights = ones(n_summary_stats)
    else
        inv_weights = [MAD(sim_sum_stats[i, :]) for i in 1:n_summary_stats]  # For each statistic, calculate the MAD across all n_sim values        
        weights = 1.0./inv_weights
    end
    we.weights = weights

    # Calculate Weighted Differences
    obs_sum_stats_vec = we.get_summary_stats(we.obs_returns)
    obs_sum_stats = repeat(obs_sum_stats_vec, outer=[1, n_particles])
    weighted_diff = repeat(weights, outer=[1, n_particles]) .* abs.(obs_sum_stats - sim_sum_stats)  # Array of Weighted Differences (stats x sims) 
    return mapslices(x -> norm(x, 2), weighted_diff, dims=1) # Vector of Normalised Weighted Differences (1 x sims)
end


function (we::WeightedEuclidean)(sim_sum_stats::Array{Float64, 1})
    # Calculate Weighted Differences
    obs_sum_stats = we.get_summary_stats(we.obs_returns)
    weighted_diff = we.weights .* abs.(obs_sum_stats .- sim_sum_stats)  # Vector of Weighted Differences (stats x 1) 
    return norm(weighted_diff, 2) # Scalar of Normalised Weighted Differences (1 x 1)
end

# Median Absolute Deviation
function MAD(x::Array{Float64, 1})
    x_median = median(x)
    return median(abs.(x .- x_median))
end


# Winker Objective Function

mutable struct WeightedBootstrap <: ABCDistance
    obs_returns::Array{Float64, 1}
    get_summary_stats::Function 
    bootstrap_weight_matrix::BlockBootstrapWeightMatrix
end

function copy(wb::WeightedBootstrap)
    return WeightedBootstrap(wb.obs_returns, wb.get_summary_stats, wb.bootstrap_weight_matrix)
end


function (wb::WeightedBootstrap)(sim_sum_stats::Array{Float64, 2})
    n_summary_stats, n_particles = size(sim_sum_stats)
    # Calculate Weighted Differences
    obs_sum_stats_vec = wb.get_summary_stats(wb.obs_returns)
    obs_sum_stats = repeat(obs_sum_stats_vec, outer=[1, n_particles])

    summary_diffs = obs_sum_stats .- sim_sum_stats
    weighted_diffs = mapslices(x -> x' * wb.bootstrap_weight_matrix.weight_matrix * x, summary_diffs, dims=1)
    return weighted_diffs
end

function (wb::WeightedBootstrap)(sim_sum_stats::Array{Float64, 1})

    # Calculate Weighted Differences
    obs_sum_stats = wb.get_summary_stats(wb.obs_returns)
    summary_diffs = obs_sum_stats .- sim_sum_stats
    weighted_diff = summary_diffs' * wb.bootstrap_weight_matrix.weight_matrix * summary_diffs
    return weighted_diff
end