abstract type ABCDistance end

mutable struct WeightedEuclidean <: ABCDistance
    obs_sum_stats::Array{Float64, 1}
    weights::Array{Float64, 1}
    scale_type::AbstractString
end

function WeightedEuclidean()
    return WeightedEuclidean(Array{Float64, 1}(undef, 0), Array{Float64, 1}(undef, 0), "MAD")
end

function WeightedEuclidean(obs_sum_stats::Array{Float64, 1}, scale_type::AbstractString, summary_stats::Array{Float64, 2})
    n_stats, n_sim = size(summary_stats)
    if scale_type=="MAD"
        inv_weights = [MAD(summary_stats[i, :]) for i in 1:n_stats]  # For each statistic, calculate the MAD across all n_sim values
    end
    return WeightedEuclidean(obs_sum_stats, 1.0./inv_weights, scale_type)
end

function WeightedEuclidean(obs_sum_stats::Array{Float64, 1})
    WeightedEuclidean(obs_sum_stats, Array{Float64}(0), "MAD")
end

function WeightedEuclidean(obs_sum_stats::Array{Float64, 1}, summary_stats::Array{Float64, 2})
    return WeightedEuclidean(obs_sum_stats, "MAD", summary_stats)
end

function WeightedEuclidean(obs_sum_stats::Array{Float64, 1}, summary_stats::Array{Float64, 2}, parameters::Array{Float64, 2})
    n_summary_stats, n_particles = size(summary_stats)
    if n_particles == 0
        sigma = ones(n_summary_stats)
    else
        return WeightedEuclidean(obs_sum_stats, "MAD", summary_stats)
    end
    return WeightedEuclidean(obs_sum_stats, 1.0./sigma, "MAD")
end

function (we::WeightedEuclidean)(sim_sum_stats::Array{Float64, 1})
    abs_diff = abs.(we.obs_sum_stats - sim_sum_stats)
    weighted_diff = abs_diff .* we.weights
    weighted_diff[abs_diff .== 0.0] .= 0.0
    return norm(weighted_diff, 2)
end

# Median Absolute Deviation
function MAD(x::Array{Float64, 1})
    return median(abs.(x .- median(x)))
end
