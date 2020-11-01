abstract type WeightMatrix end

mutable struct BlockBootstrapWeightMatrix <: WeightMatrix
    seed::Int64
    get_summary_stats::Function
    block_size::Int64
    n_bootstrap::Int64
    weight_matrix::Array{Float64, 2}
end
function BlockBootstrapWeightMatrix(seed::Int64, log_returns::Array{Float64, 1}, get_summary_stats::Function,
    block_size::Int64, n_bootstrap::Int64)
    # Step 1: Apply a Moving Block Bootstrap to the Log Return Path and Calculate Summary Statistics
    Random.seed!(seed)
    n = size(log_returns, 1)
    block_ind = 1:n-block_size+1

    obs_summary_stats = get_summary_stats(log_returns)
    b_summary_stats = Array{Float64, 2}(undef, size(obs_summary_stats, 1), n_bootstrap)
    b_samples = Array{Float64,2}(undef, n, n_bootstrap)
    
    for i in 1:n_bootstrap
        b_samples[:,i] = log_returns[block_bootstrap_index(block_ind, n, block_size)]
        b_summary_stats[:, i] = get_summary_stats(log_returns, b_samples[:,i])
    end

    # Step 3: Calculate Inverse Weight Matrix
    W = inv(cov(b_summary_stats, dims=2))
    return BlockBootstrapWeightMatrix(seed, get_summary_stats, block_size, n_bootstrap, W)
end


function block_bootstrap_index(block_ind, n, b)
    rand_blocks = sample(block_ind, floor(Int,n/b))
    sample_ind = transpose(repeat(rand_blocks, 1, b))
    sample_ind = sample_ind[:]
    addition_vec = repeat(0:b-1,floor(Int,n/b))
    sample_ind = sample_ind + addition_vec
    return sample_ind
end

