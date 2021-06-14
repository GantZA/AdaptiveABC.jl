mutable struct Prior
    distribution
    Prior(distributionarray::Array{T}) where T <: Distribution = new(distributionarray)
end

function length(prior::Prior)
    return size(prior.distribution, 1)
end

function rand(seed::Int64, d::Prior)
    Random.seed!(seed)
    x = [rand(dist) for dist in d.distribution]
    return x
end

function logpdf(d::Uniform, x::T) where T <: Real
    return log(pdf(d, x))
end

function logpdf(d::Prior, x::AbstractVector{T}) where T <: Real
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return sum([logpdf(dist, i) for (i, dist) in zip(x, d.distribution)])
end

function pdf(d::Prior, x::AbstractVector{T}) where T <: Real
    return exp(logpdf(d, x))    
end

abstract type Input end

mutable struct ABCInput <: Input
    prior::Prior
    parameter_names::Array{String,1}
    obs_summary_stats::Array{Float64,1}
    summary_fn::Function
    n_summary_stats::Int
    abc_dist::T where T <: ABCDistance
end

abstract type ABCOutput end

mutable struct ABCRejOutput <: ABCOutput
    n_parameters::Int
    n_summary_stats::Int
    n_sims::Int
    n_reps::Int
    n_successes::Int
    parameter_names::Array{String,1}
    parameters::Array{Float64,2}  # parameter[i, j,] is the ith parameter in the jth accepted simulation
    summary_stats::Array{Float64,3}  # summary_stats[i, j] is the ith summary statistic in the jth accepted simulation in the kth replication
    distances::Array{Float64,1}  # distances[i] is the distance in the ith accepted simulation 
    weights::Array{Float64,1}  # weights[i] is the weight for the ith accepted simulation
    abc_distance::T where T <: ABCDistance
    init_summary_stats::Array{Float64,3}  ##sims used for distance initialisation (only stored optionally)
    init_parameters::Array{Float64,2}  ##pars used for distance initialisation (only stored optionally)
end

function show(io::IO, out::ABCRejOutput)
    posterior_means = parameter_means(out)
    # posterior_vars = parameter_vars(out)
    # posterior_covs = parameter_cov(out)

    cred_int_lower = Array{Float64}(undef, out.n_parameters)
    cred_int_upper = Array{Float64}(undef, out.n_parameters)
    for j in 1:out.n_parameters
        cred_int_lower[j], cred_int_upper[j] = quantile(out.parameters[j,:], Weights(out.weights), [0.025,0.975]) 
    end
    ess = sum(out.weights)^2 / sum(out.weights.^2)
    println("Total number of simulations: $(out.n_sims)")
    println("Total number of acceptances: $(out.n_successes)")
    println("Acceptance percentage = $(round(out.n_successes / out.n_sims * 100.0, digits=2))% \n\n")
    println("Effective Sample Size = $(round(ess, digits=1))")
    print("\nParameters:\n\n")
    print("\tMean (95% intervals):\n")
    for j in 1:out.n_parameters
        @printf("\tParameter %s: %.2f (%.2f,%.2f)\n", out.parameter_names[j], posterior_means[j], cred_int_lower[j], cred_int_upper[j])
    end
end

function sort_ABC_output!(out::ABCRejOutput)
    sorted_order = sortperm(out.distances)
    out.parameters = out.parameters[:, sorted_order]
    out.summary_stats = out.summary_stats[:,sorted_order, :]
    out.distances = out.distances[sorted_order]
    out.weights = out.weights[sorted_order]
    return 
end

function copy(out::ABCRejOutput)
    return ABCRejOutput(
        out.n_parameters, out.n_summary_stats, out.n_reps, out.n_sims, out.n_successes, out.parameter_names, 
        out.parameters, out.summary_stats, out.distances, out.weights, out.abc_distance, out.init_summary_stats, out.init_parameters
    )
end


mutable struct ABCPMCOutput <: ABCOutput
    n_parameters::Int
    n_summary_stats::Int
    n_iterations::Int
    n_reps::Int
    n_tot_sims::Array{Int,1}
    parameter_names::Array{String,1}
    parameters::Array{Float64,3}  # parameter[i, j, k] is the ith parameter in the jth accepted simulation in the kth iteration
    summary_stats::Array{Float64,4}  # summary_stats[i, j, k] is the ith summary statistic in the jth accepted simulation in the kth replication in the lth iteration
    distances::Array{Float64,2}  # distances[i, j] is the distance in the ith accepted simulation in the jth iteration
    weights::Array{Float64,2}  # weights[i, j] is the weight for the ith accepted simulation in the jth iteration
    abc_distances::Array{ABCDistance,1}  # ABCDistance used in the ith iteration
    thresholds::Array{Float64,1}  # thresholds[i] is the acceptance threshold used in the ith iteration
    init_summary_stats::Array{Array{Float64,2},1} ##init_sims[i] is sims for distance initialisation at iteration i (only stored optionally)
    init_parameters::Array{Array{Float64,2},1} ##init_pars[i] is pars for distance initialisation at iteration i (only stored optionally)
end

function show(io::IO, out::ABCPMCOutput)
    posterior_means = parameter_means(out)
    # posterior_vars = parameter_vars(out)
    # posterior_covs = parameter_cov(out)
    println("Total number of simulations: $(sum(out.n_tot_sims))")
    println("Cumulative number of simulations = $(cumsum(out.n_tot_sims))")
    for i in 1:out.n_iterations
        print("\n\n---------------------- Iteration $i -----------------------\n\n")
        cred_int_lower = Array{Float64}(undef, out.n_parameters)
        cred_int_upper = Array{Float64}(undef, out.n_parameters)
        for j in 1:out.n_parameters
            cred_int_lower[j], cred_int_upper[j] = quantile(out.parameters[j,:, i], Weights(out.weights[:, i]), [0.025,0.975])
        end
        ess = sum(out.weights[:,i])^2 / sum(out.weights[:, i].^2)

        println("Acceptance ratio = $(round(size(out.distances[:, i], 1) / out.n_tot_sims[i], digits=2))")
        println("Tolerance schedule = $(round.(out.thresholds[i], digits=2))")
        println("Effective Sample Size = $(round(ess, digits=1))")
        print("\nParameters:\n")
        print("\tMean \t (95% intervals):\n")
        for j in 1:out.n_parameters
            @printf("\tParameter %s: %.2f (%.2f,%.2f)\n", out.parameter_names[j], posterior_means[j, i], cred_int_lower[j], cred_int_upper[j])
        end
        print("\n----------------------------------------------------------\n")

    end
    
end


function parameter_means(out::ABCRejOutput)
    posterior_means = mean(out.parameters, Weights(out.weights), dims=2)  # Average of the parameter across all accepted sims
    return posterior_means
end



function parameter_means(out::ABCPMCOutput)
    posterior_means = Array{Float64}(undef, out.n_parameters, out.n_iterations)
    for i in 1:out.n_iterations
        posterior_means[:, i] = mean(out.parameters[:, :, i], Weights(out.weights[:, i]), dims=2)  # Average of the parameters across all accepted sims in the ith iter
    end
    return posterior_means
end


function parameter_vars(out::ABCRejOutput)
    posterior_vars = var(out.parameters, Weights(out.weights), dims=2, corrected=false)  # Variance of the ith parameter across all accepted sims
    return posterior_vars
end


# function parameter_vars(out::ABCPMCOutput)
#     posterior_vars = Array{Float64}(undef, out.n_parameters, out.n_iterations)
#     for i in 1:out.n_iterations
#         posterior_vars[:, i] = var(out.parameters[:, :, i], Weights(out.weights[:, i]), dims=2, corrected=false)  # Variance of the ith parameter across all accepted sims
#     end
#     return posterior_vars
# end


function parameter_covs(out::ABCRejOutput)
    posterior_covs = cov(out.parameters, Weights(out.weights), 2, corrected=false)  # Variance of the ith parameter across all accepted sims
    return posterior_covs
end


# function parameter_covs(out::ABCPMCOutput)
#     posterior_covs = Array{Float64}(undef, out.n_parameters, out.n_iterations)
#     for i in 1:out.n_iterations
#         posterior_covs[:, i] = cov(out.parameters[:, :, i], Weights(out.weights[:, i]), dims=2, corrected=false)  # Variance of the ith parameter across all accepted sims
#     end
#     return posterior_covs
# end



