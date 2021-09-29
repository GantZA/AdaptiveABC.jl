function ABCRejection(abc_input::ABCInput, n_sims::Int, n_reps::Int; store_init=false, 
    parallel=false, seed=-1)
    # Set Seed
    if seed == -1
        seeds = Int.(rand(MersenneTwister(), UInt32, n_sims))
    else
        seeds = Int.(rand(MersenneTwister(seed), UInt32, n_sims))
    end

    prog = Progress(n_sims, 1)
    n_params = size(abc_input.prior.distribution, 1)
    parameters = Array{Float64}(undef, n_params, n_sims)
    summary_stats = Array{Float64}(undef, abc_input.n_summary_stats, n_sims, n_reps)
    successes = Array{Bool}(undef, n_sims)
    if parallel
        rnglock = ReentrantLock()

        println("Running ABCRejection in parallel on $(Threads.nthreads()) threads")
        Threads.@threads for i in 1:n_sims
            lock(rnglock)
            parameters[:, i] = rand(seeds[i], abc_input.prior)
            unlock(rnglock)
            successes[i], summary_stats[:, i, :] =  abc_input.summary_fn(parameters[:, i], abc_input.n_summary_stats, n_reps)
            next!(prog)
        end
    else
        for i in 1:n_sims
            parameters[:, i] = rand(seeds[i], abc_input.prior)
            successes[i], summary_stats[:, i, :] =  abc_input.summary_fn(parameters[:, i], abc_input.n_summary_stats, n_reps)
            next!(prog)
        end
    end

    n_successes = sum(successes)
    if n_successes == 0
        error("No successes for simulator!")
        return nothing
    end
    parameters = parameters[:, successes]
    summary_stats = summary_stats[:, successes, :]
    distances = abc_input.abc_dist(summary_stats)

    if store_init
        init_sims = summary_stats
        init_params = parameters
    else
        init_sims = Array{Float64}(undef, 0, 0, 0)
        init_params = Array{Float64}(undef, 0, 0)
    end

    out = ABCRejOutput(n_params, abc_input.n_summary_stats, n_sims, n_reps, n_successes, abc_input.parameter_names, parameters, summary_stats, distances, ones(n_sims), abc_input.abc_dist, init_sims, init_params)
    sort_ABC_output!(out)
    return out
end

function ABCRejection(abc_input::ABCInput, n_sims::Int, n_reps::Int, k::Int; 
    store_init=false, parallel=false, seed=-1)
    out = ABCRejection(abc_input, n_sims, n_reps, store_init=store_init, parallel=parallel, seed=seed)
    out.parameters = out.parameters[:, 1:k]
    out.summary_stats = out.summary_stats[:, 1:k, :]
    out.distances = out.distances[1:k]
    out.weights = out.weights[1:k]
    return out
end


function ABCRejection(abc_input::ABCInput, n_sims::Int, n_reps::Int, h::AbstractFloat; 
    store_init=false, parallel=false, seed=-1)
    out = ABCRejection(abc_input, n_sims, n_reps, store_init=store_init, parallel=parallel, seed=seed)
    if out.distances[end] <= h
        k = out.n_successes + 1
    else
        k = findfirst(x -> x > h, out.distances)
    end
    if isnothing(k)
        println("No simulations were accepted at current threshold=$h. Smallest distance is $(out.distances[1])")
        return out
    else
        k = k - 1
        out.parameters = out.parameters[:, 1:k]
        out.summary_stats = out.summary_stats[:, 1:k, :]
        out.distances = out.distances[1:k]
        out.weights = out.weights[1:k]
        out.n_successes = k
        return out
    end
end