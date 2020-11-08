# Algorithm 4 in (Prangle 2017)

function ABC_PMC(abc_input::ABCInput, n_particles::Int, n_reps::Int, α::Float64, max_sims::Int, max_iter::Int, nsims_for_init=10000; store_init=false, diag_perturb=false, h1=Inf, parallel=false, batch_size=10000)
    prog = Progress(max_iter, 1) ##Progress meter
    if parallel
        println("Running ABCRejection in parallel on $(Threads.nthreads()) threads")
    end
    k = Int(ceil(n_particles*α))
    n_parameters = length(abc_input.prior)
    iter_done = 0
    simulations_done = zeros(Int, max_iter)
    first_iter = true
    thresholds = zeros(Float64, max_iter+1)
    thresholds[1] = h1
    rej_outputs = Array{ABCRejOutput, 1}(undef, max_iter)
    perturb_dist = MvNormal(ones(n_parameters))
    num_iters = max_iter
    for i in 1:max_iter
        sample_from_prior = i==1 || thresholds[i]==Inf
        if sample_from_prior == false
            perturb_dist = get_perturb_dist(rej_outputs[i-1], diag_perturb)
        end
        # Initialise new reference table
        new_parameters = Array{Float64, 2}(undef, n_parameters, n_particles)
        new_summary_stats = Array{Float64, 3}(undef, abc_input.n_summary_stats, n_particles, n_reps)
        new_prior_weights = Array{Float64, 1}(undef, n_particles)
 
        init_summary_stats = Array{Float64, 3}(undef, abc_input.n_summary_stats, n_reps, nsims_for_init)
        init_parameters = Array{Float64, 2}(undef, n_parameters, nsims_for_init)

        if parallel
            parallel_accepts = Atomic{Int64}(0)
            total_sims = Atomic{Int64}(0)
            accepted_particles = 0
            init_particles = 0
            while total_sims[] < max_sims && accepted_particles < n_particles
                
                accepted = fill(-1, batch_size)
                proposal_parameters = Array{Float64, 2}(undef, n_parameters, batch_size)
                prop_summary_stats = Array{Float64, 3}(undef, abc_input.n_summary_stats, n_reps, batch_size)
                prior_weight = Array{Float64, 1}(undef, batch_size)
                Threads.@threads for j in 1:batch_size
                    if parallel_accepts[] >= n_particles || total_sims[] >= max_sims
                        break
                    end
                    # Sample batch_size particles using ABC-PMC

                    accepted[j], proposal_parameters[:, j], prop_summary_stats[:, :, j], prior_weight[j] = abc_pmc_iteration(
                        abc_input, sample_from_prior, rej_outputs[1:i-1], perturb_dist, thresholds[1:i-1], n_reps
                        )

                    atomic_add!(parallel_accepts, accepted[j])
                    atomic_add!(total_sims, 1)
                end
                for j in 1:batch_size
                    if accepted[j] != -1 && init_particles < nsims_for_init 
                        init_particles += 1
                        init_summary_stats[:, :, init_particles] = prop_summary_stats[:, :, j]
                        init_parameters[:, init_particles] = proposal_parameters[:, j]
                    end
                    if accepted[j]==1 && accepted_particles < n_particles
                        accepted_particles += 1
                        new_parameters[:, accepted_particles] = proposal_parameters[:, j]
                        new_summary_stats[:, accepted_particles, :] = prop_summary_stats[:, :, j]
                        new_prior_weights[accepted_particles] = prior_weight[j]
                    end
                end
            end
        else  # Serial Sampling
            parallel_accepts = Atomic{Int64}(0)
            total_sims = Atomic{Int64}(0)
            accepted_particles = 0
            init_particles = 0
            while accepted_particles < n_particles
                
                accepted = zeros(Int, batch_size)
                proposal_parameters = Array{Float64, 2}(undef, n_parameters, batch_size)
                prop_summary_stats = Array{Float64, 2}(undef, abc_input.n_summary_stats, batch_size)
                prior_weight = Array{Float64, 1}(undef, batch_size)

                for j in 1:batch_size
                    if parallel_accepts[] >= n_particles || total_sims[] >= max_sims
                        break
                    end
                    # Sample batch_size particles using ABC-PMC
                    accepted[j], proposal_parameters[:, j], prop_summary_stats[:, :, j], prior_weight[j] = abc_pmc_iteration(
                        abc_input, sample_from_prior, rej_outputs[1:i-1], perturb_dist, thresholds[1:i-1], n_reps
                        )

                    atomic_add!(parallel_accepts, accepted[j])
                    atomic_add!(total_sims, 1)
                end

                for j in 1:batch_size
                    if init_particles < nsims_for_init
                        init_particles += 1
                        init_summary_stats[:, :, init_particles] = prop_summary_stats[:, :, j]
                        init_parameters[:, init_particles] = proposal_parameters[:, j]
                    end
                    if accepted[j]==1 && accepted_particles < n_particles
                        accepted_particles += 1
                        new_parameters[:, accepted_particles] = proposal_parameters[:, j]
                        new_summary_stats[:, accepted_particles, :] = prop_summary_stats[:, :, j]
                        new_prior_weights[accepted_particles] = prior_weight[j]
                    end
                end
            end
        end

        simulations_done[i] = total_sims[]

        if simulations_done[i] < nsims_for_init
            init_summary_stats = init_summary_stats[:, :, 1:simulations_done[i]]
            init_parameters = init_parameters[:, 1:simulations_done[i]]
        end

        if accepted_particles < n_particles
            # Return all 
            println("\nEarly stopping due to max simulations reached in Iteration $(i)")
            println("with $accepted_particles Accepted Particles < $n_particles")
            num_iters = i-1
            break
        end

        if first_iter && thresholds[i]==Inf
            current_distances = zeros(Float64, n_particles)
        else
            current_distances = abc_input.abc_dist(new_summary_stats)
        end
        
        if sample_from_prior
            new_weights = ones(Float64, n_particles)
        else
            new_weights = get_weights(new_parameters, new_prior_weights, rej_outputs[i-1], perturb_dist)
        end


        rej_outputs[i] = ABCRejOutput(
            n_parameters, abc_input.n_summary_stats, simulations_done[i], n_reps, accepted_particles, abc_input.parameter_names, new_parameters,
            new_summary_stats, current_distances, new_weights, copy(abc_input.abc_dist), init_summary_stats, init_parameters 
        )
        new_distances = abc_input.abc_dist(new_summary_stats)
        new_distances = sort(new_distances)[1:k]
        thresholds[i+1] = new_distances[k]
        print("\n\n---------------------- Iteration $i - $(simulations_done[i]) Simulations Done [Total=$(sum(simulations_done))] ----------------------\n\n")
        acceptance_rate = k/simulations_done[i]
        @printf("Acceptance Rate %.2f%%\n", 100*acceptance_rate)
        print(rej_outputs[i])
        println("Next threshold: $(round(thresholds[i+1], digits=4))")
        print("\n----------------------------------------------------------\n")
        first_iter = false
        next!(prog)
    end

    # Add results to ABCPMC Output
    parameters = Array{Float64}(undef, n_parameters, n_particles, num_iters)
    summary_stats = Array{Float64}(undef, abc_input.n_summary_stats, n_particles, n_reps, num_iters)
    distances = Array{Float64}(undef, n_particles, num_iters)
    weights = Array{Float64}(undef, n_particles, num_iters)

    for i in 1:num_iters
        parameters[:, :, i] = rej_outputs[i].parameters
        summary_stats[:, :, :, i] = rej_outputs[i].summary_stats
        distances[:, i] = rej_outputs[i].distances
        weights[:, i] = rej_outputs[i].weights
    end
    if store_init
        init_summary_stats = Array{Array{Float64, 2}}(undef, num_iters)
        init_parameters = Array{Array{Float64, 2}}(undef, num_iters)
        for i in 1:num_iters
            init_summary_stats[i] = rej_outputs[i].init_summary_stats
            init_parameters[i] = rej_outputs[i].init_parameters
        end
    else
        init_summary_stats = Array{Array{Float64, 2}}(undef, 0)
        init_parameters = Array{Array{Float64, 2}}(undef, 0)
    end
    abc_dists = [rej_outputs[i].abc_distance for i in 1:num_iters]
    output = ABCPMCOutput(
        n_parameters, abc_input.n_summary_stats, num_iters, n_reps, simulations_done[1:num_iters],
        abc_input.parameter_names, parameters, summary_stats, distances, weights, abc_dists,
        thresholds[1:num_iters], init_summary_stats, init_parameters
    )
    return output
end

function abc_pmc_iteration(
    abc_input::ABCInput, sample_from_prior::Bool, rej_outputs::Array{ABCRejOutput}, perturb_dist::MvNormal,
    thresholds::Array{Float64, 1}, n_reps::Int
    )
    success = false  # simulated summary stats successfully generated
    while success == false

        if sample_from_prior
            proposal_parameters = rand(abc_input.prior)
        else
            proposal_parameters = importance_sample(rej_outputs[end], perturb_dist)
        end

        prior_weight = pdf(abc_input.prior, proposal_parameters)
        if prior_weight == 0.0
            continue
        end

        success, prop_summary_stats = abc_input.summary_fn(proposal_parameters, abc_input.n_summary_stats, n_reps)
        if !success
            continue
        end
        accept = check_proposal(prop_summary_stats, rej_outputs, thresholds)
        if accept
            return 1, proposal_parameters, prop_summary_stats, prior_weight
        else
            return 0, proposal_parameters, prop_summary_stats, prior_weight
        end
    end
end



function ABC_PMC(abc_out::T, α::Float64, max_sims::Int) where T <: ABCDistance
    return nothing
end


function get_perturb_dist(out::ABCRejOutput, diag::Bool)
    weights = Weights(out.weights)
    if diag
        cov_var = parameter_vars(out)
    else
        cov_var = parameter_covs(out)
    end
    perturb_dist = MvNormal(2.0 .* cov_var)
    return perturb_dist
end


function check_proposal(summary_stats::Array{Float64, 2}, rej_output::ABCRejOutput, threshold::Float64)
    return rej_output.abc_distance(summary_stats) <= threshold 
end


function check_proposal(summary_stats::Array{Float64, 2}, rej_outputs::Array{ABCRejOutput}, thresholds::Array{Float64, 1})
    for i in size(rej_outputs, 1):-1:1
        if thresholds[i] != Inf && !check_proposal(summary_stats, rej_outputs[i], thresholds[i])
            return false
        end
    end
    return true
end


function importance_sample(out::ABCRejOutput, dist::MvNormal)
    i = sample(Weights(out.weights))
    return out.parameters[:,i] + rand(dist)
end


function get_weight(parameters::Array{Float64, 1}, prior_weight::Float64, old_out::ABCRejOutput, perturb_dist::MvNormal)
    n_particles = size(old_out.parameters, 2)
    param_diff_probs = [pdf(perturb_dist, parameters .- old_out.parameters[:, i]) for i in 1:n_particles]
    return prior_weight /sum(old_out.weights .* param_diff_probs)
end


function get_weights(parameters::Array{Float64, 2}, prior_weights::Array{Float64, 1}, old_out::ABCRejOutput, perturb_dist::MvNormal)
    n_particles = size(parameters, 2)
    weights = [get_weight(parameters[:,i], prior_weights[i], old_out, perturb_dist) for i in 1:n_particles]
    return weights ./ sum(weights)
end


