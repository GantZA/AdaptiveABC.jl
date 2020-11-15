normal_kurtosis(x) = kurtosis(x, false)

function ks_test(a,b)
    ecdf_a = ecdf(a[:, 1])
    ecdf_b = ecdf(b[:, 1])
    all_x = vcat(a[:,1],b[:,1])
    return maximum(abs.(ecdf_a(all_x) .- ecdf_b(all_x)))
end

acf(x, lags) = autocor(x, lags)

function generalized_hurts_exp(obs)
    q = 1
    max_τ = 19

    k = 0
    H = zeros(15)
    L = size(obs, 1)

    for iter_τ = 5:max_τ
        k += 1

        x = collect(1:iter_τ)
        k_q_t = zeros(iter_τ)
        for τ = 1:iter_τ
            numer = obs[(τ+1):τ:L] - obs[1:τ:(L-τ)]
            denom = obs[1:τ:L]

            # Determine Drift
            N = size(numer, 1) + 1
            X = collect(1:N)
            Y = denom
            mx = sum(X)/N
            SSxx = sum(X.^2) - N * mx^2
            my = sum(Y) / N
            SSxy = sum(X .* Y) - N * mx * my
            cc = [SSxy/SSxx, my - SSxy/SSxx*mx]

            # Subtract Drift
            numer = numer .- cc[1]
            denom = denom .- cc[1] * collect(1:N) .- cc[2]

            k_q_t[τ] = mean(abs.(numer).^q) / mean(abs.(denom).^q)
        end

        # Calculate Hurst Exponent for current iteration
        log_10_x = log10.(x)
        mx = mean(log_10_x)
        SSxx = sum(log_10_x.^2) - iter_τ * mx^2
        log_10_k_q_t = log10.(k_q_t)
        my = mean(log_10_k_q_t)
        SSxy = sum(log_10_x .* log_10_k_q_t) - iter_τ * mx * my
        H[k] = SSxy/SSxx
    end

    H = mean(H) / q
    return H
end

function get_summary_stats(obs::Array{Float64, 1}, simulated_obs::Array{Float64, 1})
    summary_stats = zeros(17)

    summary_stats[1] = mean(simulated_obs)
    summary_stats[2] = std(simulated_obs)
    summary_stats[3] = normal_kurtosis(simulated_obs)
    summary_stats[4] = ks_test(obs, simulated_obs)
    summary_stats[5] = generalized_hurts_exp(simulated_obs)

    acf_raw = acf(simulated_obs, [1, 5])
    acf_sqr = acf(simulated_obs.^2, [1, 5])
    acf_abs = acf(abs.(simulated_obs), [1, 5])

    pacf_raw = pacf(simulated_obs, [1, 5])
    pacf_sqr = pacf(simulated_obs.^2, [1, 5])
    pacf_abs = pacf(abs.(simulated_obs), [1, 5])
    cov_stats = [acf_raw..., acf_sqr..., acf_abs..., pacf_raw..., pacf_sqr..., pacf_abs...]
    for i in 6:17
        summary_stats[i] = cov_stats[i-5]
    end
    return summary_stats
end

function get_summary_stats(obs::Array{Float64, 1})
    summary_stats = zeros(17)

    summary_stats[1] = mean(obs)
    summary_stats[2] = std(obs)
    summary_stats[3] = normal_kurtosis(obs)
    summary_stats[4] = 0.0
    summary_stats[5] = generalized_hurts_exp(obs)

    acf_raw = acf(obs, [1, 5])
    acf_sqr = acf(obs.^2, [1, 5])
    acf_abs = acf(abs.(obs), [1, 5])

    pacf_raw = pacf(obs, [1, 5])
    pacf_sqr = pacf(obs.^2, [1, 5])
    pacf_abs = pacf(abs.(obs), [1, 5])
    cov_stats = [acf_raw..., acf_sqr..., acf_abs..., pacf_raw..., pacf_sqr..., pacf_abs...]
    for i in 6:17
        summary_stats[i] = cov_stats[i-5]
    end
    return summary_stats
end

