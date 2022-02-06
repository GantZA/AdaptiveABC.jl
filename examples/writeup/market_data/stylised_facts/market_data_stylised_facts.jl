using SequentialLOB
using StylizedFacts
using Plots
using DelimitedFiles
using AdaptiveABC
using Distributions

# Read in Observed Market Data

observed_price_path = round.(readdlm("$(dirname(@__FILE__))/Original_Price_Bars_2300.csv", ',', Float64, '\n')[:,1], digits=2)
observed_log_returns = diff(log.(observed_price_path[:,1]))
observed_summary_stats = get_summary_stats(observed_log_returns)

observed_price_path

plt = plot(
    1:size(observed_log_returns, 1) + 1, observed_price_path, legend=false,
    xlab="Time", ylab="Mid-Price",
)

display(plt)
savefig(plt, "$(dirname(@__FILE__))/market_data_price_path.pdf")

market_data_stylized_facts = StylizedFactsPlot(observed_price_path)

# Plot Log Returns Hist

dist = fit(Normal, observed_log_returns)
plt = histogram(observed_log_returns, normalize=true, legend=false)
plot!(plt, x -> pdf(dist, x), xlim=xlims(), legend=false)
savefig(plt, "$(dirname(@__FILE__))/market_data_hist_log_returns.pdf")



# Plot Log Returns

plt = StylizedFacts.plot_log_returns(market_data_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/market_data_returns.pdf")

# Plot QQ Log Returns

plt = StylizedFacts.plot_qq_log_returns(market_data_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/market_data_qq_log_returns.pdf")

# Plot ACF Order Flow

plt = StylizedFacts.plot_acf_order_flow(market_data_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/market_data_acf_order_flow.pdf")

# Plot ACF Log Returns

plt = StylizedFacts.plot_acf_log_returns(market_data_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/market_data_acf_log_returns.pdf")

# Plot ACF Abs Log Returns

plt = StylizedFacts.plot_acf_abs_log_returns(market_data_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/market_data_acf_abs_log_returns.pdf")


plt = StylizedFacts.plot_all_stylized_facts(market_data_stylized_facts)
display(plt)
savefig(plt, "$(dirname(@__FILE__))/market_data_stylised_facts.pdf")

# Plot Stylised Facts from Calibrated Models

# Constants
num_paths = 1; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200 ; λ = 1.0

# Calibrated Parameters LOB Model

# ABC-PMC BBWM: D, σ, nu, μ
seed = 731986389128937
D, σ, nu, μ = [1.79, 0.401, 0.533, 1.518]
α = 0.0
model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ))
calibrated_price_path = model(seed)[:,1]

calibrated_stylized_facts = StylizedFactsPlot(calibrated_price_path)

# Plot Price num_paths
plt = plot(1:size(calibrated_stylized_facts.price_path, 1), calibrated_stylized_facts.price_path, legend=false,
    xlab="Time", ylab="Mid-Price")
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_price_path.pdf")


# Plot Log Returns Hist

dist = fit(Normal, calibrated_stylized_facts.log_returns)
plt = histogram(calibrated_stylized_facts.log_returns, normalize=true, legend=false)
plot!(plt, x -> pdf(dist, x), xlim=xlims(), legend=false)
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_hist_log_returns.pdf")



# Plot Log Returns
plt = StylizedFacts.plot_log_returns(calibrated_stylized_facts, "")
display(plt)
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_log_returns.pdf")

# Plot QQ Log Returns

plt = StylizedFacts.plot_qq_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_qq_log_returns.pdf")

# Plot ACF Order Flow

plt = StylizedFacts.plot_acf_order_flow(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_acf_order_flow.pdf")

# Plot ACF Log Returns

plt = StylizedFacts.plot_acf_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_acf_log_returns.pdf")

# Plot ACF Abs Log Returns

plt = StylizedFacts.plot_acf_abs_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_acf_abs_log_returns.pdf")

plt = StylizedFacts.plot_all_stylized_facts(calibrated_stylized_facts)
display(plt)
savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_wb_stylised_facts.pdf")


# ABC-PMC MADWE: D, σ, nu, μ
# seed = 13718273
# D, σ, nu, μ = [1.784, 1.455, 0.496, 1.547]
# α = 0.0
# model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ))
# calibrated_price_path = model(seed)[:,1]

# calibrated_stylized_facts = StylizedFactsPlot(calibrated_price_path)
# plt = StylizedFacts.plot_all_stylized_facts(calibrated_stylized_facts)
# display(plt)
# savefig(plt, "$(dirname(@__FILE__))/lob_abc_pmc_we_stylised_facts.pdf")


# Calibrated Parameters SLOB Model

# ABC-PMC BBWM: D, σ, nu, μ, α_slob
seed = 189313131
D, σ, nu, μ, α = [2.087, 0.421, 0.603, 1.202, 106.504]
model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ))
calibrated_price_path = model(seed)[:,1]

calibrated_stylized_facts = StylizedFactsPlot(calibrated_price_path)

# Plot Price num_paths
plt = plot(1:size(calibrated_stylized_facts.price_path, 1), calibrated_stylized_facts.price_path, legend=false,
    xlab="Time", ylab="Mid-Price")
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_price_path.pdf")

# Plot Log Returns Hist

dist = fit(Normal, calibrated_stylized_facts.log_returns)
plt = histogram(calibrated_stylized_facts.log_returns, normalize=true, legend=false)
plot!(plt, x -> pdf(dist, x), xlim=xlims(), legend=false)
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_hist_log_returns.pdf")



# Plot Log Returns
plt = StylizedFacts.plot_log_returns(calibrated_stylized_facts, "")
display(plt)
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_log_returns.pdf")

# Plot QQ Log Returns

plt = StylizedFacts.plot_qq_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_qq_log_returns.pdf")

# Plot ACF Order Flow

plt = StylizedFacts.plot_acf_order_flow(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_acf_order_flow.pdf")

# Plot ACF Log Returns

plt = StylizedFacts.plot_acf_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_acf_log_returns.pdf")

# Plot ACF Abs Log Returns

plt = StylizedFacts.plot_acf_abs_log_returns(calibrated_stylized_facts, "")
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_acf_abs_log_returns.pdf")

plt = StylizedFacts.plot_all_stylized_facts(calibrated_stylized_facts)
display(plt)
savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_wb_stylised_facts.pdf")


# # ABC-PMC MADWE: D, σ, nu, μ, α
# seed = 63612536215
# D, σ, nu, μ, α = [2.087, 0.421, 0.603, 1.202, 106.504]
# model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ))
# calibrated_price_path = model(seed)[:,1]

# calibrated_stylized_facts = StylizedFactsPlot(calibrated_price_path)
# plt = StylizedFacts.plot_all_stylized_facts(calibrated_stylized_facts)
# display(plt)
# savefig(plt, "$(dirname(@__FILE__))/slob_abc_pmc_we_stylised_facts.pdf")