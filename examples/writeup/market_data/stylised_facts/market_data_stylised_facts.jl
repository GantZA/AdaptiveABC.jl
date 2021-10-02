using SequentialLOB
using StylizedFacts
using Plots
using DelimitedFiles
using AdaptiveABC

# Read in Observed Market Data

observed_price_path = round.(readdlm("$(dirname(@__FILE__))/Original_Price_Bars_2300.csv", ',', Float64, '\n')[:,1], digits=2)
observed_log_returns = diff(log.(observed_price_path[:,1]))
observed_summary_stats = get_summary_stats(observed_log_returns)


plt = plot(
    1:size(observed_log_returns, 1) + 1, observed_price_path, legend=false,
    xlab="Time", ylab="Mid-Price",
)

display(plt)
savefig(plt, "$(dirname(@__FILE__))/market_data_price_path.pdf")

# Plot Log Returns

market_data_stylized_facts = StylizedFactsPlot(observed_price_path)
plt = StylizedFacts.plot_log_returns(market_data_stylized_facts, "")
display(plt)
savefig(plt, "$(dirname(@__FILE__))/market_data_log_rets.pdf")


plt = StylizedFacts.plot_all_stylized_facts(market_data_stylized_facts)
display(plt)
savefig(plt, "$(dirname(@__FILE__))/market_data_stylised_facts.pdf")