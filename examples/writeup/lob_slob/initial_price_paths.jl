using SequentialLOB
using StylizedFacts
using Plots

# Configuration Arguments
num_paths = 1
M = 400
T = 2299
p₀ = 238.75
L = 200
Δx = L / M 

# Free-Parameters
λ = 1.0
D = 1.0
Δt = (Δx^2) / (2.0 * D)
σ = 1.5
μ = 1.0
nu = 0.3
α_slob = 500.0
α_lob = 0.0
println(Δx)
println(Δt)
# LOB Model

lob_model = SLOB(num_paths,
T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))

sample_price_paths = lob_model(7136) 
plt = plot(
    1:T + 1, sample_price_paths[:,1], legend=false,
    xlab="Time", ylab="Mid-Price",
)
for i in 2:num_paths
    plot!(plt, 1:T + 1, sample_price_paths[:,i], legend=false)
end
display(plt)
savefig(plt, "examples/writeup/lob_slob/lob_model_init_pp.pdf")

# SLOB Model

slob_model = SLOB(num_paths,
T, p₀, M, L, D, σ, nu, α_slob, SourceTerm(λ, μ))

sample_price_paths = slob_model(19822) 
plt = plot(
    1:T + 1, sample_price_paths[:,1], legend=false,
    xlab="Time", ylab="Mid-Price",
)
for i in 2:num_paths
    plot!(plt, 1:T + 1, sample_price_paths[:,i], legend=false)
end
display(plt)
savefig(plt, "examples/writeup/lob_slob/slob_model_init_pp.pdf")

# Compare LOB to SLOB

lob_mid_prices = lob_model(7136)
lob_stylized_facts = StylizedFactsPlot(lob_mid_prices[:,1])
plt = StylizedFacts.plot_log_returns(lob_stylized_facts)
display(plt)
savefig(plt, "examples/writeup/lob_slob/lob_model_init_log_rets.pdf")


slob_mid_prices = slob_model(19822)
slob_stylized_facts = StylizedFactsPlot(slob_mid_prices[:,1])
plt = StylizedFacts.plot_log_returns(slob_stylized_facts)
display(plt)
savefig(plt, "examples/writeup/lob_slob/slob_model_init_log_rets.pdf")

lob_model = SLOB(10,
T, p₀, M, L, 0.4, σ, nu, α_lob, SourceTerm(λ, μ))

sample_price_paths = lob_model() 
plt = plot(
    1:T + 1, sample_price_paths[:,1], legend=false,
    xlab="Time", ylab="Mid-Price",
)
for i in 2:10
    plot!(plt, 1:T + 1, sample_price_paths[:,i], legend=false)
end
display(plt)