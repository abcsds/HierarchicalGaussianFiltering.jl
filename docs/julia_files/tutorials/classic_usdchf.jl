# # Tutorial on 2-level continuous HGF

#This is a replication of the tutorial from the MATLAB toolbox, using an HGF to filter the exchange rates between USD and CHF

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using StatsPlots

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/julia_files/tutorials/data/"

# Load the data
inputs = Float64[]
open(data_path * "classic_usdchf_inputs.dat") do f
    for ln in eachline(f)
        push!(inputs, parse(Float64, ln))
    end
end

#Create HGF
hgf = premade_hgf("continuous_2level", verbose = false);

action_model = ActionModel(HGFGaussian(; HGF = hgf))
agent = init_agent(action_model)

# Set parameters for parameter recover
parameters = (
    x_xvol_coupling_strength = 1.0,
    u_input_noise = -log(1e4),
    x_volatility = -13,
    xvol_volatility = -2,
    x_initial_mean = 1.04,
    x_initial_precision = 1 / (0.0001),
    xvol_initial_mean = 1.0,
    xvol_initial_precision = 1 / 0.1,
    action_noise = 0.01,
);

set_parameters!(agent, parameters)
reset!(agent)

# Evolve agent
actions = simulate!(agent, inputs);

# Plot trajectories
plot(
    agent,
    "u",
    size = (1300, 500),
    xlims = (0, 615),
    markersize = 3,
    markercolor = "green2",
    title = "HGF trajectory",
    ylabel = "CHF-USD exchange rate",
    xlabel = "Trading days since 1 January 2010",
)
#-
plot!(agent, ("x", "posterior"), color = "red")
plot!(actions, size = (1300, 500), xlims = (0, 614), markersize = 3, markercolor = "orange")
#-
plot(
    agent,
    "xvol",
    color = "blue",
    size = (1300, 500),
    xlims = (0, 615),
    xlabel = "Trading days since 1 January 2010",
    title = "Volatility parent trajectory",
)
#-
# Set priors for fitting
priors = (
    u_input_noise = Normal(-6, 1),
    x_volatility = Normal(-4, 1),
    xvol_volatility = Normal(-4, 1),
    action_noise = LogNormal(log(0.01), 1),
);

# Do parameter recovery
model = create_model(action_model, priors, inputs, actions, check_parameter_rejections = true)

#Fit 
posterior_chains = sample_posterior!(
    model,
    n_samples = 200,
    n_chains = 2,
)

#-
# Plot the chains
plot(posterior_chains)