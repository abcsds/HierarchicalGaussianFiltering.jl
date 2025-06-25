# # Welcome to The Hierarchical Gaussian Filtering Package!

# Hierarchical Gaussian Filtering (HGF) is a novel and adaptive package for doing cognitive and behavioral modelling. With the HGF you can fit time series data fit participant-level individual parameters, measure group differences based on model-specific parameters or use the model for any time series with underlying change in uncertainty.

# NOTE: the documentation is currently under reconstruction, and is outdated. All code snippets are tested and functional, but written descriptions may not currently be accurate.

# The HGF consists of a network of probabilistic nodes hierarchically structured. The hierarchy is determined by the coupling between nodes. A node (child node) in the network can inheret either its value or volatility sufficient statistics from a node higher in the hierarchy (a parent node). 

# The presentation of a new observation at the lower level of the hierarchy (i.e. the input node) trigger a recursuve update of the nodes belief throught the bottom-up propagation of precision-weigthed prediction error.

# The HGF will be explained in more detail in the theory section of the documentation

# It is also recommended to check out the ActionModels.jl pacakge for stronger intuition behind the use of agents and action models. 

# ## Getting started

# The last official release can be downloaded from Julia with "] add HierarchicalGaussianFiltering"

# We provide a script for getting started with commonly used functions and use cases

# Load packages 
using HierarchicalGaussianFiltering
using ActionModels

# ### Create agent
action_model = ActionModel(HGFSoftmax(; HGF = "binary_3level"))
agent = init_agent(action_model, save_history = :xbin_prediction_mean)

# ### Get states and parameters
get_states(agent)
#-
get_parameters(agent)

# Set a new parameter for initial precision of xprob and define some inputs
set_parameters!(agent, (; xprob_initial_precision = 0.9))
inputs = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0];

# ### Give inputs to the agent
actions = simulate!(agent, inputs)

# ### Plot state trajectories of input and prediction
using StatsPlots
plot(agent, ("u", "input_value"))
plot!(agent, ("xbin", "prediction"))

# Plot state trajectory of input value, action and prediction of xbin
plot(agent, ("u", "input_value"))
plot!(actions .+ 0.1, seriestype = :scatter, label = "action")
plot!(agent, ("xbin", "prediction"))


# ### Fitting parameters

prior = (; xprob_volatility = Normal(-7, 0.5))

#Create model
model = create_model(action_model, prior, inputs, Int64.(actions), check_parameter_rejections = true)

#Fit 
posterior_chains = sample_posterior!(model, n_samples = 200, n_chains = 2)

#-

# ### Plot chains
plot(posterior_chains)