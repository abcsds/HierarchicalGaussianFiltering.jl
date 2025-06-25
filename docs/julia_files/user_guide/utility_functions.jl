# # Variations of utility functions in the Hierarchical Gaussian Filtering package


# A lot of commonly used utility functions are collected here in an overview with examples. The following utility functions can be used:

# 1. [Getting Parameters](#Getting-Parameters)
# 2. [Getting States](#Getting-States)
# 3. [Setting Parameters](#Setting-Parameters)
# 4. [Giving Inputs](#Giving-Inputs)
# 5. [Getting History](#Getting-History)
# 6. [Plotting State Trajectories](#Plotting-State-Trajectories)
# 7. [Getting Predictions](#Getting-Predictions)
# 8. [Getting Surprise](#Getting-Surprise)
# 9. [Resetting an HGF-agent](#Resetting-an-HGF-agent)

# we start by defining an agent to use
using HierarchicalGaussianFiltering

# set agent
action_model = ActionModel(HGFSoftmax(; HGF = "binary_3level"))

agent = init_agent(action_model)


# ### Getting Parameters

#Let us start by defining a premade agent:

#getting all parameters 
get_parameters(agent)

# getting couplings
get_parameters(agent, :xprob_xvol_coupling_strength)

# getting multiple parameters specify them in a vector
get_parameters(agent, (:xvol_volatility, :xvol_initial_precision))


# ### Getting States

#getting all states from an agent model
get_states(agent)

#getting a single state
get_states(agent, :xprob_posterior_precision)

#getting multiple states
get_states(agent, (:xprob_posterior_precision, :xprob_effective_prediction_precision))


# ### Setting Parameters

# you can set parameters before you initialize your agent, you can set them after and change them when you wish to.
# Let's try an initialize a new agent with parameters. We start by choosing the premade unit square sigmoid action agent whose parameter is sigmoid action precision.
# We also specify our HGF and custom parameter settings:

hgf_parameters = Dict(
    ("xprob", "volatility") => -2.5,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "volatility") => -6.0,
    ("xvol", "initial_mean") => 1,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1.0,
    ("xprob", "xvol", "coupling_strength") => 1.0,
)

hgf = premade_hgf("binary_3level", hgf_parameters)

# Define our agent with the HGF and agent parameter settings
action_model = ActionModel(HGFSigmoid(; HGF = hgf, action_noise = 1.0))

agent = init_agent(
    action_model,
    save_history = [:xbin_prediction_mean, :xvol_posterior_precision],
)


# Changing a single parameter

set_parameters!(agent, :xvol_initial_precision, 4)

# Changing multiple parameters

set_parameters!(agent, (xvol_initial_precision = 5, xbin_xprob_coupling_strength = 2.0))

# ###Giving Inputs


#give single input
simulate!(agent, [0])

#-

#reset the agent
reset!(agent)

# Giving multiple inputs
inputs = [
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
]
simulate!(agent, inputs)

# ### Getting History

#getting the history from the agent
get_history(agent)

#-

# getting history of single state
get_history(agent, :xvol_posterior_precision)

#-

# getting history of multiple states:
get_history(agent, (:xbin_prediction_mean, :xvol_posterior_precision))

# ### Plotting State Trajectories

using StatsPlots
## Plotting single state:
plot(agent, ("u", "input_value"))

#Adding state trajectory on top
plot!(agent, ("xbin", "prediction"))

# Plotting more individual states:



## Plot posterior of xprob
plot(agent, ("xprob", "posterior"))

#-

## Plot posterior of xvol
plot(agent, ("xvol", "posterior"))

# ### Getting Predictions

# You can specify an HGF or an agent in the funciton. 

#specify another node to get predictions from:
get_prediction(agent.model_attributes.submodel, "xprob")

# ### Getting Purprise

#getting surprise of input node
get_surprise(agent.model_attributes.submodel, "u")

# ### Resetting an HGF-agent

# resetting the agent with reset()

reset!(agent)

# see that action state is cleared
get_history(agent)
