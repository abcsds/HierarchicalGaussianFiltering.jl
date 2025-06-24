# # Fitting parameters in HGF agents


# - [Introduction To Fitting Models](#Introduction-To-Fitting-Models)

# - [Setting Priors and The Fit_model() Function](#Setting-Priors-and-The-Fit_model()-Function)

# - [Plotting Functions](#Plotting-Functions)

# - [Predictive Simulations](#Predictive-Simulations)



# ## Introduction To Fitting Models Function

# When you work with participants' data in HGF-agents and you have one or more target parameters in sight for investigation, you can recover them with model fitting. When you fit models for different groups of participant, you can idnetify group differences based on the parameter recovery.

# ## Setting Priors and The Fit_model() 

# Hierarchical Gaussian Filtering uses the fit_model() function from the ActionModels.jl package. 
# The fit_ model() function takes the following inputs:

# ![Image1](../images/fit_model_image.png)

# Let us run through the inputs to the function one by one. 

# - agent::Agent: a specified agent created with either premade agent or init\_agent.
# - param_priors::Dict: priors (written as distributions) for the parameters you wish to fit. e.g. priors = Dict("learning_rate" => Uniform(0, 1))
# - inputs:Array: array of inputs.
# - actions::Array: array of actions.
# - fixed_parameters::Dict = Dict(): fixed parameters if you wish to change the parameter settings of the parameters you dont fit
# - sampler = NUTS(): specify the type of sampler. See Turing documentation for more details on sampler types.
# - n_iterations = 1000: iterations pr. chain.
# - n_chains = 1: amount of chains.
# - verbose = true: set to false to hide warnings
# - show\_sample\_rejections = false: if set to true, get a message every time a sample is rejected.
# - impute\_missing\_actions = false : if true, include missing actions in the fitting process.

# We will run through an example of fitting an agent model to data.

# load packages
using ActionModels
using HierarchicalGaussianFiltering

# We will define a binary 3-level HGF and its parameters

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

hgf = premade_hgf("binary_3level", hgf_parameters, verbose = false)
action_model = ActionModel(HGFSoftmax(; HGF = hgf, action_noise = 0.2));

# Create an agent
agent = init_agent(action_model, save_history = :xbin_prediction_mean);

# Define a set of inputs
inputs =
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0];

# Evolve agent and save actions
actions = simulate!(agent, inputs)


# We can  by plotting the actions our agent has produced.
using StatsPlots
plot(agent, "u")
plot!(agent, :xbin_prediction_mean)



# When defining the fixed parameters for fit_model() it overrites any previous parameter settings with the "newly" defined fixed parameters. If you dont state any fixed parameters it uses the current parameter values.

# We define a set of fixed parameters to use in this fitting process:

# Set fixed parameters. We choose to fit the evolution rate of the xprob node. 

# As you can read from the fixed parameters, the evolution rate of xprob is not configured. We set the prior for the xprob evolution rate:
prior = (; xprob_volatility = Normal(-3.0, 0.5))

# We can fit the evolution rate by inputting the variables:

# Create model
model = create_model(action_model, prior, inputs, Int64.(actions), check_parameter_rejections = true)

# Now we can fit the model using the sample_posterior! function.
posterior_chains = sample_posterior!(
    model,
    n_samples = 200,
    n_chains = 2,
)

plot(posterior_chains)