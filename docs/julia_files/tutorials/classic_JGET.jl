using ActionModels, HierarchicalGaussianFiltering
using CSV, DataFrames
using StatsPlots

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/julia_files/tutorials/data/"

#Load data
data = CSV.read(data_path * "classic_cannonball_data.csv", DataFrame)
inputs = data[(data.ID .== 20) .& (data.session .== 1), :].outcome

#Create HGF
hgf = premade_hgf("JGET", verbose = false)

action_model = ActionModel(HGFGaussian(; HGF = hgf))
agent = init_agent(action_model)

#Set parameters
parameters = (
    action_noise = 1,
    u_input_noise = 0,
    x_initial_mean = first(inputs) + 2,
    x_initial_precision = 0.001,
    x_volatility = -8,
    xvol_volatility = -8,
    xnoise_volatility = -7,
    xnoise_vol_volatility = -2,
    x_xvol_coupling_strength = 1,
    xnoise_xnoise_vol_coupling_strength = 1,
)
set_parameters!(agent, parameters)
reset!(agent)

#Simulate updates and actions
actions = simulate!(agent, inputs);
#Plot belief trajectories
plot(agent, "u")
plot!(agent, "x")
plot(agent, "xvol")
plot(agent, "xnoise")
plot(agent, "xnoise_vol")
