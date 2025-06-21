export HGFGaussian

Base.@kwdef struct HGFGaussian <: ActionModels.AbstractPremadeModel
    action_noise::Float64 = 1.0
    target_state::Symbol = :x_posterior_mean
    HGF::Union{HGF,String} = "continuous_2level"
end

function ActionModels.ActionModel(config::HGFGaussian)

    #Extract hgf
    if config.HGF isa String
        #If the HGF is a string, we assume it is a name of a premade HGF
        hgf = premade_hgf(config.HGF)
    else
        hgf = config.HGF
    end
    #Extract target state
    target_state = config.target_state

    #Create action model function
    am_function = function hgf_gaussian(
        attributes::ModelAttributes,
        hgf_observation::R,
    ) where {R<:Real}

        #Extract HGF
        hgf = attributes.submodel

        #Extract action noise
        β = load_parameters(attributes).action_noise

        #Update the HGF
        update_hgf!(hgf, hgf_observation)

        #Extract specified belief state
        μ = get_states(hgf, target_state)

        #If the gaussian mean becomes a NaN
        if isnan(μ)
            #Throw an error that will reject samples when fitted
            throw(
                RejectParameters(
                    "With these parameters and inputs, the mean of the gaussian action became $μ, which is invalid. Try other parameter settings",
                ),
            )
        end

        #Create normal distribution with mean of the target value and a standard deviation from parameters
        return Normal(μ, β)
    end

    parameters = (action_noise = Parameter(config.action_noise),)

    observations = (; hgf_observation = Observation(Real))

    actions = (; report = Action(Normal),)

    return ActionModel(
        am_function,
        parameters = parameters,
        observations = observations,
        actions = actions,
        submodel = hgf,
    )

end