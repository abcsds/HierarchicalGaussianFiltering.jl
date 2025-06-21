export HGFPredictCategory

Base.@kwdef struct HGFPredictCategory <: ActionModels.AbstractPremadeModel
    action_noise::Float64 = 1.0
    target_node::Symbol = "xcat"
    HGF::Union{HGF,String} = "categorical_3level"
end

function ActionModel(config::HGFPredictCategory)

    #Extract hgf
    if config.HGF isa String
        #If the HGF is a string, we assume it is a name of a premade HGF
        hgf = premade_hgf(config.HGF)
    else
        hgf = config.HGF
    end

    #Extract target state
    target_state = join((config.target_state, "prediction"), "_")

    #Create action model function
    am_function = function hgf_gaussian(attributes::ModelAttributes, hgf_observation::Int64)

        #Extract HGF
        hgf = attributes.submodel

        #Extract the inverse temperature
        β = 1/load_parameters(attributes).action_noise

        #Update the HGF
        update_hgf!(hgf, hgf_observation)

        #Extract specified belief state
        probabilities = get_states(hgf, target_state)

        #Softmax transform with the inverse noise as precision
        probabilities = softmax(probabilties * β)

        return Categorical(probabilities)
    end

    parameters = (; action_noise = Parameter(config.action_noise))

    observations = (; hgf_observation = Observation(discrete = true))

    actions = (; report = Action(Categorical),)

    return ActionModel(
        am_function,
        parameters = parameters,
        observations = observations,
        actions = actions,
    )

end