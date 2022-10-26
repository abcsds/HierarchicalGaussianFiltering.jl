"""
"""
function premade_hgf_multiple_actions(config::Dict)

    ## Combine defaults and user settings

    #Default parameters and settings
    defaults = Dict(
        "HGF" => "continuous_2level",
        "hgf_actions" =>
            ["gaussian_action", "softmax_action", "unit_square_sigmoid_action"],
    )

    #If there is no HGF in the user-set parameters
    if !("HGF" in keys(config))
        HGF_name = defaults["HGF"]
        #Make a default HGF
        config["HGF"] = premade_hgf(HGF_name)
        #And warn them
        @warn "an HGF was not set by the user. Using the default: a $HGF_name HGF with default settings"
    end

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(defaults, config)

    #Merge to overwrite defaults
    config = merge(defaults, config)


    ## Create agent 
    #Set the action model
    action_model = update_hgf_multiple_actions

    #Set the HGF
    hgf = config["HGF"]

    #Set parameters
    params = Dict()
    #Set states
    states = Dict()
    #Set settings
    settings = Dict("hgf_actions" => config["hgf_actions"])


    ## Add parameters for each action type
    for action_string in config["hgf_actions"]

        #Parameters for the gaussian action
        if action_string == "gaussian_action"

            #Action precision parameter
            if "gaussian_action_precision" in keys(config)
                params["gaussian_action_precision"] = config["gaussian_action_precision"]
            else
                default_action_precision = 1
                params["gaussian_action_precision"] = default_action_precision
                @warn "parameter gaussian_action_precision was not set by the user. Using the default: $default_action_precision"
            end

            #Target state setting
            if "gaussian_target_state" in keys(config)
                settings["gaussian_target_state"] = config["gaussian_target_state"]
            else
                default_target_state = ("x1", "posterior_mean")
                settings["gaussian_target_state"] = default_target_state
                @warn "setting gaussian_target_state was not set by the user. Using the default: $default_target_state"
            end

            #Parameters for the softmax action
        elseif action_string == "softmax_action"

            #Action precision parameter
            if "softmax_action_precision" in keys(config)
                params["softmax_action_precision"] = config["softmax_action_precision"]
            else
                default_action_precision = 1
                params["softmax_action_precision"] = default_action_precision
                @warn "parameter softmax_action_precision was not set by the user. Using the default: $default_action_precision"
            end

            #Target state setting
            if "softmax_target_state" in keys(config)
                settings["softmax_target_state"] = config["softmax_target_state"]
            else
                default_target_state = ("x1", "prediction_mean")
                settings["softmax_target_state"] = default_target_state
                @warn "setting softmax_target_state was not set by the user. Using the default: $default_target_state"
            end

            #Parameters for the unit square sigmoid action
        elseif action_string == "unit_square_sigmoid_action"

            #Action precision parameter
            if "sigmoid_action_precision" in keys(config)
                params["sigmoid_action_precision"] = config["sigmoid_action_precision"]
            else
                default_action_precision = 1
                params["sigmoid_action_precision"] = default_action_precision
                @warn "parameter sigmoid_action_precision was not set by the user. Using the default: $default_action_precision"
            end

            #Target state setting
            if "sigmoid_target_state" in keys(config)
                settings["sigmoid_target_state"] = config["sigmoid_target_state"]
            else
                default_target_state = ("x1", "prediction_mean")
                settings["sigmoid_target_state"] = default_target_state
                @warn "setting sigmoid_target_state was not set by the user. Using the default: $default_target_state"
            end

        else
            throw(
                ArgumentError(
                    "$action_string is not a valid action type. Valid action types are: gaussian_action, softmax_action, unit_square_sigmoid_action",
                ),
            )
        end
    end

    #Create the agent
    return init_agent(
        action_model;
        substruct = hgf,
        params = params,
        states = states,
        settings = settings,
    )
end


"""
    premade_hgf_gaussian(
        hgf = premade_hgf("continuous_2level"),
        action_precision = 1,
        target_node = "x1",
        target_state = "posterior_mean",
    )

Function that initializes as premade HGF gaussian action agent
"""
function premade_hgf_gaussian(config::Dict)

    ## Combine defaults and user settings

    #Default parameters and settings
    defaults = Dict(
        "gaussian_action_precision" => 1,
        "target_state" => ("x1", "posterior_mean"),
        "HGF" => "continuous_2level",
    )

    #If there is no HGF in the user-set parameters
    if !("HGF" in keys(config))
        HGF_name = defaults["HGF"]
        #Make a default HGF
        config["HGF"] = premade_hgf(HGF_name)
        #And warn them
        @warn "an HGF was not set by the user. Using the default: a $HGF_name HGF with default settings"
    end

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(defaults, config)

    #Merge to overwrite defaults
    config = merge(defaults, config)


    ## Create agent 
    #Set the action model
    action_model = update_hgf_gaussian_action

    #Set the HGF
    hgf = config["HGF"]

    #Set parameters
    params = Dict("gaussian_action_precision" => config["gaussian_action_precision"])
    #Set states
    states = Dict()
    #Set settings
    settings = Dict("target_state" => config["target_state"])

    #Create the agent
    return init_agent(
        action_model;
        substruct = hgf,
        params = params,
        states = states,
        settings = settings,
    )
end

"""
    premade_hgf_binary_softmax(
        hgf = premade_hgf("binary_3level"),
        action_precision = 1,
        target_node = "x1",
        target_state = "posterior_mean",
    )

Function that initializes as premade HGF binary softmax action agent
"""
function premade_hgf_binary_softmax(config::Dict)

    ## Combine defaults and user settings

    #Default parameters and settings
    defaults = Dict(
        "softmax_action_precision" => 1,
        "target_state" => ("x1", "prediction_mean"),
        "HGF" => "binary_3level",
    )

    #If there is no HGF in the user-set parameters
    if !("HGF" in keys(config))
        HGF_name = defaults["HGF"]
        #Make a default HGF
        config["HGF"] = premade_hgf(HGF_name)
        #And warn them
        @warn "an HGF was not set by the user. Using the default: a $HGF_name HGF with default settings"
    end

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(defaults, config)

    #Merge to overwrite defaults
    config = merge(defaults, config)


    ## Create agent 
    #Set the action model
    action_model = update_hgf_binary_softmax_action

    #Set the HGF
    hgf = config["HGF"]

    #Set parameters
    params = Dict("softmax_action_precision" => config["softmax_action_precision"])
    #Set states
    states = Dict()
    #Set settings
    settings = Dict("target_state" => config["target_state"])

    #Create the agent
    return init_agent(
        action_model,
        substruct = hgf,
        params = params,
        states = states,
        settings = settings,
    )
end

"""
    premade_hgf_unit_square_sigmoid(
        hgf = premade_hgf("binary_3level"),
        action_precision = 1,
        target_node = "x1",
        target_state = "posterior_mean",
    )

Function that initializes as premade HGF binary softmax action agent
"""
function premade_hgf_unit_square_sigmoid(config::Dict)

    ## Combine defaults and user settings

    #Default parameters and settings
    defaults = Dict(
        "sigmoid_action_precision" => 1,
        "target_state" => ("x1", "prediction_mean"),
        "HGF" => "binary_3level",
    )

    #If there is no HGF in the user-set parameters
    if !("HGF" in keys(config))
        HGF_name = defaults["HGF"]
        #Make a default HGF
        config["HGF"] = premade_hgf(HGF_name)
        #And warn them
        @warn "an HGF was not set by the user. Using the default: a $HGF_name HGF with default settings"
    end

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(defaults, config)

    #Merge to overwrite defaults
    config = merge(defaults, config)


    ## Create agent 
    #Set the action model
    action_model = update_hgf_unit_square_sigmoid_action

    #Set the HGF
    hgf = config["HGF"]

    #Set parameters
    params = Dict("sigmoid_action_precision" => config["sigmoid_action_precision"])
    #Set states
    states = Dict()
    #Set settings
    settings = Dict("target_state" => config["target_state"])

    #Create the agent
    return init_agent(
        action_model,
        substruct = hgf,
        params = params,
        states = states,
        settings = settings,
    )
end
