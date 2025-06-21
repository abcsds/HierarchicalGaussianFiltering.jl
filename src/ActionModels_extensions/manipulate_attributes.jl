

function ActionModels.get_parameters(hgf::HGF)
    return NamedTuple(
        map(
            param_name -> param_name => get_parameters(hgf, param_name),
            Tuple(keys(hgf.parameter_interface)),
        ),
    )

end

function ActionModels.get_states(hgf::HGF)
    return NamedTuple(
        map(
            state_name -> state_name => get_states(hgf, state_name),
            Tuple(keys(hgf.state_interface)),
        ),
    )
end






function ActionModels.get_parameters(hgf::HGF, parameter_name::Symbol)
    if parameter_name in keys(hgf.parameter_interface)
        return get_parameters(hgf, hgf.parameter_interface[parameter_name])
    else
        return AttributeError()
    end
end

function ActionModels.get_states(hgf::HGF, state_name::Symbol)
    if state_name in keys(hgf.state_interface)
        return get_states(hgf, hgf.state_interface[state_name])
    else
        return AttributeError()
    end
end



function ActionModels.set_parameters!(
    hgf::HGF,
    parameter_name::Symbol,
    parameter_value::T,
) where {T}

    #If the parameter exists
    if parameter_name in keys(hgf.parameter_interface)

        #Set the parameter in the HGF
        set_parameters!(hgf, hgf.parameter_interface[parameter_name], parameter_value)

        return true
    else
        return AttributeError()
    end
end

function ActionModels.set_states!(hgf::HGF, state_name::Symbol, state_value::T) where {T}

    if state_name in keys(hgf.state_interface)

        set_states!(hgf, hgf.state_interface[state_name], state_value)

        return true
    else
        return AttributeError()
    end
end
