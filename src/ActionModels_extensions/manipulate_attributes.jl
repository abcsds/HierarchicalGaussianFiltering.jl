function ActionModels.reset!(attributes::RescorlaWagnerAttributes)

end




function ActionModels.get_parameters(attributes::RescorlaWagnerAttributes)
    return (;
        learning_rate = attributes.learning_rate,
        initial_value = attributes.initial_value,
    )
end
function ActionModels.get_states(attributes::RescorlaWagnerAttributes)
    return (; expected_value = attributes.expected_value)
end






function ActionModels.get_parameters(
    attributes::RescorlaWagnerAttributes,
    parameter_name::Symbol,
)
    if parameter_name in [:learning_rate, :initial_value]
        return getfield(attributes, parameter_name)
    else
        return AttributeError()
    end
end
function ActionModels.get_states(attributes::RescorlaWagnerAttributes, state_name::Symbol)
    if state_name in [:expected_value]
        return getfield(attributes, state_name)
    else
        return AttributeError()
    end
end







function ActionModels.set_parameters!(
    attributes::RescorlaWagnerAttributes,
    parameter_name::Symbol,
    parameter_value::T,
) where {T<:Real}
    if parameter_name in [:learning_rate, :initial_value]
        setfield!(attributes, parameter_name, parameter_value)
        return true
    else
        return AttributeError()
    end
end
function ActionModels.set_states!(
    attributes::RescorlaWagnerAttributes,
    state_name::Symbol,
    state_value::T,
) where {T<:Real}
    if state_name in [:expected_value]
        setfield!(attributes, state_name, state_value)
        return true
    else
        return AttributeError()
    end
end
