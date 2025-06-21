function ActionModels.initialize_attributes(
    submodel::HGF,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    #For now, we don't initialize any attributes
    return submodel

end



### Functions for getting parameter and state types to use with ActionModels ###
function ActionModels.get_parameter_types(hgf::HGF)
    #Return a NamedTuple, all parameters are Float64
    return NamedTuple(
        map(param_name -> param_name => Float64, collect(keys(hgf.parameter_interface))),
    )
end

function ActionModels.get_state_types(hgf::HGF)
    #Return a NamedTuple, all states are Float64
    return NamedTuple(
        map(state_name -> state_name => Float64, collect(keys(hgf.state_interface))),
    )
end
