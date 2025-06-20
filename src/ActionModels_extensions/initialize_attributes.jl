function ActionModels.initialize_attributes(
    submodel::HGF,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    #For now, we don't initialize any attributes
    return deepcopy(submodel)

end