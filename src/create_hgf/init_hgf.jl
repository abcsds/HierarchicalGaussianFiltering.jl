function init_hgf(;
    nodes::Vector{<:AbstractNodeInfo},
    edges::Dict{Tuple{String,String},<:CouplingType},
    node_defaults::NodeDefaults = NodeDefaults(),
    parameter_groups::Vector{ParameterGroup} = Vector{ParameterGroup}(),
    update_order::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = true,
    save_history::Bool = true,
)

    ### Initialize nodes ###
    #Initialize empty dictionaries for storing nodes
    all_nodes_dict = Dict{String,AbstractNode}()
    input_nodes_dict = Dict{String,AbstractInputNode}()
    state_nodes_dict = Dict{String,AbstractStateNode}()
    input_nodes_inputted_order = Vector{String}()
    state_nodes_inputted_order = Vector{String}()

    #Dictionaries for storing the interface between symbol param/state names and the positions in the node
    parameter_interface = Dict{Symbol, Tuple}()
    state_interface = Dict{Symbol, Tuple}()

    #For each specified input node
    for node_info in nodes
        #For each field in the node info
        for fieldname in fieldnames(typeof(node_info))
            #If it hasn't been specified by the user
            if isnothing(getfield(node_info, fieldname))
                #Set the node_defaults' value instead
                setfield!(node_info, fieldname, getfield(node_defaults, fieldname))
            end
        end

        #Create the node
        node = init_node(node_info)

        #Add it to the large dictionary
        all_nodes_dict[node_info.name] = node

        #If it is an input node
        if node isa AbstractInputNode
            #Add it to the input node dict
            input_nodes_dict[node_info.name] = node
            #Store its name in the inputted order
            push!(input_nodes_inputted_order, node_info.name)

            #If it is a state node
        elseif node isa AbstractStateNode
            #Add it to the state node dict
            state_nodes_dict[node_info.name] = node
            #Store its name in the inputted order
            push!(state_nodes_inputted_order, node_info.name)
        end

        #Add joined names for parameters and states to the ActionModels interface
        for param_name in String.(fieldnames(typeof(node.parameters)))
            parameter_interface[Symbol(join((node.name, param_name), "_"))] = (node.name, param_name)
        end
        for state_name in String.(fieldnames(typeof(node.states)))
            state_interface[Symbol(join((node.name, state_name), "_"))] = (node.name, state_name)
        end

    end

    ### Set up edges ###
    #For each specified edge
    for (node_names, coupling_type) in edges

        #Extract the child and parent names
        child_name, parent_name = node_names

        #Find corresponding child node and parent node
        child_node = all_nodes_dict[child_name]
        parent_node = all_nodes_dict[parent_name]

        #Create the edge
        init_edge!(child_node, parent_node, coupling_type, node_defaults)
    end

    ## Determine Update order ##
    #If update order has not been specified
    if isnothing(update_order)

        #If verbose
        if verbose
            #Warn that automatic update order is used
            @warn "No update order specified. Using the order in which nodes were inputted"
        end

        #Use the order that the nodes were specified in
        update_order = append!(input_nodes_inputted_order, state_nodes_inputted_order)
    end

    ## Order nodes ##
    #Initialize empty struct for storing nodes in correct update order
    ordered_nodes = OrderedNodes()

    #For each node, in the specified update order
    for node_name in update_order

        #Extract node
        node = all_nodes_dict[node_name]

        #Have a field for all nodes
        push!(ordered_nodes.all_nodes, node)

        #Put input nodes in one field
        if node isa AbstractInputNode
            push!(ordered_nodes.input_nodes, node)
        end

        #Put state nodes in another field
        if node isa AbstractStateNode
            push!(ordered_nodes.all_state_nodes, node)

            #If any of the nodes' value children are continuous input nodes
            if any(isa.(node.edges.observation_children, ContinuousInputNode))
                #Add it to the early update list
                push!(ordered_nodes.early_update_state_nodes, node)
            else
                #Otherwise to the late update list
                push!(ordered_nodes.late_update_state_nodes, node)
            end
        end
    end

    #initializing shared parameters
    parameter_groups_dict = Dict()

    #Go through each specified shared parameter
    for parameter_group in parameter_groups

        #Add as a GroupedParameters to the shared parameter dictionary
        parameter_groups_dict[parameter_group.name] = ActionModels.GroupedParameters(
            value = parameter_group.value,
            grouped_parameters = parameter_group.parameters,
        )
    end

    ### Create HGF struct ###
    hgf = HGF(
        all_nodes_dict,
        input_nodes_dict,
        state_nodes_dict,
        ordered_nodes,
        parameter_groups_dict,
        parameter_interface,
        state_interface,
        save_history,
        [0],
    )

    ### Check that the HGF has been specified properly ###
    check_hgf(hgf)

    ### Initialize states and history ###
    #For each state node
    for node in hgf.ordered_nodes.all_state_nodes
        #If it is a categorical state node
        if node isa CategoricalStateNode

            #Make vector with ordered category parents
            for parent in node.edges.category_parents
                push!(node.edges.category_parent_order, parent.name)
            end

            #Set posterior to vector of missing with length equal to the number of categories
            node.states.posterior =
                Vector{Union{Real,Missing}}(missing, length(node.edges.category_parents))

            #Set posterior to vector of missing with length equal to the number of categories
            node.states.value_prediction_error =
                Vector{Union{Real,Missing}}(missing, length(node.edges.category_parents))

            #Set parent predictions from last timestep to be agnostic
            node.states.parent_predictions = repeat(
                [1 / length(node.edges.category_parents)],
                length(node.edges.category_parents),
            )

            #Set predictions from last timestep to be agnostic
            node.states.prediction = repeat(
                [1 / length(node.edges.category_parents)],
                length(node.edges.category_parents),
            )
        end
    end

    #Reset the hgf, initializing states and history
    reset!(hgf)

    return hgf
end
