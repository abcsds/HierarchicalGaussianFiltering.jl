@recipe function f(
    agent::Union{Agent{<:HGF},HGF},
    node::String,
    state::Union{String,Nothing} = nothing;
)

    #Extract HGF
    if agent isa Agent{HGF}
        hgf = agent.model_attributes.submodel
    else
        hgf = agent
    end

    #If the target node is not in in the HGF
    if !(node in keys(hgf.all_nodes))
        #Throw an error
        throw(ArgumentError("The node $node does not exist"))
    end

    #Extract node
    selected_node = hgf.all_nodes[node]

    #If default state is used
    if isnothing(state)
        #Plot posterior for continuous state nodes
        if selected_node isa ContinuousStateNode
            state = "posterior"

            #Plot prediction for binary state nodes
        elseif selected_node isa BinaryStateNode
            state = "prediction"

            #Plot prediction for categorical state nodes
        elseif selected_node isa CategoricalStateNode
            state = "prediction"

            #Plot the input value for input nodes
        elseif selected_node isa AbstractInputNode
            state = "input_value"
        end
    end

    #Get x-axis; the number of timesteps
    timesteps = hgf.timesteps

    #If the entire distribution is to be plotted
    if state in ["posterior", "prediction"] && !(selected_node isa CategoricalStateNode)

        #Get the history of the mean
        history_mean = getproperty(selected_node.history, Symbol(state * "_mean"))
        #Replace missings with NaN's for plotting
        history_mean = replace(history_mean, missing => NaN)

        #Get the history of precisions
        history_precision = getproperty(selected_node.history, Symbol(state * "_precision"))
        #Replace missings with NaN's for plotting
        history_precision = replace(history_precision, missing => NaN)
        #Transform precisions into standard deviations
        history_sd = sqrt.(1 ./ history_precision)

        @series begin
            #Set legend label
            label --> node * " " * state
            title --> "State trajectory"

            #Unless its a binary state node
            if !(selected_node isa BinaryStateNode)
                #The ribbon is the standard deviations
                ribbon := history_sd
            end

            #Plot the history of means
            (timesteps, history_mean)
        end

        #If single state is specified
    else
        #Get history of state
        state_history = getproperty(selected_node.history, Symbol(state))
        #Replace missings with NaNs for plotting
        state_history = replace(state_history, missing => NaN)

        #Begin the plot
        @series begin

            #For input values
            if state == "input_value"
                #Default to scatterplots
                seriestype --> :scatter
            else
                #Lineplots for others
                seriestype --> :path
            end

            #The categorical state node has a vector fo vectors as history
            if selected_node isa CategoricalStateNode
                #So it needs to be collapsed into a matrix
                state_history = reduce(vcat, transpose.(state_history))

                #Set the labels to be the category numbers
                category_numbers = collect(1:size(state_history, 2))
                category_labels = "Category " .* string.(category_numbers)
                label --> permutedims(category_labels)
            else
                #Set label
                label --> node * " " * state
            end

            #Set title
            title --> "State trajectory"

            #Plot the history
            (timesteps, state_history)
        end
    end

end
