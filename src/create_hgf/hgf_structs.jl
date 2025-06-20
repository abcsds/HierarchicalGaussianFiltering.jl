#####################################
######## Abstract node types ########
#####################################

#Top-level node type
abstract type AbstractNode end

#Input and state node subtypes
abstract type AbstractStateNode <: AbstractNode end
abstract type AbstractInputNode <: AbstractNode end

#Variable type subtypes
abstract type AbstractContinuousStateNode <: AbstractStateNode end
abstract type AbstractContinuousInputNode <: AbstractInputNode end
abstract type AbstractBinaryStateNode <: AbstractStateNode end
abstract type AbstractBinaryInputNode <: AbstractInputNode end
abstract type AbstractCategoricalStateNode <: AbstractStateNode end
abstract type AbstractCategoricalInputNode <: AbstractInputNode end

#Abstract type for node information
abstract type AbstractNodeInfo end
abstract type AbstractInputNodeInfo <: AbstractNodeInfo end
abstract type AbstractStateNodeInfo <: AbstractNodeInfo end

##################################
######## HGF update types ########
##################################

#Supertype for HGF update types
abstract type HGFUpdateType end

#Classic and enhance dupdate types
struct ClassicUpdate <: HGFUpdateType end
struct EnhancedUpdate <: HGFUpdateType end

################################
######## Coupling types ########
################################

#Types for specifying nonlinear transformations
abstract type CouplingTransform end

Base.@kwdef mutable struct LinearTransform <: CouplingTransform
    parameters::Dict = Dict()
end

Base.@kwdef mutable struct NonlinearTransform <: CouplingTransform
    base_function::Function
    first_derivation::Function
    second_derivation::Function
    parameters::Dict = Dict()
end

#Supertypes for coupling types
abstract type CouplingType end
abstract type ValueCoupling <: CouplingType end
abstract type PrecisionCoupling <: CouplingType end

#Concrete value coupling types
Base.@kwdef mutable struct DriftCoupling <: ValueCoupling
    strength::Union{Nothing,Float64} = nothing
    transform::CouplingTransform = LinearTransform()
end
Base.@kwdef mutable struct ProbabilityCoupling <: ValueCoupling
    strength::Union{Nothing,Float64} = nothing
end
Base.@kwdef mutable struct CategoryCoupling <: ValueCoupling end
Base.@kwdef mutable struct ObservationCoupling <: ValueCoupling end

#Concrete precision coupling types
Base.@kwdef mutable struct VolatilityCoupling <: PrecisionCoupling
    strength::Union{Nothing,Float64} = nothing
end
Base.@kwdef mutable struct NoiseCoupling <: PrecisionCoupling
    strength::Union{Nothing,Float64} = nothing
end

############################
######## HGF Struct ########
############################
"""
"""
Base.@kwdef mutable struct OrderedNodes
    all_nodes::Vector{AbstractNode} = AbstractNode[]
    input_nodes::Vector{AbstractInputNode} = []
    all_state_nodes::Vector{AbstractStateNode} = []
    early_update_state_nodes::Vector{AbstractStateNode} = []
    late_update_state_nodes::Vector{AbstractStateNode} = []
end

"""
"""
Base.@kwdef mutable struct HGF <: ActionModels.AbstractSubmodelAttributes
    all_nodes::Dict{String,AbstractNode}
    input_nodes::Dict{String,AbstractInputNode}
    state_nodes::Dict{String,AbstractStateNode}
    ordered_nodes::OrderedNodes = OrderedNodes()
    parameter_groups::Dict = Dict()
    save_history::Bool = true
    timesteps::Vector{Real} = [0]
end

##################################
######## HGF Info Structs ########
##################################
Base.@kwdef struct NodeDefaults
    input_noise::Float64 = -2
    bias::Float64 = 0
    volatility::Float64 = -2
    drift::Float64 = 0
    autoconnection_strength::Float64 = 1
    initial_mean::Float64 = 0
    initial_precision::Float64 = 1
    coupling_strength::Float64 = 1
    update_type::HGFUpdateType = EnhancedUpdate()
end

Base.@kwdef mutable struct ContinuousState <: AbstractStateNodeInfo
    name::String
    volatility::Union{Float64,Nothing} = nothing
    drift::Union{Float64,Nothing} = nothing
    autoconnection_strength::Union{Float64,Nothing} = nothing
    initial_mean::Union{Float64,Nothing} = nothing
    initial_precision::Union{Float64,Nothing} = nothing
end

Base.@kwdef mutable struct ContinuousInput <: AbstractInputNodeInfo
    name::String
    input_noise::Union{Float64,Nothing} = nothing
    bias::Union{Float64,Nothing} = nothing
end

Base.@kwdef mutable struct BinaryState <: AbstractStateNodeInfo
    name::String
end

Base.@kwdef mutable struct BinaryInput <: AbstractInputNodeInfo
    name::String
end

Base.@kwdef mutable struct CategoricalState <: AbstractStateNodeInfo
    name::String
end

Base.@kwdef mutable struct CategoricalInput <: AbstractInputNodeInfo
    name::String
end



#######################################
######## Continuous State Node ########
#######################################
Base.@kwdef mutable struct ContinuousStateNodeEdges
    #Possible parent types
    drift_parents::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()
    volatility_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()

    #Possible children types
    drift_children::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()
    volatility_children::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()
    probability_children::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
    observation_children::Vector{<:AbstractContinuousInputNode} =
        Vector{ContinuousInputNode}()
    noise_children::Vector{<:AbstractContinuousInputNode} = Vector{ContinuousInputNode}()
end

"""
Configuration of continuous state nodes' parameters 
"""
Base.@kwdef mutable struct ContinuousStateNodeParameters{T<:Real}
    volatility::T = 0
    drift::T = 0
    autoconnection_strength::T = 1
    initial_mean::T = 0
    initial_precision::T = 0
    coupling_strengths::Dict{String,T} = Dict{String,Float64}()
    coupling_transforms::Dict{String,CouplingTransform} = Dict{String,CouplingTransform}()
end

"""
Configurations of the continuous state node states
"""
Base.@kwdef mutable struct ContinuousStateNodeState{T<:Real}
    posterior_mean::Union{T} = 0
    posterior_precision::Union{T} = 1
    value_prediction_error::Union{T,Missing} = missing
    precision_prediction_error::Union{T,Missing} = missing
    prediction_mean::Union{T,Missing} = missing
    prediction_precision::Union{T,Missing} = missing
    effective_prediction_precision::Union{T,Missing} = missing
end

"""
Configuration of continuous state node history
"""
Base.@kwdef mutable struct ContinuousStateNodeHistory{T<:Real}
    posterior_mean::Vector{Union{T,Missing}} = []
    posterior_precision::Vector{Union{T,Missing}} = []
    value_prediction_error::Vector{Union{T,Missing}} = []
    precision_prediction_error::Vector{Union{T,Missing}} = []
    prediction_mean::Vector{Union{T,Missing}} = []
    prediction_precision::Vector{Union{T,Missing}} = []
    effective_prediction_precision::Vector{Union{T,Missing}} = []
end

"""
"""
Base.@kwdef mutable struct ContinuousStateNode <: AbstractContinuousStateNode
    name::String
    edges::ContinuousStateNodeEdges = ContinuousStateNodeEdges()
    parameters::ContinuousStateNodeParameters = ContinuousStateNodeParameters()
    states::ContinuousStateNodeState = ContinuousStateNodeState()
    history::ContinuousStateNodeHistory = ContinuousStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end


#######################################
######## Continuous Input Node ########
#######################################
Base.@kwdef mutable struct ContinuousInputNodeEdges
    #Possible parents
    observation_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()
    noise_parents::Vector{<:AbstractContinuousStateNode} = Vector{ContinuousStateNode}()


end

"""
Configuration of continuous input node parameters
"""
Base.@kwdef mutable struct ContinuousInputNodeParameters{T<:Real}
    input_noise::Union{T,Nothing} = nothing
    bias::Union{T,Nothing} = nothing
    coupling_strengths::Dict{String,Union{T,Nothing}} =
        Dict{String,Union{Float64,Nothing}}()
    coupling_transforms::Dict{String,CouplingTransform} = Dict{String,CouplingTransform}()
end

"""
Configuration of continuous input node states
"""
Base.@kwdef mutable struct ContinuousInputNodeState{T<:Real}
    input_value::Union{T,Missing} = missing
    value_prediction_error::Union{T,Missing} = missing
    precision_prediction_error::Union{T,Missing} = missing
    prediction_mean::Union{T,Missing} = missing
    prediction_precision::Union{T,Missing} = missing
end

"""
Configuration of continuous input node history
"""
Base.@kwdef mutable struct ContinuousInputNodeHistory{T<:Real}
    input_value::Vector{Union{T,Missing}} = []
    value_prediction_error::Vector{Union{T,Missing}} = []
    precision_prediction_error::Vector{Union{T,Missing}} = []
    prediction_mean::Vector{Union{T,Missing}} = []
    prediction_precision::Vector{Union{T,Missing}} = []
end

Base.@kwdef mutable struct ContinuousInputNode <: AbstractContinuousInputNode
    name::String
    edges::ContinuousInputNodeEdges = ContinuousInputNodeEdges()
    parameters::ContinuousInputNodeParameters = ContinuousInputNodeParameters()
    states::ContinuousInputNodeState = ContinuousInputNodeState()
    history::ContinuousInputNodeHistory = ContinuousInputNodeHistory()
end

###################################
######## Binary State Node ########
###################################
Base.@kwdef mutable struct BinaryStateNodeEdges
    #Possible parent types
    probability_parents::Vector{<:AbstractContinuousStateNode} =
        Vector{ContinuousStateNode}()

    #Possible children types
    category_children::Vector{<:AbstractCategoricalStateNode} =
        Vector{CategoricalStateNode}()
    observation_children::Vector{<:AbstractBinaryInputNode} = Vector{BinaryInputNode}()
end

"""
 Configure parameters of binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeParameters{T<:Real}
    coupling_strengths::Dict{String,T} = Dict{String,Float64}()
end

"""
Overview of the states of the binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeState{T<:Real}
    posterior_mean::Union{T1,Missing} = missing
    posterior_precision::Union{T,Missing} = missing
    value_prediction_error::Union{T,Missing} = missing
    prediction_mean::Union{T,Missing} = missing
    prediction_precision::Union{T,Missing} = missing
end

"""
Overview of the history of the binary state node
"""
Base.@kwdef mutable struct BinaryStateNodeHistory{T<:Real}
    posterior_mean::Vector{Union{T,Missing}} = []
    posterior_precision::Vector{Union{T,Missing}} = []
    value_prediction_error::Vector{Union{T,Missing}} = []
    prediction_mean::Vector{Union{T,Missing}} = []
    prediction_precision::Vector{Union{T,Missing}} = []
end

"""
Overview of edge posibilities 
"""
Base.@kwdef mutable struct BinaryStateNode <: AbstractBinaryStateNode
    name::String
    edges::BinaryStateNodeEdges = BinaryStateNodeEdges()
    parameters::BinaryStateNodeParameters = BinaryStateNodeParameters()
    states::BinaryStateNodeState = BinaryStateNodeState()
    history::BinaryStateNodeHistory = BinaryStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end



###################################
######## Binary Input Node ########
###################################
Base.@kwdef mutable struct BinaryInputNodeEdges
    observation_parents::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
end

"""
Configuration of parameters in binary input node. Default category mean set to [0,1]
"""
Base.@kwdef mutable struct BinaryInputNodeParameters{T<:Real}
    coupling_strengths::Dict{String,T} = Dict{String,Float64}()
end

"""
Configuration of states of binary input node
"""
Base.@kwdef mutable struct BinaryInputNodeState{T<:Real}
    input_value::Union{T,Missing} = missing
end

"""
Configuration of history of binary input node
"""
Base.@kwdef mutable struct BinaryInputNodeHistory{T<:Real}
    input_value::Vector{Union{T,Missing}} = [missing]
end

"""
"""
Base.@kwdef mutable struct BinaryInputNode <: AbstractBinaryInputNode
    name::String
    edges::BinaryInputNodeEdges = BinaryInputNodeEdges()
    parameters::BinaryInputNodeParameters = BinaryInputNodeParameters()
    states::BinaryInputNodeState = BinaryInputNodeState()
    history::BinaryInputNodeHistory = BinaryInputNodeHistory()
end



########################################
######## Categorical State Node ########
########################################
Base.@kwdef mutable struct CategoricalStateNodeEdges
    #Possible parents
    category_parents::Vector{<:AbstractBinaryStateNode} = Vector{BinaryStateNode}()
    #The order of the category parents
    category_parent_order::Vector{String} = []

    #Possible children
    observation_children::Vector{<:AbstractCategoricalInputNode} =
        Vector{CategoricalInputNode}()
end

Base.@kwdef mutable struct CategoricalStateNodeParameters{T<:Real}
    coupling_strengths::Dict{String,T} = Dict{String,Float64}()
end

"""
Configuration of states in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNodeState{T<:Real}
    posterior::Vector{Union{T,Missing}} = []
    value_prediction_error::Vector{Union{T,Missing}} = []
    prediction::Vector{Union{T,Missing}} = []
    parent_predictions::Vector{Union{T,Missing}} = []
end

"""
Configuration of history in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNodeHistory{T<:Real}
    posterior::Vector{Vector{Union{T,Missing}}} = []
    value_prediction_error::Vector{Vector{Union{T,Missing}}} = []
    prediction::Vector{Vector{Union{T,Missing}}} = []
    parent_predictions::Vector{Vector{Union{T,Missing}}} = []
end

"""
Configuration of edges in categorical state node
"""
Base.@kwdef mutable struct CategoricalStateNode <: AbstractCategoricalStateNode
    name::String
    edges::CategoricalStateNodeEdges = CategoricalStateNodeEdges()
    parameters::CategoricalStateNodeParameters = CategoricalStateNodeParameters()
    states::CategoricalStateNodeState = CategoricalStateNodeState()
    history::CategoricalStateNodeHistory = CategoricalStateNodeHistory()
    update_type::HGFUpdateType = ClassicUpdate()
end



########################################
######## Categorical Input Node ########
########################################
Base.@kwdef mutable struct CategoricalInputNodeEdges
    observation_parents::Vector{<:AbstractCategoricalStateNode} =
        Vector{CategoricalStateNode}()
end

Base.@kwdef mutable struct CategoricalInputNodeParameters{T<:Real}
    coupling_strengths::Dict{String,T} = Dict{String,Float64}()
end

"""
Configuration of states of categorical input node
"""
Base.@kwdef mutable struct CategoricalInputNodeState{T<:Real}
    input_value::Union{T,Missing} = missing
end

"""
History of categorical input node
"""
Base.@kwdef mutable struct CategoricalInputNodeHistory{T<:Real}
    input_value::Vector{Union{T,Missing}} = [missing]
end

"""
"""
Base.@kwdef mutable struct CategoricalInputNode <: AbstractCategoricalInputNode
    name::String
    edges::CategoricalInputNodeEdges = CategoricalInputNodeEdges()
    parameters::CategoricalInputNodeParameters = CategoricalInputNodeParameters()
    states::CategoricalInputNodeState = CategoricalInputNodeState()
    history::CategoricalInputNodeHistory = CategoricalInputNodeHistory()
end
