using ActionModels
using HierarchicalGaussianFiltering
using Test

@testset "Premade Action Models" begin

    @testset "hgf_gaussian" begin

        #Create HGF
        hgf = premade_hgf("continuous_2level", verbose = false)

        #Create action model
        action_model = ActionModel(HGFGaussian(; HGF = hgf))

        #Initialize agent
        agent = init_agent(action_model)

        #Give inputs to the agent
        actions = simulate!(agent, [0.01, 0.02, 0.03])

        #Check that actions are floats
        @test actions isa Vector

        #Check that get_surprise works
        @test get_surprise(agent.model_attributes.submodel) isa Real
    end

    @testset "hgf_binary_softmax" begin

        #Create HGF agent with binary softmax action
        hgf = premade_hgf("binary_3level", verbose = false)
        action_model = ActionModel(HGFSoftmax(; HGF = hgf))

        #Initialize agent
        agent = init_agent(action_model)

        #Give inputs to the agent
        actions = simulate!(agent, [1, 0, 1])

        #Check that actions are floats
        @test actions isa Vector

        #Check that get_surprise works
        @test get_surprise(agent.model_attributes.submodel) isa Real
    end


    @testset "hgf_unit_square_sigmoid" begin

        #Create HGF agent with binary softmax action
        hgf = premade_hgf("binary_3level", verbose = false)
        action_model = ActionModel(HGFSigmoid(; HGF = hgf))

        #Initialize agent
        agent = init_agent(action_model)

        #Give inputs to the agent
        actions = simulate!(agent, [1, 0, 1])

        #Check that actions are floats
        @test actions isa Vector

        #Check that get_surprise works
        @test get_surprise(agent.model_attributes.submodel) isa Real
    end
end
