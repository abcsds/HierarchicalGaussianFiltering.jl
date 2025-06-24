using Test

using HierarchicalGaussianFiltering
using ActionModels
using ActionModels: DataFrames

using StatsPlots

@testset "Model fitting" begin

    data = DataFrames.DataFrame(
        observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
        actions = vcat(
            [0, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0.5, 0.8, 1, 1.5, 1.8],
            [0, 2, 0.5, 4, 5, 3],
            [0, 0.1, 0.15, 0.2, 0.25, 0.3],
            [0, 0.2, 0.4, 0.7, 1.0, 1.1],
            [0, 2, 0.5, 4, 5, 3],
        ),
        age = vcat(
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
        ),
        id = vcat(
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
        ),
        treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
    )

    #Add a second set of actions and observations
    data.actions_2 = data.actions
    data.observations_2 = data.observations

    #Add multivariate actions
    data.actions_mv = Vector.(eachrow([data.actions data.actions_2]))

    #Define observation and action cols
    observation_cols = [:observations]
    action_cols = [:actions]
    session_cols = [:id, :treatment]



    #Inference parameters
    n_samples = 200
    n_chains = 2

    ad_type = AutoForwardDiff()

    @testset "test full inference procedure" begin
        
        #Create HGF
        hgf = premade_hgf("continuous_2level", verbose = false)

        #Create action model
        action_model = ActionModel(HGFGaussian(; HGF = hgf))
        
        #Set priors
        prior = (
            action_noise = LogNormal(),
            x_volatility = Normal(-6, 3),
        )

        #Create model
        model = create_model(
            action_model,
            prior,
            data,
            observation_cols = observation_cols,
            action_cols = action_cols,
            session_cols = session_cols,
        )

        #Sample Posterior
        posterior_chains = sample_posterior!(
            model,
            ad_type = ad_type,
            n_samples = n_samples,
            n_chains = n_chains,
        )

        plot(posterior_chains)

        posterior_parameters = get_session_parameters!(model, :posterior)
        posterior_parameters_df = summarize(posterior_parameters)

        posterior_trajectories = get_state_trajectories!(model, [:x_value_prediction_error], :posterior)
        posterior_trajectories_df = summarize(posterior_trajectories)

        prior_chains =
            sample_prior!(model, n_samples = n_samples, n_chains = n_chains)
        plot(prior_chains)
        prior_parameters = get_session_parameters!(model, :prior)
        summarize(prior_parameters)
        prior_trajectories = get_state_trajectories!(model, [:x_value_prediction_error], :prior)
        summarize(prior_trajectories)
    end

    @testset "Continuous 2level Gaussian" begin

        #Define inputs and responses
        observations = [1.0, 1.0, 1.0, 1.0, 1.0]
        actions = [1.0, 1.0, 1.0, 1.0, 1.0]

        #Create HGF
        hgf = premade_hgf("continuous_2level", verbose = false)

        #Create action model
        action_model = ActionModel(HGFGaussian(; HGF = hgf))
        
        #Set priors
        prior = (
            action_noise = LogNormal(),
            x_volatility = Normal(-6, 3),
        )
        
        #Create model
        model = create_model(
            action_model,
            prior,
            observations,
            actions;
        )

        #Sample Posterior
        posterior_chains = sample_posterior!(
            model,
            ad_type = ad_type,
            n_samples = n_samples,
            n_chains = n_chains,
        )
    end


    @testset "Binary 3level Sigmoid" begin

        #Set inputs and responses 
        test_input = [1, 0, 0, 1, 1]
        test_responses = [1, 0, 1, 1, 0]

        #Create HGF
        hgf = premade_hgf("binary_3level", verbose = false)

        action_model = ActionModel(HGFSigmoid(; HGF = hgf))

        prior = (
            action_noise = LogNormal(),
            xprob_volatility = truncated(Normal(-6, 1), upper = -3),
        )

        model = create_model(
            action_model,
            prior,
            test_input,
            test_responses;
            check_parameter_rejections = true,
        )

        posterior_chains = sample_posterior!(
            model,
            ad_type = ad_type,
            n_samples = n_samples,
            n_chains = n_chains,
        )
    end

    @testset "Binary 3level Softmax" begin

        #Set inputs and responses 
        test_input = [1, 0, 0, 1, 1]
        test_responses = [1, 0, 1, 1, 0]

        #Create HGF
        hgf = premade_hgf("binary_3level", verbose = false)

        action_model = ActionModel(HGFSoftmax(; HGF = hgf))

        prior = (
            action_noise = LogNormal(),
            xprob_volatility = truncated(Normal(-9, 1), upper = -3),
        )

        model = create_model(
            action_model,
            prior,
            test_input,
            test_responses;
            check_parameter_rejections = true,
        )

        posterior_chains = sample_posterior!(
            model,
            ad_type = ad_type,
            n_samples = n_samples,
            n_chains = n_chains,
        )
    end

    @testset "Categorical 3level Predict" begin

        #Set inputs and responses 
        test_input = [1, 0, 0, 1, 1]
        test_responses = [1, 0, 1, 1, 0]

        #Create HGF
        hgf = premade_hgf("categorical_3level", verbose = false)
        action_model = ActionModel(HGFPredictCategory(; HGF = hgf))

        prior = (
            action_noise = LogNormal(),
            xprob_volatility = truncated(Normal(-9, 1), upper = -3),
        )

        model = create_model(
            action_model,
            prior,
            test_input,
            test_responses;
            check_parameter_rejections = true,
        )

        posterior_chains = sample_posterior!(
            model,
            ad_type = ad_type,
            n_samples = n_samples,
            n_chains = n_chains,
        )
    end
end
