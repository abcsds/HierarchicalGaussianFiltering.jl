using HierarchicalGaussianFiltering, StatsPlots

nodes = [
    ContinuousInput(name = "u"),
    ContinuousState(name = "x1"),
    ContinuousState(name = "x2"),
]

sine_transform = NonlinearTransform(
    function (x, parameters::Dict)
        sin(x)
    end, #base function
    function (x, parameters::Dict)
        cos(x)
    end, #first derivative
    function (x, parameters::Dict)
        -sin(x)
    end, #second derivative
    Dict(), #no parameters
)


edges_linear = Dict(("u", "x1") => ObservationCoupling(), ("x1", "x2") => DriftCoupling())

edges_nonlinear = Dict(
    ("u", "x1") => ObservationCoupling(),
    ("x1", "x2") => DriftCoupling(1, sine_transform),
)



hgf_linear = init_hgf(nodes = nodes, edges = edges_linear, verbose = false)
hgf_nonlinear = init_hgf(nodes = nodes, edges = edges_nonlinear, verbose = false)


parameters = Dict(
    ("x1", "autoconnection_strength") => 0,
    ("x1", "volatility") => -4,
    ("x2", "volatility") => -4,
    ("u", "input_noise") => log(0.25),
)


set_parameters!(hgf_linear, parameters)

set_parameters!(hgf_nonlinear, parameters)


sample_rate = 20

inputs = sin.(collect(0:(1/sample_rate):35))

inputs = rand(Normal(0, 0.25), length(inputs)) + inputs
plot(inputs)



reset!(hgf_linear)
multiple_inputs!(hgf_linear, inputs)
reset!(hgf_nonlinear)
multiple_inputs!(hgf_nonlinear, inputs)


plot_settings = (; label = "", title = "")


plot(
    plot(hgf_linear, "u"; plot_settings..., title = "inputs"),
    plot(hgf_linear, ("u", "prediction"); plot_settings..., title = "u prediction"),
    plot(hgf_linear, ("x1"); plot_settings..., title = "x1 posterior"),
    plot(hgf_linear, ("x2"); plot_settings..., title = "x2 posterior"),
)


plot(
    plot(hgf_nonlinear, "u"; plot_settings..., title = "inputs"),
    plot(hgf_nonlinear, ("u", "prediction"); plot_settings..., title = "u prediction"),
    plot(hgf_nonlinear, ("x1"); plot_settings..., title = "x1 posterior"),
    plot(hgf_nonlinear, ("x2"); plot_settings..., title = "x2 posterior"),
)


μ₁_linear = hgf_linear.all_nodes["x1"].history.posterior_mean

#Band mu1 between -1 and 1 (the noise makes it occasionally jump over)
μ₁_linear = min.(μ₁_linear, 1.0)
μ₁_linear = max.(μ₁_linear, -1.0)

μ₂_linear = hgf_linear.all_nodes["x2"].history.posterior_mean



μ₁_nonlinear = hgf_nonlinear.all_nodes["x1"].history.posterior_mean

#Band mu1 between -1 and 1 (the noise makes it occasionally jump over)
μ₁_nonlinear = min.(μ₁_nonlinear, 1.0)
μ₁_nonlinear = max.(μ₁_nonlinear, -1.0)

μ₂_nonlinear = hgf_nonlinear.all_nodes["x2"].history.posterior_mean



#Plot sine transformed x2 against x1 - should be equal
plot(
    plot(sin.(μ₂_linear) - μ₁_linear, title = "linear"),
    plot(sin.(μ₂_nonlinear) - μ₁_nonlinear, title = "nonlinear"),
)


#Plot asine transformed x1 against x2 - should be equal
plot(
    plot(asin.(μ₁_linear) - μ₂_linear, title = "linear"),
    plot(asin.(μ₁_nonlinear) - μ₂_nonlinear, title = "nonlinear"),
)
#The linear fares worse


linear_plt = plot(asin.(μ₁_linear), label = "μ₁ asin")
plot!(μ₂_linear, label = "μ₂", title = "linear")

nonlinear_plt = plot(asin.(μ₁_nonlinear), label = "μ₁ asin")
plot!(μ₂_nonlinear, label = "μ₂", title = "nonlinear")

plot(linear_plt, nonlinear_plt)
