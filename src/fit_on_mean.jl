
function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end


magnitudes = [0.7]
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
states_to_plot = [3]

colours = specify_colours(length(lambdas))
linestyles = [:solid, :dash, :dot, :dashdot]
legendsize = 30;
axlabelsize = 30;
axticksize = 10;

line_options = (; linewidth = 3)

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
ax = Axis(fig[1, 1])

for (sindx, state) in enumerate(states_to_plot)
    data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state_coupled/"
    for mag in magnitudes
        for (cindx, λ) in enumerate(lambdas)
            data_name =  "mag_" * string(mag) * "_U_" * string(1.0) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_" * string(state) * ".jld2"
            load_name = joinpath(data_folder, data_name)
            #try
                @load load_name cs mean_theta
                lines!(mean_theta, color = colours[cindx], linestyle = linestyles[sindx], label = "λ = " * string(λ) * ", N = " * string(state) * ", Δ = " * string(mag); line_options...)
            #end
        end
    end
end

ax.ylabel = "⟨c̅⟩"
ax.ylabelsize = axlabelsize
ax.xticklabelsize = axticksize
ax.yticklabelsize = axticksize
legend = axislegend(labelsize = legendsize, nbanks = 1)

fig_name = "mean_over_time_N_" * string(states_to_plot[1]) * ".png" 

save(fig_name, fig)

