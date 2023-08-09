# this code plots the effect of changing lambda on all the different quantities

using GLMakie
using JLD2
using FileIO
using NaNStatistics
using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

function name_figure(var_choice, quantity_to_vary, line_variable_num, line_variable, mag, U, λ)
    method_name = "plot_c_inv_"

    if quantity_to_vary == 1
        xvar = "panel_Δ_"
    elseif quantity_to_vary == 2
        xvar = "panel_U_"
    else
        xvar = "panel_λ_"
    end
    
    if line_variable_num == 1
        if length(line_variable) == 1
            lvar = "line_Δ_" * string(line_variable[1]) * "_"
        else
            lvar = "line_Δ_"
        end
    elseif line_variable_num == 2
        if length(line_variable) == 1
            lvar = "line_U_" * string(line_variable[1]) * "_"
        else
            lvar = "line_U_"   
        end    
    else
        if length(line_variable) == 1
            lvar = "line_λ_" * string(line_variable[1]) * "_"
        else
            lvar = "line_λ_"   
        end   
    end

    if quantity_to_vary!= 1 && line_variable_num != 1
        fvar = "Δ_" * string(mag)
    elseif quantity_to_vary!= 2 && line_variable_num != 2
        fvar = "U_" * string(U)
    else
        fvar = "lambda_" * string(λ)
    end

    fig_save_name = method_name * xvar * lvar * fvar * ".mp4"
    return fig_save_name
end


function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

function load_inverse_data(data_folder, state, mag = 0.7, u_force = 1.0)
    data = zeros(x_length, length(panel_variable), length(line_variable))
    # load in the data
    nindx = 1
    for pvar in ProgressBar(panel_variable)

        # decide what variable we are obtaining
        if panel_variable_num == 1
            mag = pvar
        elseif panel_variable_num == 2
            u_force = pvar
        else
            λ = pvar
        end
        mindx = 1
        for lvar in line_variable

            # variable for each line
            if line_variable_num == 1
                mag = lvar
            elseif line_variable_num == 2
                u_force = lvar
            elseif line_variable_num == 3
                λ = lvar
            else
                κ = lvar
            end
            
            # load in the data
                data_name =  "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(0.001) * "_N_" * string(state) * "_inv.jld2"
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name cs
                    if cs[1, 1, end] == 0
                        end_indx = size(cs)[3]-1
                    else
                        end_indx = size(cs)[3]
                    end
                    c_mean = sum(cs[:, :, end_indx], dims = 2)
                    data[:, nindx, mindx] = c_mean
                catch systemerror
                    print("no file named " * data_name)
                end

            mindx = mindx + 1
        end
        nindx = nindx + 1
    end
    return data
end

# var choice options
#1: FFT(<c)
#2: ∇c, <uc>
#3: <c>
#4: <c'^2>
#5: <c'^2>/<c>^2>
#6: d<uc>/dx
#7: plot the error in mean estimate as a function of x ((⟨c⟩ - c_0)^2/c_0^2)
#8: plot ⟨c^2⟩
#9: plot the magnitude of the terms in the equation for d<c'2>/dt (this needs to be fixed)
#10: FFT(⟨c'^2⟩)
#11: scaled <c'^2>/<c>^2

markers = ['o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd'] #, "hexagon", "cross", "xcross", "utriangle", "dtriangle"];
linestyles = [:solid, :dash, :dot, :dashdot]

# choose what to plot
var_choice = 8 # number to choose what to plot 
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels, 4: animation
panel_variable_num = 3 # choose what each panel in the animation will vary with, 1: magnitude, 2: U, 3: lambda
line_variable_num = 2 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 
states_to_compare_to = [2] #, 3, 10]

plot_approx = false # will plot the approximations
shareaxis = true # on the panel will plot all with the same axis
no_u = false #true #false # either plot u = 0 or u non 0
include_inverse = false

varu = 1 # variance of u

# variables to choose to plot (these are chosen but line_variable_num/panel_variable_num will otherwise select more options)
mag = 0.7
u_force = 1.0
κ = 0.001
λ = 1

# c̅ fitted values
c0_dict = Dict("0.1" => 1.5, "0.5" => 1.5169, "0.7" => 1.5260, "0.9" => 1.5446)
c1_dict = Dict("0.1" => 1.5, "0.5" => 1.5261, "0.7" => 1.5466, "0.9" => 1.5909)

# options to choose from

if varu != 0
    magnitudes = [0.7] #[0.1, 0.5, 0.7, 0.9]
    velocities = [1.0]
    lambdas = [0.5, 1, 1.5] #
    lambdas = [0.1, 1.0, 10.0, 100.0]
    lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
    #lambdas = sort([0.01, 0.1, 0.5, 1.0, 1.2, 1.4, 1.5, 10.0, 100.0])

    # subgroup for each line plot
    line_magnitudes = [0.1, 0.5, 0.7, 0.9]
    line_velocity = [1.0]
    line_lambdas = [3, 5, 6, 10.0]
    line_kappas = [0.01, 0.001]
end

x_length = 512 #1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

### plot the data
if panel_variable_num == 1
    panel_variable = magnitudes
elseif panel_variable_num == 2
    panel_variable = velocities
else
    panel_variable = lambdas
end

if line_variable_num == 1
    line_variable = line_magnitudes
elseif line_variable_num == 2
    line_variable = line_velocity
elseif line_variable_num == 3
    line_variable = line_lambdas
else
    line_variable = line_kappas
end

# sizes
legendsize = 30;
axlabelsize = 30;
axtitlesize = 40;

line_options = (; linewidth = 3, markersize = 1)

# load the data

# matrix to store all the data
leg_end = fill("", length(states_to_compare_to)+1)
all_data = zeros(length(states_to_compare_to), x_length, length(panel_variable), length(line_variable))
indx = 0
for state in states_to_compare_to
    indx = indx+1
    leg_end[indx] = ", N = " * string(state)
    data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state_inverse/"
    all_data[indx, :, :, :] = load_inverse_data(data_folder, state)
end

if plot_approx
    data2, data3, fluctuation_labels = load_approximation(data_folder)
end

fig = Figure(resolution = (3024, 1964),)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

# now plot the data
panel_indx = Observable(1)
end_time = length(panel_variable)
ylabel = "⟨c⁻¹⟩" 

# colours
if length(line_variable) == 1
    colours = specify_colours(2)
else
    colours = specify_colours(length(line_variable))
end
colours = specify_colours(4)

# legend name
if line_variable_num == 1 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 
    leg_start = "Δ = "
elseif line_variable_num == 2 
    leg_start = "U = "
elseif line_variable_num == 3
    leg_start = "λ = "
else
    leg_start = "κ = "
end

if plot_approx
else
    fig = scatterlines(x, @lift(all_data[1, :, $panel_indx, 1]), linestyle = linestyles[1], color = colours[1], marker = markers[1], label = leg_start[1] * string(line_variable[1]) * leg_end[1]; line_options...,
    axis = (xlabel = "x", ylabel = ylabel, title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    for dat_indx = 2:length(states_to_compare_to)
        scatterlines!(fig.axis, x,  @lift(all_data[dat_indx, :, $panel_indx, 1]), color = colours[dat_indx], linestyle = linestyles[dat_indx], marker = markers[dat_indx], label = leg_start * string(line_variable[1]) * leg_end[dat_indx]; line_options...)
    end
    for indx = 2:length(line_variable)
        for dat_indx = 1:length(states_to_compare_to)+1
            scatterlines!(fig.axis, x,  @lift(all_data[dat_indx, :, $panel_indx, indx]), color = colours[indx], marker = markers[dat_indx], label = leg_start * string(line_variable[indx]) * leg_end[dat_indx]; line_options...)
        end
    end
    ylims!(nanminimum(all_data), nanmaximum(all_data))
    axislegend()
end


framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

movie_name =  name_figure(var_choice, panel_variable_num, line_variable_num, line_variable, mag, u_force, λ)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    panel_indx[] = t
end

