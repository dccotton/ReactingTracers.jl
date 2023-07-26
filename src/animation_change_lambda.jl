using GLMakie
using JLD2
using FileIO

using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

function name_figure(var_choice, quantity_to_vary, line_variable_num, line_variable, mag, U, λ)
    if var_choice == 1
        method_name = "FT_"
    elseif var_choice == 2
        method_name = "flux_"
    elseif var_choice == 3
        method_name = "concentration_"
    elseif var_choice == 4
        method_name = "concentration_fluctuation_squared_"
    elseif var_choice == 5
        method_name = "relavitve_concentration_squared_"
    elseif var_choice == 6
        method_name = "flux_gradient_"
    elseif var_choice == 7
        method_name = "concentration_against_prediction"
    else
        method_name = "concentration_squared_"
    end

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

function load_variables_u(ax, var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k)
    if var_choice == 1
        ft_cf=abs.(fft(c_mean))[:]
        xvar = k
        yvar = log10.(ft_cf)[:]
        ylims!(minimum(-16), maximum(5))
        xlims!(minimum(k), maximum(k))
    elseif var_choice == 2
        xvar = gc[:]
        yvar = flux_mean[:]
    elseif var_choice == 3
        xvar = x
        yvar = c_mean[:]
    elseif var_choice == 4
        xvar = x
        yvar = c_squared_mean[:] .- c_mean[:].^2;
    elseif var_choice == 5
        xvar = x
        yvar = (c_squared_mean[:] .- (c_mean[:]).^2)./c_mean[:].^2
    elseif var_choice == 6
        xvar = x
        yvar = real(ifft(im*k[:,1].*fft(flux_mean)))[:];
    elseif var_choice == 7
        xvar = x
        yvar = abs.(c_mean[:] .- (1 .+ mag*cos.(xvar)))./abs.(c_mean[:])
    else
        xvar = x
        yvar = c_squared_mean[:]
    end
    return xvar, yvar
end

function choose_axis(var_choice)
    if var_choice == 1
        xlabel = "k"
        ylabel = "FFT"
    elseif var_choice == 2
        xlabel = "∇c"
        ylabel = "⟨uc⟩"
    elseif var_choice == 3
        xlabel = "x"
        ylabel = "⟨c⟩"
    elseif var_choice == 4
        xlabel = "x"
        ylabel = L"⟨c'⟩^2⟩"
    elseif var_choice == 5
        xlabel = "x"
        ylabel = L"⟨c'^2⟩/⟨c⟩^2"
    elseif var_choice == 6
        xlabel = "x"
        ylabel = "∂⟨uc⟩/∂x"
    elseif var_choice == 7
        xlabel = "x"
        ylabel = "|⟨c⟩ - (1+Δ(x))|/|⟨c⟩|"
    else
        xlabel = "x"
        ylabel = "⟨c^2⟩"
    end
    return ylabel
end


function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

# choose what to plot
var_choice = 2 # number to choose what to plot 1: FFT(<c) and FFT(Δ(x)), 2: ∇c, <uc>, 3: <c>, 4: <c'^2>, 5: <c'^2>/<c>^2>, 6: d<uc>/dx, 7: plot the error in mean estimate as a function of x ((⟨c⟩ - c_0)^2/c_0^2), 8: plot ⟨c^2⟩
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels, 4: animation
panel_variable_num = 3 # choose what each panel will vary with, 1: magnitude, 2: U, 3: lambda
line_variable_num = 1 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 

shareaxis = true # on the panel will plot all with the same axis
no_u = false #true #false # either plot u = 0 or u non 0

varu = 1 # variance of u

# variables to choose to plot (these are chosen but line_variable_num/panel_variable_num will otherwise select more options)
mag = 0.7
u_force = 1.0
κ = 0.001
λ = 1

# options to choose from

if varu != 0
    magnitudes = [0.7] #[0.5, 0.7, 0.9]
    velocities = [1, 3, 6, 13, 25]
    lambdas = [0.5, 1, 1.5] #
    lambdas = [0.1, 1.0, 10.0, 100.0] #[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5] #[0.01, 0.1, 1, 5] #[0.01, 0.1, 0.5, 1, 1.5, 5, 10]
    lambdas = sort([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 100.0])#lambdas = [3.0, 5.0, 6.0, 10.0]
    #lambdas = [0.1, 1.0, 2.0, 10.0]

    # subgroup for each line plot
    line_magnitudes = [0.7] #[0.5, 0.7, 0.9]
    line_velocity = [1]
    line_lambdas = [3, 5, 6, 10.0]
    line_kappas = [0.01, 0.001]
end

x_length = 1024
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

line_options = (; linewidth = 3)

# first load in the data
data = zeros(x_length, length(panel_variable), length(line_variable))
data2 = zeros(x_length, length(panel_variable), length(line_variable))

# plot nabla c against <uc> for various choices of delta
fig = Figure(resolution = (3024, 1964),)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

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
        if varu != 0
            data_folder = "data/new_scaling"
            if κ != 0.001
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            else
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * ".jld2"
            end
            # Concatenate the folder and file name to get the full path
            load_name = joinpath(data_folder, data_name)
            #load_name = data_name
            try
                @load load_name c_mean flux_mean c_squared_mean gc cs fs

                        # obtain the variables for plotting
        if varu != 0
            xvar, yvar = load_variables_u(ax, var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k)
            data[:, nindx, mindx] = yvar

            if var_choice == 2
                data2[:, nindx, mindx] = gc[:]
            end
        end
            catch systemerror
                print("no file named " * data_name)
            end
        end

        mindx = mindx + 1
    end
    nindx = nindx + 1
end

# now plot the data
panel_indx = Observable(1)
choice_line = @lift(data[:, $panel_indx, 1])
if var_choice == 2
    choice_line2 = @lift(data2[:, $panel_indx, 1])
end
end_time = length(panel_variable)
ylabel = choose_axis(var_choice)

if var_choice == 2
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x, choice_line2, color = :red, linewidth = 4)
    ylims!(minimum(data), maximum(data))

else
    fig = lines(x, choice_line, color = :blue, linewidth = 4; line_options...,
    axis = (xlabel = "x", ylabel = ylabel, title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    ylims!(minimum(data), maximum(data))
end

#fig = Figure(resolution = (3024, 1964),)
#ax = choose_axis(fig, var_choice)
#ax.xlabelsize = axlabelsize
#ax.ylabelsize = axlabelsize
#ax.titlesize = axtitlesize
#lines!(x, choice_line, color = :blue, linewidth = 4, label = "c"; line_options...,)
#setfield!(ax, :title, @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"))
#ylims!(minimum(data), maximum(data))

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

movie_name =  name_figure(var_choice, panel_variable_num, line_variable_num, line_variable, mag, u_force, λ)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    panel_indx[] = t
end