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
    if var_choice == 1
        method_name = "FT_"
    elseif var_choice == 2
        method_name = "flux_"
    elseif var_choice == 3
        method_name = "concentration_"
    elseif var_choice == 4
        method_name = "concentration_fluctuation_squared_"
    elseif var_choice == 5
        method_name = "relative_concentration_squared_"
    elseif var_choice == 6
        method_name = "flux_gradient_"
    elseif var_choice == 7
        method_name = "concentration_against_prediction"
    elseif var_choice ==8
        method_name = "concentration_squared_"
    elseif var_choice ==9
        method_name = "concentration_squared_terms_dominance"
    else
        method_name = "normalised_concentration_fluctutation_squared_"
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

function load_variables_u(var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k, λ, mag)
    if var_choice == 1
        ft_cf=abs.(fft(c_mean))[:]
        xvar = k
        yvar = log10.(ft_cf)[:]
        #xlims!(minimum(k), maximum(k))
    elseif var_choice == 2
        xvar = gc[:]
        yvar = flux_mean[:]
    elseif var_choice == 3
        xvar = x
        yvar = c_mean[:]
    elseif var_choice == 4 || var_choice == 11
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
    elseif var_choice == 8
        xvar = x
        yvar = c_squared_mean[:]
    elseif var_choice == 9
        xvar = x
        yvar = abs.(λ*c_squared_mean[:].* (1 .- 2*c_mean[:]./(1 .+ mag*cos.(x))))
    elseif var_choice == 10
        ft_cf=abs.(fft(c_squared_mean .- c_mean.^2))[:]
        #ft_cf=abs.(fft(c_squared_mean))[:]
        xvar = k
        yvar = log10.(ft_cf)[:]
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
        ylabel = L"⟨c'^2⟩"
    elseif var_choice == 5
        xlabel = "x"
        ylabel = L"⟨c'^2⟩/⟨c⟩^2"
    elseif var_choice == 6
        xlabel = "x"
        ylabel = "∂⟨uc⟩/∂x"
    elseif var_choice == 7
        xlabel = "x"
        ylabel = "|⟨c⟩ - (1+Δ(x))|/|⟨c⟩|"
    elseif var_choice ==8
        xlabel = "x"
        ylabel = "⟨c^2⟩"
    elseif var_choice == 9
        ylabel = L"λ⟨c_1^2⟩(1-2c̅/(1+Δ(x)))"
    else
        ylabel = "scaled ⟨c'^2⟩/⟨c⟩^2"
    end
    return ylabel
end


function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
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

# choose what to plot
var_choice = 2 # number to choose what to plot 
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels, 4: animation
panel_variable_num = 3 # choose what each panel in the animation will vary with, 1: magnitude, 2: U, 3: lambda
line_variable_num = 2 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 

compare_to_two_state = true
compare_to_three_state = true 
plot_approx = true #false
shareaxis = true # on the panel will plot all with the same axis
no_u = false #true #false # either plot u = 0 or u non 0

varu = 1 # variance of u

# variables to choose to plot (these are chosen but line_variable_num/panel_variable_num will otherwise select more options)
mag = 0.7
u_force = 1.0
κ = 0.001
λ = 1

data_folder = "data/gpu/kappa_0.001/code_fixes/"

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

line_options = (; linewidth = 3)

# matrices to store the data
data = zeros(x_length, length(panel_variable), length(line_variable))
data2 = zeros(x_length, length(panel_variable), length(line_variable))
data3 = zeros(x_length, length(panel_variable), length(line_variable))
fluctuation_labels =fill("", length(panel_variable), length(line_variable))

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
            data_folder = "data/gpu/kappa_0.001/code_fixes/"
            if κ != 0.0001
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            else
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * ".jld2"
            end
            # Concatenate the folder and file name to get the full path
            load_name = joinpath(data_folder, data_name)
            #try
                @load load_name c_mean flux_mean c_squared_mean gc #c_cube_mean flux_c_square_mean # cs fs
                # obtain the variables for plotting
                if varu != 0
                    xvar, yvar = load_variables_u(var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k, λ, mag)
                    data[:, nindx, mindx] = yvar

                    if var_choice == 2
                        if compare_to_two_state
                            data_folder = "data/gpu/kappa_0.001/two_state/"
                            data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_2.jld2"
                            load_name = joinpath(data_folder, data_name)
                            @load load_name cs
                            if cs[50, 1, end] == 0
                                data2[:, nindx, mindx] = -cs[:, 1, end-1] + cs[:, 2, end-1]
                            else
                                data2[:, nindx, mindx] = -cs[:, 1, end] + cs[:, 2, end]
                            end
                            if compare_to_three_state
                                data_folder = "data/gpu/kappa_0.001/three_state/"
                                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_3.jld2"
                                load_name = joinpath(data_folder, data_name)
                                @load load_name cs
                                if cs[50, 1, end] == 0
                                    data3[:, nindx, mindx] = -sqrt(2)*cs[:, 1, end-1] + sqrt(2)*cs[:, 3, end-1]
                                else
                                    data3[:, nindx, mindx] = -sqrt(2)*cs[:, 1, end] + sqrt(2)*cs[:, 3, end]
                                end
                            end
                        else
                            data2[:, nindx, mindx] = gc[:]
                        end                        
                    elseif var_choice == 6 || var_choice == 3
                        c_0_val = c0_dict[string(mag)]
                        c_1_val = c1_dict[string(mag)]
                        c_0_mean = (0.5*tanh(c_0_val*log10(λ))+0.5) * (1 - (1-mag^2)^0.5) + (1-mag^2)^0.5
                        c_1_mean = (0.5*tanh(c_0_val*log10(λ))+0.5)*0.7
                        c_mean_pred = c_0_mean .+ c_1_mean*cos.(x)                        
                        if var_choice == 6
                            data2[:, nindx, mindx] = λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(x)))[:]
                            data3[:, nindx, mindx] = λ*c_mean_pred.*(1 .- c_mean_pred./(1 .+ mag*cos.(x)))[:]
                        elseif var_choice == 3
                            if compare_to_two_state
                                data_folder = "data/gpu/kappa_0.001/two_state/"
                                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_2.jld2"
                                load_name = joinpath(data_folder, data_name)
                                @load load_name cs
                                if cs[50, 1, end] == 0
                                    data2[:, nindx, mindx] = cs[:, 1, end-1] + cs[:, 2, end-1]
                                else
                                    data2[:, nindx, mindx] = cs[:, 1, end] + cs[:, 2, end]
                                end
                                if compare_to_three_state
                                    try
                                    data_folder = "data/gpu/kappa_0.001/three_state/"
                                    data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_3.jld2"
                                    load_name = joinpath(data_folder, data_name)
                                    @load load_name cs
                                    if cs[50, 1, end] == 0
                                        data3[:, nindx, mindx] = cs[:, 1, end-1] + cs[:, 2, end-1] + cs[:, 3, end-1]
                                    else
                                        data3[:, nindx, mindx] = (cs[:, 1, end] + cs[:, 2, end] + cs[:, 3, end])
                                    end
                                catch SystemError
                                end
                                end
                            else
                                data2[:, nindx, mindx] = c_mean_pred
                            end

                        end
                    elseif var_choice == 9
                        data2[:, nindx, mindx] = abs.(real(ifft(im*k[:,1].*fft(flux_c_square_mean))));
                        data3[:, nindx, mindx] = abs.(-λ*c_cube_mean./(1 .+ mag*cos.(x)));
                    elseif var_choice == 4
                        if compare_to_two_state
                            data_folder = "data/gpu/kappa_0.001/two_state/"
                            data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_2.jld2"
                            load_name = joinpath(data_folder, data_name)
                            @load load_name cs
                            if cs[50, 1, end] == 0
                                data2[:, nindx, mindx] = (cs[:, 1, end-1] - cs[:, 2, end-1]).^2
                            else
                                data2[:, nindx, mindx] = (cs[:, 1, end] - cs[:, 2, end]).^2
                            end
                        else
                            ft_cf=(fft(c_squared_mean .- c_mean.^2))[:]
                            fluctuation_label = string(round(real(ft_cf[1])/x_length, sigdigits = 2))
                            max_k = 2
                            approx_func = zeros(x_length)
                            for indx = 1:max_k
                                if indx % 2 == 0
                                    a = real(ft_cf[indx+1])*2/x_length #cosine component
                                    b = -imag(ft_cf[indx+1])*2/x_length #sin component
                                else
                                    a = -real(ft_cf[indx+1])*2/x_length
                                    b = imag(ft_cf[indx+1])*2/x_length
                                end                            
                                approx_func = approx_func .+ a*cos.(indx*x) .+ b*sin.(indx*x)
                                fluctuation_label = fluctuation_label * "+ " * string(round(a, sigdigits = 2)) * "cos(" * string(indx) *"x) + " * string(round(b, sigdigits = 2)) * "sin(" * string(indx) *"x)"
                            end
                    
                            fluctuation_labels[nindx, mindx] = fluctuation_label
                            approx_func = approx_func .+ real(ft_cf[1])/x_length

                            
                            #ft_cf[4:1022] = zeros(1022-4+1)
                            data2[:, nindx, mindx] = approx_func #real((ifft(ft_cf)));
                        end
                    end
                end
            #catch systemerror
            #    print("no file named " * data_name)
            #end
        end

        mindx = mindx + 1
    end
    nindx = nindx + 1
end

if var_choice == 11
    for m_indx = 1:length(panel_variable)
        for n_indx = 1:length(line_variable) 
            data[:, m_indx, n_indx] = data[:, m_indx, n_indx]/maximum(data[:, m_indx, n_indx])
        end
    end
end

# now plot the data
panel_indx = Observable(1)
end_time = length(panel_variable)
ylabel = choose_axis(var_choice)

# colours
if length(line_variable) == 1
    colours = specify_colours(2)
else
    colours = specify_colours(length(line_variable))
end

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

if var_choice == 3
    if compare_to_two_state
        leg_end = ", N = 2" 
    else
        leg_end = ", c̅ ≈ 1/2*[tanh(a*log10(λ)+1]*(1-√(1-|Δ|^2)) + √(1-|Δ|^2) + Δ/2*tanh(b*log10(λ))" 
    end
    if compare_to_three_state
        leg_end_2 = ", N = 3"
    end
elseif var_choice == 2
    if compare_to_two_state
        leg_end = ", N = 2"
    else
    leg_end = ", ∇c"
    end
    if compare_to_three_state
        leg_end_2 = ", N = 3"
    end
elseif var_choice ==4
    if compare_to_two_state
        leg_end = ", N = 2" 
    else
        leg_end = ", fourier approx"
    end
elseif var_choice == 6
    leg_end_2 = "λc̅(1-c̅/(1+Δ(x)))"
    leg_end_3 = "λc̅_{pred}(1-c̅_{pred}/(1+Δ(x)))"
end

if var_choice == 2 && plot_approx == true || var_choice == 3 && plot_approx == true
    fig = lines(
        x, @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x,  @lift(data2[:, $panel_indx, 1]), color = colours[1], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[1]) * leg_end)
    if compare_to_three_state
        lines!(fig.axis, x,  @lift(data3[:, $panel_indx, 1]), color = colours[1], linewidth = 4, linestyle = :dot, label = leg_start * string(line_variable[1]) * leg_end_2)
    end
    for indx = 2:length(line_variable)
        lines!(fig.axis, x,  @lift(data[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, label = leg_start * string(line_variable[indx]))
        lines!(fig.axis, x,  @lift(data2[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[indx]) * leg_end)
        if compare_to_three_state
            lines!(fig.axis, x,  @lift(data3[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, linestyle = :dot, label = leg_start * string(line_variable[indx]) * leg_end_2)
        end
    end
    if compare_to_three_state
        ylims!(nanminimum([nanminimum(data), nanminimum(data2), nanminimum(data3)]), nanmaximum([(nanmaximum(data)), nanmaximum(data2), nanminimum(data3)])) 
    else
       ylims!(nanminimum([nanminimum(data), nanminimum(data2)]), nanmaximum([(nanmaximum(data)), nanmaximum(data2)])) 
    end
    axislegend()
elseif var_choice == 6 && plot_approx == true
    fig = lines(
        x, @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
            lines!(fig.axis, x,  @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4, label = leg_start * string(line_variable[1]))
            lines!(fig.axis, x,  @lift(data2[:, $panel_indx, 1]), color = colours[1], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[1]) * leg_end_2)
            lines!(fig.axis, x,  @lift(data3[:, $panel_indx, 1]), color = colours[1], linewidth = 4, linestyle = :dot, label = leg_start * string(line_variable[1]) * leg_end_3)

            for indx = 2:length(line_variable)
                lines!(fig.axis, x,  @lift(data[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, label = leg_start * string(line_variable[indx]))
                lines!(fig.axis, x,  @lift(data2[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[indx]) * leg_end_2)
                lines!(fig.axis, x,  @lift(data3[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, linestyle = :dot, label = leg_start * string(line_variable[indx]) * leg_end_3)
            end
    ylims!(nanminimum(data), nanmaximum(data)) 
    axislegend()
elseif var_choice == 4 && plot_approx == true
    fig = lines(
        x, @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4, label = leg_start * string(line_variable[1]);
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("λ = " * string((round(panel_variable[$panel_indx], digits = 2))) * string(fluctuation_labels[$panel_indx, :])),))
    lines!(fig.axis, x, @lift(data2[:, $panel_indx, 1]), color = colours[1], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[1]) * leg_end)
    for indx = 2:length(line_variable)
        lines!(fig.axis, x,  @lift(data[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, label = leg_start * string(line_variable[indx]))
        lines!(fig.axis, x,  @lift(data2[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, linestyle = :dash, label = leg_start * string(line_variable[indx]) * leg_end)
    end
    ylims!(nanminimum([nanminimum(data), nanminimum(data2)]), nanmaximum([(nanmaximum(data)), nanmaximum(data2)])) 
    axislegend()
elseif var_choice == 1 || var_choice == 10
    fig = scatterlines(k, @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4, label = leg_start * string(line_variable[1]); line_options...,
    axis = (xlabel = "k", ylabel = ylabel, title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    for indx = 2:length(line_variable)
        scatterlines!(fig.axis, x,  @lift(data[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, label = leg_start * string(line_variable[indx]))
    end
    ylims!(minimum(-10), maximum(5))
    #xlims!(minimum(k), maximum(k))
    xlims!(-30, 30)
    axislegend()
elseif var_choice == 9 # need to sort
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (yscale = log10,
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x, choice_line3, color = :red, linewidth = 4, linestyle = :dash, label = L"λ⟨c'^3⟩/(1+Δ(x)))")
    lines!(fig.axis, x, choice_line2, color = :green, linewidth = 4, linestyle = :dot, label = L"∂⟨uc'^2⟩/∂x")
    #ylims!(minimum([minimum(data), minimum(data2), minimum(data3)]), maximum([maximum(data), maximum(data2), maximum(data3)]))
    ylims!(10^-6, (maximum([maximum(data), maximum(data2), maximum(data3)])))
    #ylims!(0, @lift(maximum(data[:, $panel_indx, 1])))
    axislegend() 
else
    fig = lines(x, @lift(data[:, $panel_indx, 1]), color = colours[1], linewidth = 4, label = leg_start * string(line_variable[1]); line_options...,
    axis = (xlabel = "x", ylabel = ylabel, title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
    for indx = 2:length(line_variable)
        lines!(fig.axis, x,  @lift(data[:, $panel_indx, indx]), color = colours[indx], linewidth = 4, label = leg_start * string(line_variable[indx]))
    end
    ylims!(nanminimum(data), nanmaximum(data))
    axislegend()
end

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

movie_name =  name_figure(var_choice, panel_variable_num, line_variable_num, line_variable, mag, u_force, λ)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    panel_indx[] = t
end