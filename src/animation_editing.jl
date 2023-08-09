# this code plots the effect of changing lambda on all the different quantities
# at the moment it doesn't include a comparison for u = 0

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
        method_name = "flux_conc_"
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
    elseif var_choice == 11
        method_name = "normalised_concentration_fluctutation_squared_"
    else
        method_name = "flux_"
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

    if length(states_to_compare_to) > 1
        state_var = "_states"
    elseif length(states_to_compare_to) > 0
        state_var = "_states_" * string(states_to_compare_to[1])
    else
        state_var = ""
    end

    fig_save_name = method_name * xvar * lvar * fvar * state_var * ".mp4"
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
    elseif var_choice == 12
        xvar = x
        yvar = flux_mean[:]
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
        ylabel = L"⟨c'²⟩"
    elseif var_choice == 5
        xlabel = "x"
        ylabel = L"⟨c'²⟩/⟨c⟩²"
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
    elseif var_choice == 10
        ylabel = "FT⟨c'²⟩"
    elseif var_choice == 11
        ylabel = "scaled ⟨c'²⟩/⟨c⟩²"
    else
        ylabel = "⟨uc⟩"
    end
    return ylabel
end

function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

function choose_colours(col_option, approx_num, dat_indx, line_indx)
    # returns the colour selection
    if col_option == 1 #each type of approx is a different_colour
        col_indx = approx_num
        # get the number of coloured lines
        #if no_u
            #num_lines = 1
        #    num_lines = num_lines + 1
        #end
        #if include_no_coupling
        #    num_lines = num_lines + 1
        #if include_inverse
        #    num_lines = num_lines + 1
        #end
        #if include_coupling
        #    num_lines = num_lines + 1
        #end
        colours = specify_colours(5)
    elseif col_option == 2 # each state is a different colour
        col_indx = dat_indx
        colours = specify_colours(length(states_to_compare_to) + 2)
    else
        col_indx = line_indx # each line_var is a different colour
        colours = specify_colours(length(line_variable))
    end
    colour = colours[col_indx]
    return colour
end

function choose_markers(marker_option, approx_num, dat_indx, line_indx)
    # returns the colour selection
    if marker_option == 1 #each type of approx is a different_colour
        col_indx = approx_num
    elseif marker_option == 2 # each state is a different colour
        col_indx = dat_indx
    else
        col_indx = line_indx # each line_var is a different colour
    end
    marker = markers[col_indx]
    return marker
end

function choose_linestyles(l_option, approx_num, dat_indx, line_indx)
    # returns the colour selection
    if l_option == 1 #each type of approx is a different_colour
        col_indx = approx_num
    elseif l_option == 2 # each state is a different colour
        col_indx = dat_indx
    else
        col_indx = line_indx # each line_var is a different colour
    end
    ls = linestyles[col_indx]
    return ls
end

function load_data(data_folder, state, var_choice, mag = 0.7, u_force = 1.0, κ = 0.001)
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
            if data_folder == "data/gpu/kappa_0.001/code_fixes/"
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
                # Concatenate the folder and file name to get the full path
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name c_mean flux_mean c_squared_mean gc #c_cube_mean flux_c_square_mean # cs fs
                    # obtain the variables for plotting
                    if varu != 0
                        xvar, yvar = load_variables_u(var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k, λ, mag)
                        data[:, nindx, mindx] = yvar
                    end
                catch systemerror
                    print("no file named " * data_name)
                end
            else
                data_name =  "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_" * string(state) * ".jld2"
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name cs
                    c_mean = sum(cs)
                    us = u_list(state)
                    flux_mean = zeros(x_length)

                    for i = 1:state
                        flux_mean = flux_mean + us[i]*cs[i]
                    end
                    c_squared_mean = c_mean.^2
                    gc=real(ifft(im*k[:,1].*fft(c_mean)));
                    xvar, yvar = load_variables_u(var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k, λ, mag)
                    if var_choice == 4 && state == 2
                        yvar = (cs[1]-cs[2]).^2
                    end
                    data[:, nindx, mindx] = yvar
                catch systemerror
                    print("no file named " * data_name)

                end
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
    return data
end

function load_approximation(data_folder)
    data2 = zeros(x_length, length(panel_variable), length(line_variable))
    data3 = zeros(x_length, length(panel_variable), length(line_variable))
    fluctuation_labels =fill("", length(panel_variable), length(line_variable))

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
            if data_folder == "data/gpu/kappa_0.001/code_fixes/"
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
                # Concatenate the folder and file name to get the full path
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name c_mean flux_mean c_squared_mean gc #c_cube_mean flux_c_square_mean # cs fs
                    # obtain the variables for plotting
                    if var_choice == 2
                        data2[:, nindx, mindx] = gc[:]                      
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
                            data2[:, nindx, mindx] = c_mean_pred
                        end
                    elseif var_choice == 9
                        data2[:, nindx, mindx] = abs.(real(ifft(im*k[:,1].*fft(flux_c_square_mean))));
                        data3[:, nindx, mindx] = abs.(-λ*c_cube_mean./(1 .+ mag*cos.(x)));
                    elseif var_choice == 4
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
                catch systemerror
                    print("no file named " * data_name)
                end
            end
        end
    end

    return data2, data3, fluctuation_labels
end

function update_data_with_inverse(inverse, c_mean, var_choice)
    if var_choice == 3
        data = 1 ./inverse  #inverse.*noninverse  
    elseif var_choice == 4
        data = (c_mean).^3 .*inverse .- c_mean.^2
    elseif var_choice == 8
        data = (c_mean).^3 .*inverse
    end
    return data

end

function u_list(number_of_states)
    N = number_of_states-1
    u = zeros(number_of_states)
    for m = 0:N
        u[m+1] = 2/sqrt(N)*(m - N/2)
    end
    return u
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
#12: <uc>

markers = ['o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd'] #, "hexagon", "cross", "xcross", "utriangle", "dtriangle"];
linestyles = [:solid, :dash, :dashdot, :dot, :solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]

# choose what to plot
var_choice = 12 # number to choose what to plot 
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels, 4: animation
panel_variable_num = 3 # choose what each panel in the animation will vary with, 1: magnitude, 2: U, 3: lambda
line_variable_num = 2 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 
states_to_compare_to = [2, 3, 10, 20] #, 3, 10]

plot_approx = false # will plot the approximations
shareaxis = true # on the panel will plot all with the same axis
no_u = false #true #false # either plot u = 0 or u non 0
include_inverse = false #true # will plot calculations assuming <1/c>
include_coupling = true # will plot the results when the two are coupled
include_no_coupling = true # will plot when not coupled

# plotting options
col_option = 2  #1: each type of approx is a different_colour, 2: each state is a different colour, 3: each line_var is a different colour
marker_option = 3 # same as above
line_option = 1

if var_choice == 3 || var_choice == 4 || var_choice == 8
    include_inverse = include_inverse
else
    global include_inverse
    include_inverse = false
end



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
    lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
    #lambdas = sort([0.01, 0.1, 0.5, 1.0, 1.2, 1.4, 1.5, 10.0, 100.0])

    # subgroup for each line plot
    line_magnitudes = [0.7] #[0.1, 0.5, 0.7, 0.9]
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
leg_end2 = fill("", length(states_to_compare_to)+1)
leg_end3 = fill("", length(states_to_compare_to)+1)
all_data = zeros(length(states_to_compare_to)+1, x_length, length(panel_variable), length(line_variable))
all_data2 = ones(length(states_to_compare_to)+1, x_length, length(panel_variable), length(line_variable))
all_data3 = ones(length(states_to_compare_to)+1, x_length, length(panel_variable), length(line_variable))
data_folder = "data/gpu/kappa_0.001/code_fixes/"
data = load_data(data_folder, 0, var_choice)
all_data[1, :, :, :] = data
indx = 1
for state in states_to_compare_to
    data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state/"
    data = load_data(data_folder, state, var_choice)

    indx = indx+1
    all_data[indx, :, :, :] = data
    leg_end[indx] = ", N = " * string(state)

    if include_inverse
        data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state_inverse/"
        data2 = load_data(data_folder, state, var_choice)
        current_data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state/"
        c_mean = load_data(current_data_folder, state, 3)
        all_data2[indx, :, :, :] = update_data_with_inverse(data2, c_mean, var_choice)
        leg_end2[indx] = ", with inv N = " * string(state)
    end
    if include_coupling
        data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state_coupled_bigger_t_step/"
        data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state_coupled/"
        data = load_data(data_folder, state, var_choice)
        all_data3[indx, :, :, :] = data
        leg_end3[indx] = ", coupled N = " * string(state)
    end
end

if plot_approx
    data2, data3, fluctuation_labels = load_approximation(data_folder)
end

fig = Figure(resolution = (3024, 1964),)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

# now plot the data
panel_indx = Observable(1)
end_time = length(panel_variable)
ylabel = choose_axis(var_choice)

# choose what to have for colours, linestyles, markers etc.

# option 1:
    # colours = different colour for no_u, different approximations
    # linestyle = different linestyle for each number of state
    # marker = different marker for each line_variable
# option 2
    # colours = different colour for u, different state
    # linestyle = different linestyle for each 

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



    # this plots the full_ensemble
    approx_num = 1
    dat_indx = 1
    line_indx = 1
    colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
    marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
    linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
    fig = scatterlines(x, @lift(all_data[1, :, $panel_indx, 1]), color = colour, marker = marker, linestyle = linestyle, label = leg_start[1] * string(line_variable[1]) * leg_end[1]; line_options...,
        axis = (xlabel = "x", ylabel = ylabel, title = @lift("λ = $(round(panel_variable[$panel_indx], digits = 2))"),))
        for dat_indx = 2:length(states_to_compare_to)+1 
            if include_no_coupling # plots the line for each state
                approx_num = 2
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data[dat_indx, :, $panel_indx, 1]), color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[1]) * leg_end[dat_indx]; line_options...)
            end
            if include_inverse # plots the line for each state
                approx_num = 3
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data2[dat_indx, :, $panel_indx, 1]), color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[1]) * leg_end2[dat_indx]; line_options...)
            end
            if include_coupling # plots the line for each state
                approx_num = 4
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data3[dat_indx, :, $panel_indx, 1]), color = color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[1]) * leg_end3[dat_indx]; line_options...)
            end
            if no_u
                approx_num = 5
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data4[dat_indx, :, $panel_indx, 1]), color = color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[1]) * leg_end4[dat_indx]; line_options...)
            end
        end
    for line_indx = 2:length(line_variable)
        scatterlines!(x, @lift(all_data[1, :, $panel_indx, line_indx]), color = colour, marker = marker, linestyle = linestyle, label = leg_start[1] * string(line_variable[line_indx]) * leg_end[1]; line_options...,)
        for dat_indx = 2:length(states_to_compare_to)+1
            if include_no_coupling # plots the line for each state
                approx_num = 2
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data[dat_indx, :, $panel_indx, line_indx]), color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[line_indx]) * leg_end[dat_indx]; line_options...)
            end
            if include_inverse # plots the line for each state
                approx_num = 3
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data2[dat_indx, :, $panel_indx, line_indx]), color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[line_indx]) * leg_end2[dat_indx]; line_options...)
            end
            if include_coupling # plots the line for each state
                approx_num = 4
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                print(approx_num, linestyle)
                scatterlines!(fig.axis, x,  @lift(all_data3[dat_indx, :, $panel_indx, line_indx]), color = color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[line_indx]) * leg_end3[dat_indx]; line_options...)
            end
            if no_u
                approx_num = 5
                colour = choose_colours(col_option, approx_num, dat_indx, line_indx)
                marker = choose_markers(marker_option, approx_num, dat_indx, line_indx)
                linestyle = choose_linestyles(line_option, approx_num, dat_indx, line_indx)
                scatterlines!(fig.axis, x,  @lift(all_data4[dat_indx, :, $panel_indx, line_indx]), color = color = colour, marker = marker, linestyle = linestyle, label = leg_start * string(line_variable[line_indx]) * leg_end4[dat_indx]; line_options...)
            end
        end
    end
    #if include_inverse
    #    ylims!(nanminimum([nanminimum(all_data), nanminimum(all_data2)]), nanmaximum([nanmaximum(all_data), nanmaximum(all_data2)]))
    #else
        ylims!(nanminimum(all_data), nanmaximum(all_data))
    #end
    axislegend()
end



framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

movie_name =  name_figure(var_choice, panel_variable_num, line_variable_num, line_variable, mag, u_force, λ)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    panel_indx[] = t
end
