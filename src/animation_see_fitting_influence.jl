using GLMakie
using JLD2
using FileIO

using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

function name_figure(var_choice, mag, U, λ, averaging)
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
    else
        method_name = "concentration_squared_terms_dominance"
    end

    if averaging == 1
        av_name = ""
    elseif averaging == 2
        av_name = "_diff_100s"
    else
        av_name = "_from_t_" * string(load_times[indx_to_average_from])
    end

    fig_save_name = method_name * "Δ_" * string(mag) * "_U_" * string(U) * "_λ_" * string(λ) * av_name * ".mp4"
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
    elseif var_choice == 8
        xvar = x
        yvar = c_squared_mean[:]
    elseif var_choice == 9
        xvar = x
        yvar = abs.(λ*c_squared_mean[:].* (1 .- 2*c_mean[:]./(1 .+ mag*cos.(x))))
    else
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
    else
        ylabel = L"λ⟨c_1^2⟩(1-2c̅/(1+Δ(x)))"
    end
    return ylabel
end


function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

# choose what to plot
var_choice = 4 # number to choose what to plot 1: FFT(<c), 2: ∇c, <uc>, 3: <c>, 4: <c'^2>, 5: <c'^2>/<c>^2>, 6: d<uc>/dx, 7: plot the error in mean estimate as a function of x ((⟨c⟩ - c_0)^2/c_0^2), 8: plot ⟨c^2⟩, 9: plot the magnitude of the terms in the equation for d<c'2>/dt, 10: FFT(⟨c'^2⟩)
averaging = 1 # 1: no averaging, 2: plot t+1 - t, 3: average from time indx
indx_to_average_from = 1 # must be between 1 and length(load_times)

# variables to choose to plot (these are chosen but line_variable_num/panel_variable_num will otherwise select more options)
mag = 0.7
u_force = 1.0
κ = 0.001
λ = 1.0

# load in the data
x_length = 512 #1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

# sizes
legendsize = 30;
axlabelsize = 30;
axtitlesize = 40;

line_options = (; linewidth = 3)
load_times = collect(100: 100: 4000)

# first load in the data
data = zeros(x_length, length(load_times))
data2 = zeros(x_length, length(load_times))
data3 = zeros(x_length, length(load_times))

fluctuation_labels =fill("", length(load_times))

fig = Figure(resolution = (3024, 1964),)

# load in the data
nindx = 1
for time in ProgressBar(load_times)
        data_folder = "data/gpu/kappa_0.001/test/code_fixes"
        if time == 5000
            data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            load_name = joinpath(data_folder, data_name)
            #try
            @load load_name c_mean flux_mean c_squared_mean gc
            xvar, yvar = load_variables_u(ax, var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k, λ, mag)
            #catch systemerror
            #    print("no file named " * data_name)
            #end
        else
            data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_num_" * string(time) * ".jld2"
            load_name = joinpath(data_folder, data_name)
            #try
            @load load_name c_mean_current flux_mean_current c_squared_mean c_current u_current
            c_mean = c_mean_current
            flux_mean = flux_mean_current
            gc = real(ifft(im * Array(k)[:, 1] .* fft(c_mean_current)))
            xvar, yvar = load_variables_u(var_choice, c_mean_current, flux_mean_current, c_squared_mean, gc, x, k, λ, mag)
            #catch systemerror
            #   print("no file named " * data_name)
            #end
        end
        
            # obtain the variables for plotting
            try    
                data[:, nindx] = yvar

                if var_choice == 2
                    data2[:, nindx] = gc[:]
                elseif var_choice == 6
                    data2[:, nindx] = λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(x)))[:]
                    data3[:, nindx] = λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(x))) .- λ*(c_squared_mean .- c_mean.^2) ./(1 .+ mag*cos.(x))[:]
                    c_0_mean = (0.5*tanh(1.437392192121757*log10(λ))+0.5) * (1 - (1-mag^2)^0.5) + (1-mag^2)^0.5
                    c_1_mean = (0.5*tanh(1.44621301525833*log10(λ))+0.5)*0.7
                    c_mean_pred = c_0_mean .+ c_1_mean*cos.(x)
                    data3[:, nindx] = λ*c_mean_pred.*(1 .- c_mean_pred./(1 .+ mag*cos.(x)))[:]
                    data3[:, nindx] = -1/(1+λ)*mag^2*sin.(2*x)
                elseif var_choice == 3
                    data2[:, nindx] = mean(c_mean) .+ (maximum(c_mean) - minimum(c_mean))/2 * cos.(x)[:]
                    c_0_mean = (0.5*tanh(1.437392192121757*log10(λ))+0.5) * (1 - (1-mag^2)^0.5) + (1-mag^2)^0.5
                    c_1_mean = (0.5*tanh(1.44621301525833*log10(λ))+0.5)*0.7
                    c_mean_pred = c_0_mean .+ c_1_mean*cos.(x)
                    data3[:, nindx] = c_mean_pred
                elseif var_choice == 9
                    data2[:, nindx] = abs.(real(ifft(im*k[:,1].*fft(flux_c_square_mean))));
                    data3[:, nindx] = abs.(-λ*c_cube_mean./(1 .+ mag*cos.(x)));
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
                
                        fluctuation_labels[nindx] = fluctuation_label
                        approx_func = approx_func .+ real(ft_cf[1])/x_length

                        
                        #ft_cf[4:1022] = zeros(1022-4+1)
                        data2[:, nindx] = approx_func #real((ifft(ft_cf)));

                end
            catch systemerror
                print("no file named " * data_name)
             end
    nindx = nindx + 1
end

if averaging == 2
    data_new = zeros(x_length, length(load_times)-1)
    for indx = 1:length(load_times)-1
        data_new[:, indx] = (data[:, indx + 1]*load_times[indx + 1] - data[:, indx]*load_times[indx])/(load_times[indx + 1]-load_times[indx])
    end
    global data
    data = data_new
    load_times = load_times[1:end-1]
elseif averaging == 3
    data_new = zeros(x_length, length(load_times) - indx_to_average_from)
    for indx = indx_to_average_from:length(load_times)-1
        data_new[:, indx - indx_to_average_from + 1] = (data[:, indx + 1]*load_times[indx + 1] - data[:, indx_to_average_from]*load_times[indx_to_average_from])/(load_times[indx + 1] - load_times[indx_to_average_from])
    end
    global data
    data = data_new
    load_times = load_times[indx_to_average_from:end]
end

# now plot the data
panel_indx = Observable(1)
choice_line = @lift(data[:, $panel_indx])
ymax = @lift(maximum(data[:, $panel_indx]))
if var_choice == 2 || var_choice == 6 || var_choice == 3 || var_choice == 9 || var_choice == 4
    choice_line2 = @lift(data2[:, $panel_indx])
end
if var_choice == 6 || var_choice == 9
    choice_line3 = @lift(data3[:, $panel_indx])
end

ylabel = choose_axis(var_choice)

if var_choice == 2 || var_choice == 3
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("t = $(round(load_times[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x, choice_line2, color = :red, linewidth = 4, linestyle = :dash)
    ylims!(minimum(data), maximum(data)) 
    #axislegend()
elseif var_choice == 6
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("t = $(round(load_times[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x, choice_line2, color = :red, linewidth = 4, label = L"λc̅(1-c̅/(1+Δ(x)))")
    c̅ = 1/2*(1 .+ mag*cos.(x)) .+ sqrt(1-mag^2)/2
    c_0 = 1 .+ mag*cos.(x)
    lines!(fig.axis, x, choice_line3, color = :green, linewidth = 4, label = L"λc̅_{pred}(1-c̅_{pred}/(1+Δ(x)))")
    ylims!(minimum(data), maximum(data)) 
    axislegend()
elseif var_choice == 4
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (
            xlabel = "x",
            ylabel = ylabel,
            title = @lift( "t = " * string(round(load_times[$panel_indx], digits = 2)) * string(fluctuation_labels[$panel_indx, :])),))
    lines!(fig.axis, x, choice_line2, color = :red, linewidth = 4, linestyle = :dash)
    ylims!(minimum(data), maximum(data)) 
elseif var_choice == 1 || var_choice == 10
    fig = scatterlines(k, choice_line, color = :blue, linewidth = 4; line_options...,
    axis = (xlabel = "k", ylabel = ylabel, title = @lift("t = $(round(load_times[$panel_indx], digits = 2))"),))
    ylims!(minimum(-10), maximum(5))
    #xlims!(minimum(k), maximum(k))
    xlims!(-30, 30)
elseif var_choice == 9
    fig = lines(
        x, choice_line, color = :blue, linewidth = 4;
        line_options...,
        axis = (yscale = log10,
            xlabel = "x",
            ylabel = ylabel,
            title = @lift("t = $(round(load_times[$panel_indx], digits = 2))"),))
    lines!(fig.axis, x, choice_line3, color = :red, linewidth = 4, linestyle = :dash, label = L"λ⟨c'^3⟩/(1+Δ(x)))")
    lines!(fig.axis, x, choice_line2, color = :green, linewidth = 4, linestyle = :dot, label = L"∂⟨uc'^2⟩/∂x")
    #ylims!(minimum([minimum(data), minimum(data2), minimum(data3)]), maximum([maximum(data), maximum(data2), maximum(data3)]))
    ylims!(10^-6, (maximum([maximum(data), maximum(data2), maximum(data3)])))
    #ylims!(0, @lift(maximum(data[:, $panel_indx, 1])))
    axislegend() 
else
    fig = lines(x, choice_line, color = :blue, linewidth = 4; line_options...,
    axis = (xlabel = "x", ylabel = ylabel, title = @lift("t = $(round(load_times[$panel_indx], digits = 2))"),))
    ylims!(minimum(data), maximum(data))
end

size(data, 2)

framerate = 10
timestamps = range(start = 1, stop = size(data, 2), step=1)

movie_name =  name_figure(var_choice, mag, u_force, λ, averaging)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    panel_indx[] = t
end
