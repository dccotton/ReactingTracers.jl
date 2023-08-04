using GLMakie
using JLD2
using FileIO

using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

function name_figure(plot_type, var_choice, quantity_to_vary, line_variable_num, line_variable, mag, U, λ)
    if plot_type == 2
        plot_panels_name = "panels_"
    elseif plot_type == 3
        plot_panels_name = "four_"
    else
        plot_panels_name = "single_"
    end
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

    fig_save_name = method_name * plot_panels_name * xvar * lvar * fvar * ".png"
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

function choose_axis(fig, plot_type, nindx, axtitle, var_choice)
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
        ylabel = "⟨c^2 - ⟨c⟩^2⟩"
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
    
    if plot_type == 2
        ax = Axis(fig[nindx, 1], title = axtitle, xlabel = xlabel, ylabel = ylabel)
    elseif plot_type == 3
        ax = Axis(fig[four_panels[nindx], four_panels[nindx+4]], title = axtitle, xlabel = xlabel, ylabel = ylabel)
    else
        ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel)
    end

    return ax
end

function axis_title(mag, U, λ; quantity_to_vary = 3)
    if quantity_to_vary == 1
        axtitle = "|Δ| = " * string(mag)
    elseif quantity_to_vary == 2
        axtitle = "U = " * string(U)
    else
        axtitle = "λ = " * string(λ)
    end
    return axtitle
end

function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

# choose what to plot
var_choice = 3 # number to choose what to plot 1: FFT(<c) and FFT(Δ(x)), 2: ∇c, <uc>, 3: <c>, 4: <c'^2>, 5: <c'^2>/<c>^2>, 6: d<uc>/dx and λ⟨c⟩(1-⟨c⟩/(1+Δ(x))) and λ⟨c⟩(1-⟨c⟩+⟨c'2⟩/⟨c⟩>/(1+Δ(x))), 7: plot the error in mean estimate as a function of x ((⟨c⟩ - c_0)^2/c_0^2), 8: plot ⟨c^2⟩
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels, 4: animation
panel_variable_num = 3 # choose what each panel will vary with, 1: magnitude, 2: U, 3: lambda
line_variable_num = 4 # choose what each line in each panel will be, 1: magnitude, 2: U, 3: lambda, 4: kappa 

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
    lambdas = [0.01, 0.1, 1.0, 10.0] #, 100.0] #[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5] #[0.01, 0.1, 1, 5] #[0.01, 0.1, 0.5, 1, 1.5, 5, 10]
    #lambdas = [1.0]
    #lambdas = sort([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 100.0])#lambdas = [3.0, 5.0, 6.0, 10.0]
    #lambdas = [0.1, 1.0, 2.0, 10.0]

    # subgroup for each line plot
    line_magnitudes = [0.7] #[0.5, 0.7, 0.9]
    line_velocity = [1]
    line_lambdas = [3, 5, 6, 10.0]
    line_kappas = [0.01, 0.001, 0.0001]
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


line_options = (; linewidth = 6)

# plot nabla c against <uc> for various choices of delta
fig = Figure(resolution = (3024, 1964),)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

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

        # set the axis values

        if plot_type != 1 && mindx == 1 || nindx == 1 && mindx == 1 
            axtitle = axis_title(mag, u_force, λ; quantity_to_vary = panel_variable_num)
            #print(nindx)
            ax = choose_axis(fig, plot_type, nindx, axtitle, var_choice)
            ax.xlabelsize = axlabelsize
            ax.ylabelsize = axlabelsize
            ax.titlesize = axtitlesize
        end
        
        # load in the data
        if varu != 0
            data_folder = "data/gpu/kappa_0.001"
            if κ == 0.001
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            elseif κ == 0.01
                data_folder = "data/gpu/kappa_0.01"
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            elseif κ == 0.0001
                data_folder = "data/gpu/kappa_0.0001"
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * ".jld2"
            else
                data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * ".jld2"
            end
            # Concatenate the folder and file name to get the full path
            load_name = joinpath(data_folder, data_name)
            #load_name = data_name
            try
                @load load_name c_mean flux_mean c_squared_mean gc# cs fs

                        # obtain the variables for plotting
        if varu != 0
            xvar, yvar = load_variables_u(ax, var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k)
        end
        if length(line_variable) == 1
            colours = specify_colours(2)
        else
            colours = specify_colours(length(line_variable))
        end

        # plot the data
        lines!(ax, xvar, yvar, label = "κ =" * string(κ), color = colours[mindx]; line_options...)
        #lines!(ax, xvar, yvar, label = "U = " * string(u_force) * ", |Δ| = " * string(mag) * ", λ = " * string(round(λ, sigdigits = 1)) * ", κ =" * string(κ), color = colours[mindx]; line_options...)
        if var_choice ==6
            if varu != 0
                #lines!(ax, xvar, (λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(x))))[:], label = "λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))", linestyle = :dash, color = colours[mindx]; line_options...)
                #lines!(ax, xvar, (λ*c_mean .- λ*c_squared_mean./(1 .+ mag*cos.(x)))[:], label = "λ(⟨c⟩-(⟨c⟩^2 + c'^2)/(1+Δ(x)))", linestyle = :dot, color = colours[mindx]; line_options...)
                #lines!(ax, xvar, -1/(1+λ)*mag^2*sin.(2*x)[:], label = "-1/(1+λ)|Δ|^2sin(2x)", linestyle = :dashdot, color = colours[mindx]; line_options...)
                lines!(ax, xvar, 1 ./(4*c_0) .* (mag*(mag .+ mag*cos.(x).^2 .+ 2*cos.(x))), label = L"λ\frac{|Δ|(|Δ| + |Δ|cos^2x + 2cosx)}{4Δ(x)}", linestyle = :dash, color = colours[mindx]; line_options...)
            end
        elseif var_choice == 4 && λ > 9
            lines!(ax, xvar, (sin.(xvar)).^2*mag^2*u_force^2/(λ*(1+λ)), linestyle = :dash, color = :black, label = L"Δ^2U^2sin^2(x)/(λ(1+λ))^2"; line_options...)
            #lines!(ax, xvar, mean((cs[:, 101:end-1] .- c_mean[:, 1]).^2, dims = 2)[:], linestyle = :dot, color = colours[mindx], label = L"⟨(c-⟨c⟩)^2⟩"; line_options...)
        end
            if mindx == 1
                if var_choice == 1
                    Δconc = mag*cos.(x)
                    lines!(ax, k, log10.(abs.(fft(Δconc))), color = :black, label = "scaled Δ(x)"; line_options...)
                elseif var_choice == 2
                    model = lm(@formula(y ~ x), DataFrame(x=xvar, y=yvar))
                    coeffs = coef(model)  
                    print(coeffs)        # Linear regression
                    lines!(ax, xvar, coeffs[1] .+ coeffs[2]*xvar, color = :black, label = "∇⟨uc⟩ = -" * string(round(-coeffs[2], sigdigits = 2)) * "∇c + " * string(round(coeffs[1], sigdigits = 1)); line_options...)
                elseif var_choice == 3
                    #lines!(ax, xvar, 1 .+ mag*cos.(xvar), label = "1+Δ(x)"; line_options...)
                    #lines!(ax, xvar, ones(length(xvar))*(1 - mag^2)^0.5, label = "√(1-|Δ|^2)"; line_options...)
                    #lines!(ax, xvar, 1/2 .+ mag*cos.(xvar)/2 .+ (1 - mag^2)^0.5/2, label = "1/2(Δ(x) + √(1-|Δ|^2))"; line_options...)

                elseif var_choice == 7
                    lines!(ax, xvar, abs.(c_mean[:] .- (1 - mag^2)^0.5)./abs.(c_mean[:]), label = "|⟨c⟩ - √(1-|Δ|^2)|/|⟨c⟩|")
                end
            elseif mindx == 2 && plot_type != 1 && shareaxis == false
                ax.yticklabelcolor = colours[mindx]
                ax.yaxisposition = :right
                ax.xticklabelcolor = colours[mindx]
                ax.xaxisposition = :top
                axislegend(position = :lt, labelsize = legendsize)
            end
            if shareaxis == true || mindx == 1
                axislegend(position = :rt, labelsize = legendsize)
            end
            catch systemerror
                print("no file named " * data_name)
            end
        end

        mindx = mindx + 1
    end
    nindx = nindx + 1
end
#if plot_type == 1
#    axislegend(labelsize = legendsize)
#end
display(fig)

save(name_figure(plot_type, var_choice, panel_variable_num, line_variable_num, line_variable, mag, u_force, λ), fig) #, pt_per_unit=2) # size = 600 x 450 