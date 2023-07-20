using GLMakie
using JLD2
using FileIO
using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

# use if wanting to vary both quantities

function name_figure(plot_type, var_choice, quantity_to_vary, line_variable_num, line_variable, fix_to_one_wavelength, mag, div, λ)
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
        method_name = "concentration_squared_"
    elseif var_choice == 5
        method_name = "relavitve_concentration_squared_"
    else
        method_name = "flux_gradient_"
    end

    if quantity_to_vary == 1
        xvar = "panel_mag_"
    elseif quantity_to_vary == 2
        xvar = "panel_k_"
    else
        xvar = "panel_λ_"
    end
    
    if line_variable_num == 1
        if length(line_variable) == 1
            lvar = "line_mag_" * string(line_variable[1]) * "_"
        else
            lvar = "line_mag_"
        end
    elseif line_variable_num == 2
        if length(line_variable) == 1
            lvar = "line_k_" * string(line_variable[1]) * "_"
        else
            lvar = "line_k_"   
        end    
    else
        if length(line_variable) == 1
            lvar = "line_λ_" * string(line_variable[1]) * "_"
        else
            lvar = "line_λ_"   
        end   
    end

    if quantity_to_vary!= 1 && line_variable_num != 1
        fvar = "mag_" * string(mag)
    elseif quantity_to_vary!= 2 && line_variable_num != 2
        fvar = "k_" * string(div)
    else
        fvar = "lambda_" * string(λ)
    end

    if fix_to_one_wavelength
        wave = "_one_wavelength"
    else
        wave = ""
    end

    fig_save_name = method_name * plot_panels_name * xvar * lvar * fvar * wave * ".png"
    return fig_save_name
end

function load_variables(ax, var_choice, gc, ff, cf, cs, x, k)
    if var_choice == 1
        ft_cf=abs.(fft(cf))[:]
        xvar = k
        yvar = log10.(ft_cf)[:]
        ylims!(minimum(-16), maximum(5))
        xlims!(minimum(k), maximum(k))
    elseif var_choice == 2
        xvar = gc[:]
        yvar = ff[:]
    elseif var_choice == 3
        xvar = x
        yvar = cf[:]
    elseif var_choice == 4
        xvar = x
        yvar = mean(cs[:,201:end].^2,dims = 2)[:];
    elseif var_choice == 4
        xvar = x
        yvar = mean(cs[:,201:end].^2,dims = 2)[:]./(1 .+ cf[:]).^2
    else
        xvar = x
        yvar = real(ifft(im*k[:,1].*fft(ff)))[:];
    end
    return xvar, yvar
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
        yvar = c_squared_mean[:];
    elseif var_choice == 4
        xvar = x
        yvar = (c_squared_mean.^2 .- (c_mean[:]).^2)/c_mean.^2
    else
        xvar = x
        yvar = real(ifft(im*k[:,1].*fft(flux_mean)))[:];
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
        ylabel = "⟨c'⟩"
    elseif var_choice == 4
        xlabel = "x"
        ylabel = "⟨c'⟩"
    elseif var_choice == 5
        xlabel = "x"
        ylabel = L"⟨c'^2⟩/⟨c⟩^2"
    elseif var_choice == 6
        xlabel = "x"
        ylabel = "∂⟨uc⟩/∂x"
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

function axis_title(mag, div, λ, r, varu; quantity_to_vary = 3)
    if quantity_to_vary == 1
        axtitle = "|Δ| = " * string(mag)
    elseif quantity_to_vary == 2
        axtitle = "L = " * string(round(2*pi/div*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r"
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
var_choice = 6 # number to choose what to plot 1: FFT(<c) and FFT(Δ(x)), 2: ∇c, <uc>, 3: <c'>, 4: <c'^2>, 5: <c'^2>/<c>^2>, 6: d<uc>/dx and and λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels 
#quantity_to_vary = 2 # 1: vary magnitude, 2: vary wavelength of forcing, 3: vary both mag and wavelength, 4: vary lambda
panel_variable_num = 3 # choose what each panel will vary with, 1: magnitude, 2: k, 3: lambda
line_variable_num = 2 # choose what each line in each panel will be, 1: magnitude, 2: k, 3: lambda 
no_u = false #true #false # either plot u = 0 or u non 0
fix_to_one_wavelength = false # true # plot all over only one wavelength

r = 0.2    # damping rate in OE
varu = r^2 #0 #0.1   # variance of u

# variables to choose to plot
mag = 0.7
div = 3
κ = 0.01
λ = 1 #0.1 #0.05

# options to choose from

if varu == r^2
    magnitudes = [0.1, 0.5, 0.7, 0.9]
    divisor = [1, 3, 6, 13, 25]
    lambdas = [0.5, 1, 1.5] #[0.01, 0.1, 1, 5] #[0.01, 0.1, 0.5, 1, 1.5, 5, 10]

    # subgroup for each line plot
    line_magnitudes = [0.7]
    line_divisor = [3]
    line_lambdas = [1]

else
    if plot_type == 3
        magnitudes = [0.025, 0.1, 0.5, 0.7]
        divisor = [1, 13, 25, 125]
    else
        magnitudes = [0.01, 0.7] # [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        divisor = [1, 3, 6, 13, 25] #[1, 3, 6, 13, 25, 63, 125]
    end
end

x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

### plot the data
if panel_variable_num == 1
    panel_variable = magnitudes
elseif panel_variable_num == 2
    panel_variable = divisor
else
    panel_variable = lambdas
end

if line_variable_num == 1
    line_variable = line_magnitudes
elseif line_variable_num == 2
    line_variable = line_divisor
else
    line_variable = line_lambas
end

# sizes
legendsize = 30;
axlabelsize = 30;
axtitlesize = 40;

# plot nabla c against <uc> for various choices of delta
fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

nindx = 1
for pvar in ProgressBar(panel_variable)

    # decide what variable we are obtaining
    if panel_variable_num == 1
        mag = pvar
    elseif panel_variable_num == 2
        div = pvar
    else
        λ = pvar
    end
    mindx = 1
    for lvar in line_variable
        if line_variable_num == 1
            mag = lvar
        elseif line_variable_num == 2
            div = lvar
        else
            λ = lvar
        end
        if varu == r^2
            data_folder = "data/u_rat_1"
            data_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_lambda_" * string(λ) * "_FT.jld2"
            # Concatenate the folder and file name to get the full path
            load_name = joinpath(data_folder, data_name)
            @load load_name c_mean flux_mean c_squared_mean gc
        elseif no_u
            if λ == 0.1 || λ == 0.05
                load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_kappa_" * string(κ) * "_lambda_" * string(λ) * "_nou_FT.jld2"
            else
                load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_kappa_" * string(κ) * "_nou_FT.jld2"
            end
            @load load_name cs cf
            #load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(round(κ, sigdigits = 3)) * "_nou_FT.jld2"
            #@load load_name cs cf
            gc = []
            fs = []
            ff = []
        else
            load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(κ) * "_FT.jld2"
            @load load_name cs fs ff cf gc
        end

        axtitle = axis_title(mag, div, λ, r, varu; quantity_to_vary = panel_variable_num)
        ax = choose_axis(fig, plot_type, nindx, axtitle, var_choice)
        ax.xlabelsize = axlabelsize
        ax.ylabelsize = axlabelsize
        ax.titlesize = axtitlesize

        # obtain the variables for plotting
        if varu == r^2
            xvar, yvar = load_variables_u(ax, var_choice, c_mean, flux_mean, c_squared_mean, gc, x, k)
        else
            xvar, yvar = load_variables(ax, var_choice, gc, ff, cf, cs, x, k)
        end
        if length(line_variable) == 1
            colours = specify_colours(2)
        else
            colours = specify_colours(length(line_variable))
        end
        if fix_to_one_wavelength
            xlims!(ax, -pi/div, pi/div)
        end
        lines!(ax, xvar, yvar, label = "L = " * string(round(2*pi/div*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r, |Δ| = " * string(mag) * ", λ = " * string(round(λ, sigdigits = 1)), color = colours[mindx])
        if var_choice ==6
            if varu == r^2
                lines!(ax, xvar, (λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(div*x))))[:], label = "λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))", linestyle = :dash, color = colours[mindx])
                lines!(ax, xvar, (λ*c_mean .- λ*c_squared_mean./(1 .+ mag*cos.(div*x)))[:], label = "λ(⟨c⟩-(⟨c⟩^2 + c'^2)/(1+Δ(x)))", linestyle = :dot, color = colours[mindx])
            end
        end
            if mindx == 1
                if var_choice == 1
                    Δconc = mag*cos.(div*x)
                    lines!(ax, k, log10.(abs.(fft(Δconc))), color = :black, linewidth = 2, label = "scaled Δ(x)")
                elseif var_choice == 2
                    model = lm(@formula(y ~ x), DataFrame(x=xvar, y=yvar))
                    coeffs = coef(model)  
                    print(coeffs)        # Linear regression
                    lines!(ax, xvar, coeffs[1] .+ coeffs[2]*xvar, color = :black, label = "∇⟨uc⟩ = -" * string(round(-coeffs[2], sigdigits = 2)) * "∇c + " * string(round(coeffs[1], sigdigits = 1)))
                elseif var_choice == 3
                    lines!(ax, xvar, cos.(div*xvar)*(maximum(yvar) - minimum(yvar))/2 .+ (maximum(yvar) + minimum(yvar))/2, linestyle = :dash, color = :black, label = L"cos(kx)")
                elseif var_choice == 4
                    lines!(ax, xvar, (cos.(div*xvar)).^2*(maximum(yvar) - minimum(yvar))/2 .+ (maximum(yvar) + minimum(yvar))/2, linestyle = :dash, color = :black, label = L"cos^2(kx)")
                end
                axislegend(position = :rt, labelsize = legendsize)
            elseif mindx == 2
                ax.yticklabelcolor = colours[mindx]
                ax.yaxisposition = :right
                ax.xticklabelcolor = colours[mindx]
                ax.xaxisposition = :top
                axislegend(position = :lt, labelsize = legendsize)
            end
        mindx = mindx + 1
    end
    nindx = nindx + 1
end
if plot_type == 1
    axislegend(labelsize = legendsize)
end
display(fig)

save(name_figure(plot_type, var_choice, panel_variable_num, line_variable_num, line_variable, fix_to_one_wavelength, mag, div, λ), fig) #, pt_per_unit=2) # size = 600 x 450 pt

# test glenn's plots
# 1) plot <c> and 1+Δ(x)
# 2) plot <c> and d<c>/dx
# 3) plot d<c>/dx against <u'c'>
# 4) plot λ<c>(1-<c>/(1+Δ(x))) and d<u'c'>/dx

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
ax = Axis(fig[1, 1])
#xlabel!(fig, L"{x}")

# load in the data
mag = 0.7;
div = 3;
λ= 0.1; 
#load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "test_FT.jld2"
#@load load_name cs fs ff cf gc
#x_glenn = nodes(512, a = -pi, b = pi)

# glenn has used 1+b, I've used b
save_name = "mag_0.1_k_1.0_lambda_0.1_FT.jld2"

data_folder = "data/test_glenn"
data_name = "test_mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_lambda_" * string(λ) * "_FT.jld2"
load_name = joinpath(data_folder, data_name)
@load load_name c_mean flux_mean c_squared_mean gc
x_glenn = nodes(1024, a = -pi, b = pi)
cf = c_mean .- 1
ff = flux_mean

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
ax = Axis(fig[1, 1])

lines!(ax, x_glenn, 1 .+ cf[:], label = "⟨c⟩")
lines!(ax, x_glenn, 1 .+ mag*cos.(div*x_glenn), label = "1+Δ(x)")
lines!(ax, x_glenn, gc[:], label= "∇c")
lines!(ax, x_glenn, λ*(1 .+ cf[:]).*(1 .- (1 .+ cf[:])./(1 .+mag*cos.(div*x_glenn))), label = "λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))")
lines!(ax, x_glenn, (λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(div*x_glenn))))[:], label = "λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))")
lines!(ax, gc[:], ff[:])
∇uc=real(ifft(im*k[:,1].*fft(ff)));
lines!(ax, x_glenn, ∇uc[:], label = "∇⟨uc⟩")
axislegend()
#save("t1_0.5-c0-and-c.png", fig) #, pt_per_unit=2) # size = 600 x 450 pt
#save("t1_0.5-c-and-grad.png", fig) 
#save("t1_0.5-grad-flux.png", fig)
save("t1_0.5-rs-and-div-flux.png", fig)

# plot the particle concentration against delta

x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)
Δconc = mag*cos.(div*x)
varu= 0 #0.1   # variance of u
r=0.2    # damping rate in OE
# variables to choose to plot
mag = 0.01
div = 1
κ = 10
λ = 0.05 #0.1 #0.05

# load in the data
movie_name = "mag_" * string(mag) * "_k_" * string(div) * "_kappa_" * string(κ) * "_lambda_" *string(λ) * "_FT.mp4"
if varu == 0
    if λ == 0.1 || λ  == 0.05
        load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_kappa_" * string(round(κ, sigdigits = 3)) * "_lambda_" * string(λ) * "_nou_FT.jld2"
    else
        load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_kappa_" * string(κ) * "_nou_FT.jld2"
    end
    @load load_name cs cf
else
        load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits = 3)) * "_FT.jld2"
        @load load_name cs fs ff cf gc
end
if cs[10, 1000] == 0
    end_time = 999
else
    end_time = 1000
end
#isnan(cs[:, 1])

#nanrows = any(isnan, cs[:, 1])
#cs[!vec(nanrows), :]

@load "test_mag_0.7_k_1.0_lambda_0.05_FT.jld2" c_mean flux_mean c_squared_mean gc

time = Observable(1)
#x = nodes(size(cs, 1))
cs_line = @lift(cs[:, $time])

fig = lines(x, cs_line, color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Delta(x) = " *string(mag) * "cos" * "(" * string(div) * ",x) t = $(round($time, digits = 1))"),))
    ylims!(minimum(cs[:, 1:end_time]), maximum(cs[:, 1:end_time]))
#lines!(x, Δconc/maximum(Δconc)*(maximum(cs) - minimum(cs))/2 .+ (maximum(cs) + minimum(cs))/2, color = :red, linewidth = 0.5, label = L"scaled \Delta(x)")
axislegend()

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    time[] = t
end