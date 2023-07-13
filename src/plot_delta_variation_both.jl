using GLMakie
using JLD2
using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
GLMakie.activate!(inline=false)

# use if wanting to vary both quantities

function name_figure(plot_type, plot_divisor, var_choice)
    if plot_type == 2
        plot_panels_name = "panels_"
    elseif plot_type == 3
        plot_panels_name = "four_"
    else
        plot_panels_name = "single_"
    end
    if var_choice == 1
        method_name = "FT"
    elseif var_choice == 2
        method_name = "flux"
    elseif var_choice == 3
        method_name = "concentration"
    else
        method_name = "concentration_squared"
    end
    
    fig_save_name = "both_" * plot_panels_name * method_name * ".png"
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
    else
        xvar = x
        yvar = mean(cs[:,201:end].^2,dims = 2)[:]./(1 .+ cf[:]).^2
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

function axis_title(mag, div, r, varu; quantity_to_vary = 3)
    if quantity_to_vary == 1
        axtitle = "|Δ| = " * string(mag)
    else
        axtitle = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r"
    end
    return axtitle
end

function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

# choose what to plot
var_choice = 2 # number to choose what to plot 1: FFT(<c) and FFT(Δ(x)), 2: ∇c, <uc>, 3: <c'>, 4: <c'^2>, 5: <c'^2>/<c>^2>
plot_type = 2 # number to choose which panels to plot 1: one plot, 2: panels, 3: four panels
quantity_to_vary = 3 # 1: vary magnitude, 2: vary wavelength of forcing, 3: vary both 

varu=0.1   # variance of u
r=0.2    # damping rate in OE

# variables to choose to plot
mag = 0.7
div = 3
κ = 0.01

# options to choose from

if plot_type == 3
    magnitudes = [0.025, 0.1, 0.5, 0.7]
    divisor = [1, 13, 25, 125]
else
    magnitudes = [0.01, 0.7] # [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    divisor = [1, 3, 6, 13, 25] #[1, 3, 6, 13, 25, 63, 125]
end

x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

### plot the data
variable = divisor
plot_magnitudes = [0.7]

# plot nabla c against <uc> for various choices of delta
fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

nindx = 1
for var in variable
    div = var
    mindx = 1
    for mag in plot_magnitudes
        load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(κ) * "_FT.jld2"
        @load load_name cs fs ff cf gc

        axtitle = axis_title(mag, div, r, varu) # need to add quantity to vary = 1
        ax = choose_axis(fig, plot_type, nindx, axtitle, var_choice)
        xvar, yvar = load_variables(ax, var_choice, gc, ff, cf, cs, x, k)
        if length(plot_magnitudes) == 1
            colours = specify_colours(2)
        else
            colours = specify_colours(length(plot_magnitudes))
        end
        lines!(ax, xvar, yvar, label = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag), color = colours[mindx])

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
                axislegend(position = :rt)
            elseif mindx == 2
                ax.yticklabelcolor = colours[mindx]
                ax.yaxisposition = :right
                ax.xticklabelcolor = colours[mindx]
                ax.xaxisposition = :top
                axislegend(position = :lt)
            end
        mindx = mindx + 1
    end
    nindx = nindx + 1
end
if plot_type == 1
    axislegend()
end
display(fig)

save(name_figure(plot_type, plot_divisor, var_choice), fig) #, pt_per_unit=2) # size = 600 x 450 pt

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
λ=0.1; 
load_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "test_FT.jld2"
@load load_name cs fs ff cf gc
x_glenn = nodes(512, a = -pi, b = pi)

# glenn has used 1+b, I've used b

lines!(ax, x_glenn, 1 .+ cf[:], label = "⟨c⟩")
lines!(ax, x_glenn, 1 .+ mag*cos.(div*x_glenn), label = "1+Δ(x)")
lines!(ax, x_glenn, gc[:], label= "∇c")
lines!(ax, x_glenn, λ*(1 .+ cf[:]).*(1 .- (1 .+ cf[:])./(1 .+mag*cos.(div*x_glenn))), label = "λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))")
lines!(ax, gc[:], ff[:])
∇uc=real(ifft(im*k[:,1].*fft(ff)));
lines!(ax, x_glenn, ∇uc[:], label = "∇⟨uc⟩")
axislegend()
#save("0.5-c0-and-c.png", fig) #, pt_per_unit=2) # size = 600 x 450 pt
#save("0.5-c-and-grad.png", fig) 
#save("0.5-grad-flux.png", fig)
save("0.5-rs-and-div-flux.png", fig)

# plot the particle concentration against delta

x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)
Δconc = mag*cos.(div*x)

# load in the data
movie_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.mp4"
load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits = 3)) * "_FT.jld2"
@load load_name cs fs ff cf gc

#isnan(cs[:, 1])

#nanrows = any(isnan, cs[:, 1])
#cs[!vec(nanrows), :]

time = Observable(1)
#x = nodes(size(cs, 1))
cs_line = @lift(cs[:, $time])

fig = lines(x, cs_line, color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Delta(x) = " *string(mag) * "cos" * "(" * string(div) * ",x) t = $(round($time, digits = 1))"),))
    ylims!(1, -2) #(minimum(cs), maximum(cs))
#lines!(x, Δconc/maximum(Δconc)*(maximum(cs) - minimum(cs))/2 .+ (maximum(cs) + minimum(cs))/2, color = :red, linewidth = 0.5, label = L"scaled \Delta(x)")
axislegend()

framerate = 10
timestamps = range(start = 1, stop = 1000, step=1)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    time[] = t
end

@load load_name cs fs ff cf gc
ft_cf=abs.(fft(cf))[:]
xvar = k
yvar = log10.(ft_cf)[:]