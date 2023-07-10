using GLMakie
using JLD2
using ReactingTracers
using FFTW
GLMakie.activate!(inline=false)

function name_figure(plot_panels, plot_divisor, plot_four, take_fourier_transform)
    if plot_panels
        plot_panels_name = "panels_"
    elseif plot_four
        plot_panels_name = "four_"
    else
        plot_panels_name = "single_"
    end
    if plot_divisor
        var_change_name = "mag_" * string(mag) * "_"
    else
        var_change_name = "L_" * string(round(div*r/varu; sigdigits = 3)) * "_"    
    end
    if take_fourier_transform
        method_name = "FT"
    else
        method_name = "flux"
    end
    
    fig_save_name = var_change_name * plot_panels_name * method_name * ".png"
    return fig_save_name
end

# choose what to plot
plot_panels = true #false #true # either plots all the data in one plot or on separate panels
plot_divisor = true #false #true #false #true #false #true # either plot the effect of varying magnitude or varying divisor
take_fourier_transform = true #false #true
plot_four = false #true # only plot four panels to stop things becoming cluttered (can't have plot_panels true and this true as well)

varu=0.1   # variance of u
r=0.2    # damping rate in OE

# variables to choose to plot
mag = 0.5
div = 13

# options to choose from

if plot_four
    magnitudes = [0.025, 0.1, 0.5, 0.7]
    divisor = [1, 13, 25, 125]
else
    magnitudes = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    divisor = [1, 3, 6, 13, 25, 63, 125]
end

x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)

### plot the data
if plot_divisor
    variable = divisor
else
    variable = magnitudes
end

# plot nabla c against <uc> for various choices of delta
fig = Figure(resolution = (3024, 1964), xlabel = L"\nabla c",
xlabelsize = 22, ylabel = L"\overline{uc}", ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
ax = Axis(fig[1, 1])


four_panels = [1, 1, 2, 2, 1, 2, 1, 2]

nindx = 1
for var in variable
    if plot_divisor
        div = var
    else
        mag = var
    end
    load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_FT.jld2"
    @load load_name cs fs ff cf gc

    if take_fourier_transform
        #Δconc = mag*sin.(2*pi/div*x)
        Δconc = mag*sin.(div*x)
        ft_cf=abs.(fft(cf))[:]
        if plot_panels
            ax = Axis(fig[nindx, 1], title = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1    
            lines!(ax, k, log10.(ft_cf)[:], color = :blue, linewidth = 4, label = "c")
            lines!(ax, k, log10.(abs.(fft(Δconc))), color = :red, linewidth = 2, label = "scaled Δ(x)")
            axislegend()
        end
        if plot_four
            ax = Axis(fig[four_panels[nindx], four_panels[nindx + 4]], title = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1    
            lines!(ax, k, log10.(ft_cf)[:], color = :blue, linewidth = 4, label = "c")
            lines!(ax, k, log10.(abs.(fft(Δconc))), color = :red, linewidth = 2, label = "scaled Δ(x)")
            axislegend()
        end            
    else
        if plot_panels
            ax = Axis(fig[nindx, 1], title = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1
        end
        if plot_four
            ax = Axis(fig[four_panels[nindx], four_panels[nindx+4]], title = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1
        end
        lines!(ax, gc[:],ff[:], label = "L = " * string(round(2*pi/div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
    end
end
if !plot_panels
    axislegend()
end
display(fig)

save(name_figure(plot_panels, plot_divisor, plot_four, take_fourier_transform), fig) #, pt_per_unit=2) # size = 600 x 450 pt


# plot the particle concentration against delta

x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)
Δconc = mag*sin.(div*x)

# load in the data
movie_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.mp4"
load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits = 3)) * "_FT.jld2"
@load load_name cs fs ff cf gc

time = Observable(1)
#x = nodes(size(cs, 1))
cs_line = @lift(cs[:, $time])

fig = lines(x, cs_line, color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Delta(x) = " *string(mag) * "sin" * "2pi x/" * string(div) * ", t = $(round($time, digits = 1))"),))
    ylims!(minimum(cs), maximum(cs))
lines!(x, Δconc/maximum(Δconc)*(maximum(cs) - minimum(cs))/2 .+ (maximum(cs) + minimum(cs))/2, color = :red, linewidth = 0.5, label = L"scaled \Delta(x)")
axislegend()

framerate = 10
timestamps = range(start = 1, stop = 1000, step=1)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    time[] = t
end