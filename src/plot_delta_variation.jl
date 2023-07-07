using GLMakie
using JLD2
using ReactingTracers
GLMakie.activate!(inline=false)

# choose what to plot
plot_panels = true # either plots all the data in one plot or on separate panels
plot_divisor = true #false #true # either plot the effect of varying magnitude or varying divisor
take_fourier_transform = true

varu=0.1   # variance of u
r=0.2    # damping rate in OE

# variables to choose to plot
mag = 0.7
div = round(varu/r*4*pi/13, sigdigits = 3)

# options to choose from
#divisor = varu/r*[0.1, 0.5, 1, 2, 5, 10]
divisor = round.(varu/r*4*pi*[1/126, 1/63, 1/25, 1/13, 1/2, 1], sigdigits = 3) #peridic c0
magnitudes = [0.1, 0.5, 0.7, 0.8]

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
fig = Figure(xlabel = L"\nabla c",
xlabelsize = 22, ylabel = L"\overline{uc}", ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
ax = Axis(fig[1, 1])

nindx = 1
for var in variable
    if plot_divisor
        div = var
    else
        mag = var
    end
    load_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.jld2"
    @load load_name cs fs ff cf gc

    if take_fourier_transform
        Δconc = mag*sin.(2*pi/div*x)
        ft_cf=abs.(fft(cf))[:]
        if plot_panels
            ax = Axis(fig[nindx, 1], title = "L = " * string(round(div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1    
            lines!(ax, k, log10.(ft_cf)[:], color = :blue, linewidth = 4, label = "c")
            lines!(ax, k, log10.(abs.(fft(Δconc))), color = :red, linewidth = 0.5, label = "scaled Δ(x)")
            axislegend()
        end
    else
        if plot_panels
            ax = Axis(fig[nindx, 1], title = "L = " * string(round(div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
            nindx= nindx + 1
        end
        lines!(ax, gc[:],ff[:], label = "L = " * string(round(div*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " *string(mag))
    end
end
if !plot_panels
    axislegend()
end
display(fig)
save("normal.png", fig, pt_per_unit=2) # size = 600 x 450 pt
# plot the particle concentration against delta

# chosen variables
mag = 0.7
div = round(varu/r*4*pi/1, sigdigits = 3) #[1/126, 1/63, 1/25, 1/13, 1/2, 1]
lengthscale = 2*pi/div
x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)
Δconc = mag*sin.(2*pi/div*x)
ft_cf=abs.(fft(cf))[:]
fig = Figure()
ax = Axis(fig[1, 1], xminorgridvisible = true, xgridwidth = 1)
lines!(ax, k, log10.(ft_cf)[:], color = :blue, linewidth = 4, label = "c")
    #ylims!(minimum(cs), maximum(cs))
lines!(ax, k, log10.(abs.(fft(Δconc))), color = :red, linewidth = 0.5, label = L"scaled Δ(x)")
axislegend()

test = abs.(fft(Δconc))

# load in the data
movie_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.mp4"
load_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.jld2"
@load load_name cs fs ff cf gc

fig = lines(x, cf[:], color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Delta(x) = " *string(mag) * "sin" * "2pi x/" * string(div) * ", t = $(round($time, digits = 1))"),))
    #ylims!(minimum(cs), maximum(cs))
lines!(x, Δconc/maximum(Δconc)*(maximum(cf) - minimum(cf))/2 .+ (maximum(cf) + minimum(cf))/2, color = :red, linewidth = 0.5, label = L"scaled \Delta(x)")
axislegend()


time = Observable(1)

x = nodes(size(cs, 1))
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