using GLMakie
using JLD2
GLMakie.activate!(inline=false)

# choose what to plot
plot_panels = true # either plots all the data in one plot or on separate panels
plot_divisor = false #true # either plot the effect of varying magnitude or varying divisor

varu=0.1   # variance of u
r=0.2    # damping rate in OE

# variables to choose to plot
mag = 0.7
div = round(varu/r*4*pi/13, sigdigits = 3)
#divisor = varu/r*[0.1, 0.5, 1, 2, 5, 10]
divisor = round.(varu/r*4*pi*[1/126, 1/63, 1/25, 1/13, 1/2, 1], sigdigits = 3) #peridic c0
magnitudes = [0.1, 0.5, 0.7]
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
        print(div)
    else
        mag = var
    end
    load_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.jld2"

    @load load_name cs fs ff cf gc
    if plot_panels
        ax = Axis(fig[nindx, 1], title = "L = " * string(round(div*r/varu; sigdigits =  3)) * "<v>/r, |Delta| = " *string(mag))
        nindx= nindx + 1
    end
    lines!(ax, gc[:],ff[:], label = "L = " * string(round(div*r/varu; sigdigits =  3)) * "<v>/r, |Delta| = " *string(mag))
end
if !plot_panels
    axislegend()
end
display(fig)
#f3


# plot the particle concentration against delta

# chosen variables
mag = 0.5
div = rounnd(varu/r*4*pi/25, sigdigits = 3) #[1/126, 1/63, 1/25, 1/13, 1/2, 1]
#lengthscale = 2*pi/div

# load in the data
#movie_name = "mag_" * string(mag) * "_k_" * string(lengthscale) * "_FT.mp4"
movie_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.mp4"
#load_name = "mag_" * string(mag) * "_k_" * string(lengthscale) * "_FT.jld2"
load_name = "mag_" * string(mag) * "_k_" * string(div) * "_FT.jld2"
@load load_name cs fs ff cf gc

#Δconc = mag*sin.(2*pi/lengthscale*x)
Δconc = mag*sin.(2*pi/div*x)


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