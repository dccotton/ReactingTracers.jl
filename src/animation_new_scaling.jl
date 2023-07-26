using GLMakie
using JLD2
using FileIO
using ReactingTracers
using FFTW
using PerceptualColourMaps
using DataFrames, GLM
using ProgressBars
GLMakie.activate!(inline=false)

# plot the particle concentration time evolution against delta

x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

# variables to choose to plot
mag = 0.7
u_force = 1.0
λ = 10.0

Δconc = mag*cos.(x)

# load in the data
data_folder = "data/new_scaling"
#data_folder = ""
data_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_0.001.jld2"
movie_name = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_0.001.mp4"
movie_name_fluc = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_0.001_fluc.mp4"
movie_name_fluc_squared = "mag_" * string(mag) * "_U_" * string(u_force) * "_lambda_" * string(λ) * "_k_0.001_fluc_squared.mp4"
# Concatenate the folder and file name to get the full path
load_name = joinpath(data_folder, data_name)
#load_name = data_name
@load load_name c_mean flux_mean c_squared_mean gc cs fs

if cs[10, end] == 0
    end_time = size(cs, 2) - 1
else
    end_time = size(cs, 2)
end

time = Observable(1)
cs_line = @lift(cs[:, $time])

fig = lines(x, cs_line, color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Δ(x) = " * string(mag) * "cos(x) t = $(round($time, digits = 1))"),))
    ylims!(minimum(cs[:, 1:end_time]), maximum(cs[:, 1:end_time]))
#lines!(x, Δconc/maximum(Δconc)*(maximum(cs) - minimum(cs))/2 .+ (maximum(cs) + minimum(cs))/2, color = :red, linewidth = 0.5, label = L"scaled Δ(x)")
lines!(x, 1 .+ Δconc, color = :red, linewidth = 0.5, label = L"1+Δ(x)")
lines!(x, c_mean[:], color = :black, linewidth = 1, label = L"⟨c⟩")
axislegend()

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

record(fig, movie_name, timestamps;
        framerate = framerate) do t
    time[] = t
end

# find the time to decay to statistically steady
# also plot the fluctuations minus the mean sum((c-⟨c⟩)^2)

ax_options = (; ylabelsize = 30, xlabelsize = 30, titlesize = 40)


fig2 = Figure()
ax = Axis(fig2[1, 1], yscale = log10) #; ax_options...)
lines!(ax, sum((cs[:, 1:end_time] .- c_mean[:, 1]).^2, dims = 1)[:])
ax.xlabel = "Time"
ax.ylabel = "Diff"

sum((cs[:, 1:end_time] .- c_mean[:, 1]).^2, dims = 1)

# test whether the fluctuations sum to 0
fig2 = Figure()
ax = Axis(fig2[1, 1]) #; ax_options...)
#lines!(ax, c_mean[:, 1])
lines!(ax, sum(cs[:, 101:end_time] .- c_mean[:, 1], dims = 2)[:])
ax.xlabel = "x"
ax.ylabel = "⟨c'⟩(x)"

# test whether the fluctuations sum to 0
fig2 = Figure()
ax = Axis(fig2[1, 1]) #; ax_options...)
#lines!(ax, c_mean[:, 1])
lines!(ax, sum(cs[:, 50:end_time] .- c_mean[:, 1], dims = 1)[:])
ax.xlabel = "time"
ax.ylabel = "⟨c'⟩(t)"

# test whether the fluctuations sum to 0
fig2 = Figure()
ax = Axis(fig2[1, 1]) #; ax_options...)
#lines!(ax, c_mean[:, 1])
lines!(ax, mean((cs[:, 101:end_time] .- c_mean[:, 1]).^2, dims = 2)[:])
lines!(ax, (c_squared_mean .- c_mean.^2)[:], linestyle = :dash)
#lines!(ax, (mean(cs[:, 101:end_time].*c_mean[:, 1], dims = 2) - (c_mean.^2))[:])

lines!(ax, mean((cs[:, 101:end_time] .- c_mean[:, 1]).^2, dims = 2)[:] - (c_squared_mean .- c_mean.^2)[:])

mean(cs[:, 101:end_time].*c_mean[:, 1], dims = 2)
mean(cs[:, 101:end_time].*c_mean[:, 1], dims = 2) - (c_mean.^2)

ax.xlabel = "x"
ax.ylabel = "⟨c'⟩(x)"


sum(sum(cs[:, 101:end_time] .- c_mean[:, 1], dims = 2)[:])
sum(sum(cs[:, 101:end_time] .- c_mean[:, 1], dims = 1)[:])

# choose to animate the mean of the fluctuations evolving in time
time = Observable(1)
cs_line = @lift(cs[:, $time].^2 .- c_mean[:].^2)

fig = lines(x, cs_line, color = :blue, linewidth = 4, label = "c",
    axis = (title = @lift("Δ(x) = " * string(mag) * "cos(x) t = $(round($time, digits = 1))"),))
    ylims!(minimum(cs[:, 1:end_time].^2 .- c_mean[:].^2), maximum(cs[:, 1:end_time].^2 .- c_mean[:].^2))
#lines!(x, Δconc/maximum(Δconc)*(maximum(cs) - minimum(cs))/2 .+ (maximum(cs) + minimum(cs))/2, color = :red, linewidth = 0.5, label = L"scaled Δ(x)")
lines!(x, c_squared_mean[:] .- c_mean[:].^2, color = :black, linewidth = 1, label = L"⟨c'2⟩")
axislegend()

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

record(fig, movie_name_fluc_squared, timestamps;
        framerate = framerate) do t
    time[] = t
end