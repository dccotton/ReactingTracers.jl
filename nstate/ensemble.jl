using FFTW
using GLMakie
using ReactingTracers
using ProgressBars
using Statistics
using HDF5
using Random
using LinearAlgebra
using MarkovChainHammer
GLMakie.activate!(inline=false)
Random.seed!(1234)
rng = MersenneTwister(1234);

N  = 32
Nₛ = 3
Nₑ = 10^2
Nₜ = 10^7
timesteps = 10^5

Q = ou_transition_matrix(Nₛ)
u⃗ = ou_velocity_fields(Nₛ)
Λ, V =  eigen(Q)
p = steady_state(Q)

##
function rhs!(θ̇, θ, t, simulation_parameters)
    (; u, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ, θ²) = simulation_parameters
    # dynamics
    P * θ # in place fft
    # ∇θ
    @. ∂ˣθ = ∂x * θ
    # κΔθ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
    # compute θ̇ in real space
    @. θ² = θ .^ 2
    @. θ̇ = real(-u * ∂ˣθ + κΔθ + λ * (θ - θ² / c⁰))
    return nothing
end

function ql_rhs!(θ̇, θ, t, simulation_parameters)
    (; u, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ, θ²) = simulation_parameters
    # dynamics
    P * θ # in place fft
    # ∇θ
    @. ∂ˣθ = ∂x * θ
    # κΔθ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
    # compute θ̇ in real space
    θ² .= sum(θ, dims = 2) .^2 + 2 * θ .* sum(θ, dims = 2) .+ (θ .-sum(θ, dims = 2)) .^2 
    @. θ̇ = real(-u * ∂ˣθ + κΔθ + λ * (θ - θ² / c⁰))
    return nothing
end

##
# generate ensemble timeseries

λ = 1.0
λs = [1.0]# [0.01, collect(range(0.1, 3, length = 11))..., 5, 10]

U = 1.0
δ = 0.7
κ = 1e-3

x = reshape(nodes(N), (N, 1))
k  = reshape(wavenumbers(N), (N, 1))
∂x = im * k
Δ = -k.^2

c⁰ = @.  1 + δ * cos(x)
θ = zeros(N, Nₑ) .+ im
θ .= c⁰
∂ˣθ = similar(θ)
κΔθ = similar(θ)
θ̇ = similar(θ)
θ² = similar(θ)
P = plan_fft!(θ,  1)
P⁻¹ = plan_ifft!(θ, 1)
u = reshape(zeros(Nₑ), (1, Nₑ))
runge_kutta = RungeKutta4(θ)

for λ ∈ ProgressBar(λs)
    simulation_parameters = (; u, c⁰, ∂ˣθ, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ, θ²) 
    ##
    cfl = 0.9
    dt = cfl * minimum([ (x[2]-x[1]) / maximum(u⃗), 1/λ, (x[2]-x[1])^2/κ])
    ms = ones(Int, Nₑ, timesteps)
    for i in ProgressBar(1:Nₑ)
        ms[i, :] .= generate(Q, timesteps; dt = dt) 
    end
    ##
    for i in ProgressBar(1:timesteps)
        u .= U * reshape(u⃗[ms[:, i]], (1, Nₑ))
        runge_kutta(rhs!, θ, simulation_parameters, dt)
        θ .= runge_kutta.xⁿ⁺¹
        if any(isnan.(θ[1]))
            println("nan")
            break
        end
    end

    ##
    fig = Figure() 
    ax11 = Axis(fig[1,1])
    for i in 1:100
        lines!(ax11, x[:], real.(θ[:, i]), color = (:blue, 0.1))
    end
    lines!(ax11, x[:], real.(c⁰[:, 1]), color = :black, linewidth = 3, linestyle = :dash)
    ylims!(ax11, (1-δ,   1+δ))
    lines!(ax11, x[:], real.(mean(θ, dims = 2))[:], color = :red, linewidth = 3)
    lines!(ax11, x[:], real.(mean(θ, dims = 2) .* 0 .+ sqrt(1- δ^2))[:], color = :black, linewidth = 6, linestyle = :dot)
    ylims!(ax11, (1-δ,   1+δ))
    ax12 = Axis(fig[1,2])
    lines!(ax12, x[:],  mean((real.(θ) .- real.(mean(θ, dims = 2))).^2, dims = 2)[:] )
    ylims!(ax12, (0,   sqrt(1-δ^2)))
    Θs = []
    ax21 = Axis(fig[2, 1])
    for i in 1:Nₛ
        push!(Θs, real.(mean(θ[:, ms[:, end] .== i], dims = 2)[:]))
        lines!(ax21, x[:], Θs[i])
    end
    # ylims!(ax12, (1-δ,   1+δ))
    ax22 = Axis(fig[2, 2])
    lines!(ax22, real.(mean(∂ˣθ, dims = 2)[:]), real.(mean(u .* θ, dims = 2)[:]), color = :black)
    # ylims!(ax22, (-0.5, 0.5))
    # xlims!(ax22, (-0.5, 0.5))
    display(fig)
    ##
    save("λ_$(λ)_δ_$(δ)_n_$(Nₛ)_quick_fig.png", fig)
    ##
    hfile = h5open("λ_$(λ)_δ_$(δ)_n_$(Nₛ)_quick_data.hdf5", "w")
    hfile["u"] = u
    hfile["θ"] = real.(θ)
    hfile["dxθ"] = real.(∂ˣθ)
    hfile["mean"] = real.(mean(θ, dims = 2))[:]
    hfile["variance"] = mean((real.(θ) .- real.(mean(θ, dims = 2))).^2, dims = 2)[:]
    hfile["gradient"] =  real.(mean(∂ˣθ, dims = 2)[:])
    hfile["flux"] = real.(mean(u .* θ, dims = 2)[:])
    close(hfile)
end