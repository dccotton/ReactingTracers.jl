using FFTW
using GLMakie
using ReactingTracers
using ProgressBars
using Statistics
using HDF5
using Random
using LinearAlgebra
GLMakie.activate!(inline=false)
Random.seed!(1234)
rng = MersenneTwister(1234);
N = 32
Nₑ = 1
𝒩 = zeros(N, N, Nₑ)
ϕ = randn(ComplexF64, N,N, Nₑ)
ϕ̇ = randn(ComplexF64, N,N, Nₑ)
x = nodes(N; a = 0, b = 2π)
k = wavenumbers(N; L = 2π)
x₁ = reshape(x, (N, 1, 1))
x₂ = reshape(x, (1, N, 1))
k₁ = reshape(k, (N, 1, 1))
k₂ = reshape(k, (1, N, 1))
∂x = im * k₁
∂y = im * k₂
Δ = @.  -(k₁^2 + k₂^2)
Δ⁻¹ = @. 1.0 / Δ 
Δ⁻¹[1] = 0.0
filter = @. 0 * exp(- (k₁^2 + k₂^2)/( 2 * (2π)^2) ) + 1.0 # 0.01

ℱ = plan_fft!(ϕ, (1,2))
ℱ⁻¹ = plan_ifft!(ϕ, (1,2))
@. ϕ = sin(x₁) * cos(x₂)
##
function auxiliary_fields(ϕ, x₁, x₂)
    ϕ∂ˣϕ = similar(ϕ) 
    Δϕ = similar(ϕ)
    s = similar(ϕ) # source term, zero for now
    @. s = 0.0 * sin(3*x₁) * sin(3*x₂)
    return (; ϕ∂ˣϕ, Δϕ, s, x₁, x₂)
end

κᵩ = 0.0 # 0.0025 for N = 32
A = 1.0
operators = (; Δ, Δ⁻¹, ∂x, ∂y, ℱ, ℱ⁻¹)
auxiliary = auxiliary_fields(ϕ, x₁, x₂)
constants = (; κᵩ, A)
parameters = (; operators, auxiliary, constants)

function rhs!(ϕ̇, ϕ, t, parameters)
    (; Δ, Δ⁻¹, ∂x, ∂y, ℱ, ℱ⁻¹) = parameters.operators
    (; ϕ∂ˣϕ, Δϕ,  s, x₁, x₂) = parameters.auxiliary
    (; κᵩ, A) = parameters.constants
    ϕ .= real.(ϕ)
    ℱ * ϕ # compute ϕ̂
    @. Δϕ = κᵩ * Δ * ϕ 
    # @. ϕ∂ˣϕ = ∂x * ϕ
    ℱ⁻¹ * Δϕ # compute Δϕ
    ℱ⁻¹ * ϕ
    # ℱ⁻¹ * ϕ∂ˣϕ
    # @. ϕ∂ˣϕ =  -ϕ + ϕ^3 + ϕ * ϕ∂ˣϕ # 10 * ϕ∂ˣϕ#
    # s .= mean(ϕ, dims = (1,2)) # zero for now
    @. ϕ̇ = real(Δϕ - ϕ)#  - s)
    return nothing
end

function score(ϕ, parameters)
    s = copy(ϕ)
    rhs!(s, ϕ, [0.0], parameters)
    return real.(s)
end
struct RungeKutta4{S, T, U}
    k⃗::S
    x̃::T
    xⁿ⁺¹::T
    t::U
end
RungeKutta4(ϕ) = RungeKutta4([similar(ϕ) for i in 1:4], similar(ϕ), similar(ϕ), [0.0])
rk = RungeKutta4(ϕ)
function (runge_kutta::RungeKutta4)(rhs!, x, parameters, dt)
    @inbounds let
        @. runge_kutta.x̃ = x
        rhs!(runge_kutta.k⃗[1], runge_kutta.x̃, runge_kutta.t[1], parameters)
        @. runge_kutta.x̃ = x + runge_kutta.k⃗[1] * dt / 2
        @. runge_kutta.t += dt / 2
        rhs!(runge_kutta.k⃗[2], runge_kutta.x̃, runge_kutta.t[1], parameters)
        @. runge_kutta.x̃ = x + runge_kutta.k⃗[2] * dt / 2
        rhs!(runge_kutta.k⃗[3], runge_kutta.x̃, runge_kutta.t[1], parameters)
        @. runge_kutta.x̃ = x + runge_kutta.k⃗[3] * dt
        @. runge_kutta.t += dt / 2
        rhs!(runge_kutta.k⃗[4], runge_kutta.x̃, runge_kutta.t[1], parameters)
        @. runge_kutta.xⁿ⁺¹ = x + (runge_kutta.k⃗[1] + 2 * runge_kutta.k⃗[2] + 2 * runge_kutta.k⃗[3] + runge_kutta.k⃗[4]) * dt / 6
    end
    return nothing
end
##
kernel(x, y; σ = 2π/32) = exp(-norm(sin.((x - y)/2))^2 / (2σ^2))
xy = [[x, y] for x in x₁, y in x₂][:]
K = [kernel(xy[i], xy[j]) for i in 1:N^2, j in 1:N^2]
Knew = Symmetric(K + 0.1 * I)
Σ = cholesky(Knew).U # Σ' * Σ
##
randn!(rng, 𝒩);
@. ϕ = 𝒩 
##
dt = 0.05 * 32 / N 
ϵ = 0.5
xs = Float64[]
xs1 = Float64[]
ϕs = typeof(ϕ)[]
tend = floor(Int,  10^5 / 32 * N)
for i in ProgressBar(1:tend)
    randn!(rng, 𝒩) 
    # 𝒩 .= reshape(Σ' * reshape(𝒩, (N^2, Nₑ)), (N, N , Nₑ))
    # 𝒩 .*= filter
    tmp = real.(ifft(𝒩, (1,2))) * sqrt(m *n)
    rk(rhs!, ϕ, parameters, dt)
    ϕ .= rk.xⁿ⁺¹ .+ ϵ * sqrt(dt) * tmp
    if any(isnan.(ϕ))
        @info "nan detected"
        break
    end
    if i%10==0
        if i > tend/10
            push!(ϕs, copy(ϕ))
            push!(xs1, real.(ϕ[1]))
            push!(xs, real.(ϕ)[:]...)
        end
    end
end
##
fig = Figure() 
colorrange = (-0.1,0.1) ./ 5
ax1 = Axis(fig[1,1]; title = "ensemble member 1")
heatmap!(ax1, real.(ϕ)[:,:,1], colormap = :balance, interpolate = false)
ax2 = Axis(fig[1,2]; title = "ensemble member 2")
heatmap!(ax2, real.(ϕ)[:,:,2]; colormap = :balance)
ax21 = Axis(fig[2,1]; title = "histogram of all pixels")
hist!(ax21, xs, bins = 100, color = :black)
ax22 = Axis(fig[2,2]; title = "histogram of pixel 1 for ensemble member 1")
hist!(ax22, xs1, bins = 100, color = :black)
display(fig)
##
fig = Figure() 
ax1 = Axis(fig[1,1])
sl = Slider(fig[2,1], range = 1:length(ϕs))
sl2 = Slider(fig[1:2, 2:3], range = 1:Nₑ, horizontal = false)
obs = sl.value
obs2 = sl2.value
field = @lift(real.(ϕs[$obs])[:,:,$obs2])
colorrange = (-0.1,0.1) .* 10
heatmap!(ax1, field, colorrange = colorrange, colormap = :balance, interpolate = false)
display(fig)
##
m, n, _ = size(ϕs[1])
ℓ = length(ϕs)
ϕ_save = zeros(Float64, m, n, 1, ℓ * Nₑ)
score_save = copy(ϕ_save)
for i in ProgressBar(1:ℓ)
    for j in 1:Nₑ
        ϕ_save[:,:,1, (i-1)*Nₑ + j] .= real.(ϕs[i])[:,:,j]
        score_save[:,:,1, (i-1)*Nₑ + j] .= score(ϕs[i], parameters)[:,:,j]
    end
end
##
size(ϕ_save)
ϕ_data = reshape(ϕ_save, (m * n, ℓ * Nₑ))
Σ²ₑ = cov(ϕ_data')
Σ² = Knew * ϵ^2  / 2
luΣ² = lu(Σ²)
scorish = reshape(luΣ² \ ϕ_data[:,1], (m, n))
##
norm(Σ²ₑ - Σ²) / norm(Σ²ₑ)