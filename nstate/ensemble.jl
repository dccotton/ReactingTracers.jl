using FFTW
using GLMakie
using ReactingTracers
using ProgressBars
using Statistics
using HDF5
using Random
using LinearAlgebra
using MarkovChainHammer.Trajectory: generate
using MarkovChainHammer.TransitionMatrix: steady_state
GLMakie.activate!(inline=false)
Random.seed!(1234)
rng = MersenneTwister(1234);

N = 32
Nₑ = 3

Q = ou_transition_matrix(Nₑ)
u⃗ = ou_velocity_fields(Nₑ)
Λ, V =  eigen(Q)
p = steady_state(Q)
V[:, end] .= p
[V[:, end-i] ./= V[1, end-i] for i in 1:Nₑ-1]
W = inv(V)

𝒩 = zeros(N, N, Nₑ)
ϕ = randn(ComplexF64, N,N, Nₑ)
ϕ̇ = randn(ComplexF64, N,N, Nₑ)
x = nodes(N; a = 0, b = 2π)
k = wavenumbers(N; L = 2π)
x₁ = reshape(x, (N, 1, 1))
x₂ = reshape(x, (1, N, 1))
k₁ = reshape(k, (N, 1, 1))
k₂ = reshape(k, (1, N, 1))
Δ = @.  -(k₁^2 + k₂^2)
Δ⁻¹ = @. 1.0 / Δ 
Δ⁻¹[1] = 0.0

ℱ = plan_fft!(ϕ, (1,2))
ℱ⁻¹ = plan_ifft!(ϕ, (1,2))
@. ϕ = sin(x₁) * cos(x₂)
##
function auxiliary_fields(ϕ, x₁, x₂)
    ϕ³ = similar(ϕ) 
    Δϕ = similar(ϕ)
    s = similar(ϕ) # source term, zero for now
    @. s = 0.0 * sin(3*x₁) * sin(3*x₂)
    return (; ϕ³, Δϕ, s, x₁, x₂)
end

κᵩ = 0.0025 # 0.0025 for N = 32
A = 1.0
operators = (; Δ, Δ⁻¹, ℱ, ℱ⁻¹)
auxiliary = auxiliary_fields(ϕ, x₁, x₂)
constants = (; κᵩ, A)
parameters = (; operators, auxiliary, constants)


function rhs!(ϕ̇, ϕ, t, parameters)
    (; Δ, Δ⁻¹,ℱ, ℱ⁻¹) = parameters.operators
    (; ϕ³, Δϕ,  s, x₁, x₂) = parameters.auxiliary
    (; κᵩ, A) = parameters.constants
    ϕ .= real.(ϕ)
    ℱ * ϕ # compute ϕ̂
    @. Δϕ = κᵩ * Δ * ϕ 
    ℱ⁻¹ * Δϕ # compute Δϕ
    ℱ⁻¹ * ϕ
    @. ϕ³ = ϕ^3
    s .=  0.0 * mean(ϕ, dims = (1,2)) # zero for now
    @. ϕ̇ = real( Δϕ + A * (ϕ - ϕ³) - s)
    return nothing
end

function score(ϕ, parameters)
    s = copy(ϕ)
    rhs!(s, ϕ, [0.0], parameters)
    return real.(s)
end

rk = RungeKutta4(ϕ)