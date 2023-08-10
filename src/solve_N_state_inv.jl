using LinearAlgebra, Statistics
using ReactingTracers
using Statistics
using FFTW, GLMakie, ProgressBars
using JLD2
using PerceptualColourMaps

function specify_colours(num_colours)
  colours = cmap("Gouldian", N = num_colours)
  return colours
end

# For conditional mean equations 
function rhs!(θ̇s, θs, simulation_parameters)
    (; p, Q, us, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ) = simulation_parameters
    for (i, θ) in enumerate(θs)
        u = us[i]
        pⁱ = p[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
        # compute θ̇ in real space
        @. θ̇ = real(-u * ∂ˣθ + κΔθ - λ * θ + λ * pⁱ /c⁰  )
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end
    return nothing
end

function allocate_fields(N, M; arraytype = Array)
    # capacity
    c⁰ = arraytype(zeros(ComplexF64, N))
    # theta
    θs = [similar(c⁰) for i in 1:M]
    ∂ˣθ = similar(c⁰)
    κΔθ = similar(c⁰)
    θ̇s = [similar(c⁰) .* 0 for i in 1:M]
    return (; c⁰, θs,  ∂ˣθ, κΔθ, θ̇s)
end

function construct_p(number_of_states)
    p = ones(number_of_states)
    for i = 0:number_of_states-1
        p[i+1] = 2.0^(-(number_of_states-1))*factorial(number_of_states-1)/(factorial(i)*factorial(number_of_states-1-i))
    end
    return p
end

function u_list(number_of_states)
    N = number_of_states-1
    u = zeros(number_of_states)
    for m = 0:N
        u[m+1] = 2/sqrt(N)*(m - N/2)
    end
    return u
end

function create_qmn_matrix(m, n, number_of_states)
    N = number_of_states - 1
    if m == n
        return -N/2
    elseif m + 1 == n
        return n/2
    elseif m - 1 == n
        return (N-n)/2
    else
        return 0
    end
end

function full_qmn_matrix(number_of_states)
    matrix = zeros(number_of_states, number_of_states)
    for m = 0:number_of_states-1
        for n = 0:number_of_states-1
            matrix[m+1, n+1] = create_qmn_matrix(m, n, number_of_states)
        end
    end
    return matrix
end

number_of_states = 2
p = construct_p(number_of_states) # get the probabilities
us = u_list(number_of_states) # get the velocities
Q = full_qmn_matrix(number_of_states) # get the transition matrix

κ = 0.001

x_length = 32
field_tuples = allocate_fields(x_length, number_of_states; arraytype = Array); # allocate all the variables and say what type of variable they are


x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

# forcing conditions
magnitudes = [0.7] # [0.9, 0.5, 0.1]
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.1, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])

# define what the functions to
∂x = im * k
Δ = @. ∂x^2
P = plan_fft!(field_tuples.θs[1])
P⁻¹ = plan_ifft!(field_tuples.θs[1])


cauchy_criteria = 1e-7
dt = minimum([0.25/(x_length  * sqrt(number_of_states)), 1/(x_length^2 * κ)])


for δ in ProgressBar(magnitudes)
    for λ in ProgressBar(lambdas)
            mean_theta = Float64[]
            simulation_parameters = (; p, Q, ∂x, Δ, us, P, P⁻¹, κ, λ, field_tuples...)
            (; θ̇s, θs, c⁰) = field_tuples # extract
            @. c⁰ = 1 + δ * cos(x)
            # Initialize with c⁰ 
            [θ .= c⁰ * p[i] for (i,θ) in enumerate(θs)] # initiate the initial concentrations with the initial probabilities
            simulation_parameters = (; p, Q, ∂x, Δ, us, P, P⁻¹, κ, λ, field_tuples...)
            cauchy_criteria = 1e-7
            for i in ProgressBar(1:1000000)
            rhs!(θ̇s, θs, simulation_parameters)
            @. θs += θ̇s * dt
            if any(isnan.(θs[1]))
                println("nan")
                break
            end
            push!(mean_theta, mean(real.(sum(θs))))
            if i > 100000 
                if abs(mean_theta[i] - mean_theta[i-100])/(mean_theta[i]) < cauchy_criteria
                    println(λ, "converged")
                    println(maximum(maximum(real.(θs))))
                    break
                end
            end
        end
        cs = real.(θs)
        println(abs(mean_theta[end] - mean_theta[end-100])/(mean_theta[end]))
        println(maximum(maximum(cs)))
        save_name = "mag_" * string(δ) * "_U_" * string(1.0) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_" * string(number_of_states) * "inv.jld2"
        @save save_name cs
    end
end