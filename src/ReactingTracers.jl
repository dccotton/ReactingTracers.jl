module ReactingTracers

export nodes, wavenumbers, RungeKutta4, ou_transition_matrix, ou_velocity_fields

"""
nodes(n; a = 0, b = 2π)
# Description
- Create a uniform grid of points for periodic functions
# Arguments
- `N`: integer | number of evenly spaced points 
# Keyword Arguments
- `a`: number | starting point of interval [a, b)
- `b`: number | ending point of interval [a, b)
# Return
- `g`: array | an array of points of evenly spaced points from [a, b)
"""
function nodes(N; a = 0, b = 2π)
    return (b-a) .* collect(0:(N-1))/N .+ a
end


"""
wavenumbers(N; L = 2π)
# Description
- Create wavenumbers associated with the domain of length L
# Argumentsfff
- `N`: integer | number of wavevectors
# Keyword Arguments
- `L`: number | length of interval [a, b), L = b-a
# Return
- `wavenumbers`: array | an array of wavevectors
"""
function wavenumbers(N; L = 2π, edgecase = false)
    up = collect(0:1:N-1)
    down = collect(-N:1:-1)
    indices = up
    indices[div(N,2)+1:end] = down[div(N,2)+1:end]
    indices[1] = 0 # edge case
    if edgecase
    indices[div(N,2)+1] = 0 # edge case
    end
    wavenumbers = 2π / L .* indices
    return wavenumbers
end

struct RungeKutta4{S, T, U}
    k⃗::S
    x̃::T
    xⁿ⁺¹::T
    t::U
end
RungeKutta4(ϕ) = RungeKutta4([similar(ϕ) for i in 1:4], similar(ϕ), similar(ϕ), [0.0])
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

function ou_transition_matrix_raw(n)
    Mⱼₖ = zeros(n + 1, n + 1)
    δ(j, k) = (j == k) ? 1 : 0

    for j in 0:n, k in 0:n
        jj = j + 1
        kk = k + 1
        Mⱼₖ[jj, kk] =
            (-n * δ(j, k) + k * δ(j + 1, k) + (n - k) * δ(j - 1, k)) / 2
    end
    return Mⱼₖ
end

function ou_velocity_fields_raw(N)
    Δx = 2 / √N
    uₘ = [Δx * (i - N / 2) for i in 0:N]
    return uₘ
end

ou_transition_matrix(n) = ou_transition_matrix_raw(n-1)
ou_velocity_fields(N) = ou_velocity_fields_raw(N-1)


end # module ReactingTracers
