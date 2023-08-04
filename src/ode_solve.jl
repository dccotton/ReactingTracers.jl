using DifferentialEquations
using ReactingTracers
using GLMakie
GLMakie.activate!(inline=false)

x_length = 512
x = nodes(x_length, a = -pi, b = pi)

# This is a function of two variables c1(t) and c2(t) as opposed
# to a PDE of of two variables c¹(x, t), c²(x, t)
function two_state!(dc, c, p, t)
    λ = 1
    mag = 0.7
    dc[1] = λ*c[1] - 2*λ*c[1]^2/(1 + mag * cos(p[1]))  - c[1]/2 + c[2]/2
    dc[2] = λ*c[2] - 2*λ*c[2]^2/(1 + mag * cos(p[2]))  + c[1]/2 - c[2]/2
    #dc[1] = -λ*c[1] .+ c[1]/2 .- c[1]/2
    #dc[2] = λ*c[1] .- c[1]/2 .+ c[1]/2
    return nothing
end

c_initial = [0.5, 0.5]
c = copy(c_initial)
tspan = (0, 10)
p = (-π, π)
prob = ODEProblem(two_state!, c_initial, tspan, p)
solver = RK4()  # Choose a solver (e.g., Tsit5, RK4, etc.)
sol = solve(prob, solver)

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
ax = Axis(fig[1, 1])

lines!(sol.t, sol[1,:])  # Plot the solution for c1 as a function of x
lines!(sol.t, sol[2,:]) 
display(fig)