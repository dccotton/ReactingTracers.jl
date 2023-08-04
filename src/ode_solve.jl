using DifferentialEquations
using ReactingTracers
using GLMakie
GLMakie.activate!(inline=false)

x_length = 512
x = nodes(x_length, a = -pi, b = pi)

λ = 1
mag = 0.7


function two_state(dc, c, x)
    c0  = 1 + mag*cos(x)
    dc[1] = -λ*c[1] + 2*λ*c[1]^2/c0 + c[1]/2 - c[2]/2
    dc[2] = λ*c[2] - 2*λ*c[2]^2/c0 - c[2]/2 + c[1]/2
    #dc[1] = -λ*c[1] .+ c[1]/2 .- c[1]/2
    #dc[2] = λ*c[1] .- c[1]/2 .+ c[1]/2
end

c_initial = [0.5, 0.5]
c = c_initial
xspan = (x[1], x[end])
prob = ODEProblem(two_state, c_initial, xspan)
solver = RK4()  # Choose a solver (e.g., Tsit5, RK4, etc.)
sol = solve(prob, solver)

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
ax = Axis(fig[1, 1])

lines!(sol.t, sol[1,:])  # Plot the solution for c1 as a function of x
lines!(sol.t, sol[2,:]) 