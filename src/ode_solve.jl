using DifferentialEquations
using ReactingTracers
using GLMakie
GLMakie.activate!(inline=false)

x_length = 512
x = nodes(x_length, a = -pi, b = pi)

# This is a function of two variables c1(t) and c2(t) as opposed
function two_state_p!(dp, p, t, param)
    dp[1] =  - p[1]/2 + p[2]/2
    dp[2] =   p[1]/2 - p[2]/2

    return nothing
end

function three_state_p!(dp, p, t, param)
    dp[1] =  -p[1] + p[2]/2
    dp[2] =   -p[2] + p[1] + p[3]
    dp[3] = -p[3] + p[2]/2
    return nothing
end


c_initial = [0.8, 0.2]
tspan = (0, 10)
p = (-π, π)
prob = ODEProblem(two_state_p!, c_initial, tspan, p)
solver = RK4()  # Choose a solver (e.g., Tsit5, RK4, etc.)
sol = solve(prob, solver)

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
ax = Axis(fig[1, 1])

lines!(sol.t, sol[1,:])  # Plot the solution for c1 as a function of x
lines!(sol.t, sol[2,:], linestyle = :dash) 
display(fig)

p_initial_2 = [0.5, 0.5]
p_initial_3 = [0.3, 0.3, 0.4]

tspan = (0, 10)
#prob = ODEProblem(two_state_p!, p_initial_2, tspan, p)
prob = ODEProblem(three_state_p!, p_initial_3, tspan, p)
solver = RK4()  # Choose a solver (e.g., Tsit5, RK4, etc.)
sol = solve(prob, solver)

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
ax = Axis(fig[1, 1])
lines!(sol.t, sol[1,:], label = "p1")  # Plot the solution for c1 as a function of x
lines!(sol.t, sol[2,:], linestyle = :dash, label = "p2") 
lines!(sol.t, sol[3,:], linestyle = :dot, label = "p3") 
axislegend()
display(fig)