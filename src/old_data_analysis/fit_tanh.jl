using LsqFit
using GLMakie
GLMakie.activate!(inline=false)

# Define the tanh function
function tanh_function_fixed(x, a)
    return 0.5 .* tanh.(a .* x) .+ 0.5
end

# Define the tanh function
function tanh_function(x, p)
    a, b, c, d = p
    return a .* tanh.(b .* (x .- c)) .+ d
end

function tanh_function_semi_fixed(x, p)
    b, c = p
    return 0.5 .* tanh.(b .* (x .- c)) .+ 0.5
end

mag = 0.7
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0, 100.0])
data_folder = "data/gpu/kappa_0.001/code_fixes"

stat = zeros(length(lambdas))
for l_indx in 1:length(lambdas)
    data_name = data_name = "mag_" * string(mag) * "_U_" * string(1.0) * "_lambda_" * string(lambdas[l_indx]) * ".jld2"

    # Concatenate the folder and file name to get the full path
    load_name = joinpath(data_folder, data_name)
    @load load_name c_mean flux_mean c_squared_mean gc
    stat[l_indx] = (mean(c_mean[:]) - (1-mag^2)^0.5)/(1 - (1-mag^2)^0.5)
    #stat[l_indx] = (maximum(c_mean) - minimum(c_mean))/(2*mag)
end
# Sample data
xdata = log10.(lambdas)
ydata = stat

# Initial parameters guess [a, b, c, d]
p0_fix = [1.0]
p0_fluid = [0.5, 1.0, 0.0, 0.5]
p0_semi = [1.0, 0.0]

# Fitting the data to the tanh function
fit_result_fix = curve_fit(tanh_function_fixed, xdata, ydata, p0_fix)
fit_result_fluid = curve_fit(tanh_function, xdata, ydata, p0_fluid)
fit_result_semi = curve_fit(tanh_function_semi_fixed, xdata, ydata, p0_semi)

# Extracting the fitted parameters
best_parameters_fix = fit_result_fix.param
best_parameters_fluid = fit_result_fluid.param
best_parameters_semi = fit_result_semi.param

# Generating the fitted curve using the best parameters
yfit_fix = tanh_function_fixed(xdata, best_parameters_fix)
yfit_fluid = tanh_function(xdata, best_parameters_fluid)
yfit_semi = tanh_function_semi_fixed(xdata, best_parameters_semi)

# Plotting the original data and the fitted curve
fig = Figure(resolution = (3024, 1964))
ax = Axis(fig[1, 1])
#lines!(stat)

lines!(xdata, ydata, label="Data")
lines!(xdata, yfit_fix, label="Fitted Curve fixed")
lines!(xdata, yfit_fluid, label="Fitted Curve fluid")
lines!(xdata, yfit_semi, label="Fitted Curve semi")
#title!("Tanh Function Fitting")
axislegend()

# Display the plot