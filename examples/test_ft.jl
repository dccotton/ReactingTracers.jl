test = 1 .+ 2*cos.(x) .+ 4*cos.(2*x)
ft_test = fft(test)


as = zeros(Int(x_length/2))
bs = zeros(Int(x_length/2))
y = zeros(x_length)
for indx = 1:Int(x_length/2)-1
    print(indx)
    if indx % 2 == 0
        #print(indx)
        as[indx] = real(ft_test[indx+1])*2/x_length
        bs[indx] = -imag(ft_test[indx+1])*2/x_length
    else
        as[indx] = -real(ft_test[indx+1])*2/x_length
        bs[indx] = imag(ft_test[indx+1])*2/x_length
    end
    global y
    y = y .+ as[indx]*cos.(indx*x) .+ bs[indx]*sin.(indx*x)
end
y = y .+ real(ft_test[1])/x_length

print(fit_result.param)

# attempt tanh fit
using LsqFit
using GLMakie
GLMakie.activate!(inline=false)

# Define the tanh function
function tanh_function(x, p)
    a, b, c, d = p
    return a .* tanh.(b .* (x .- c)) .+ d
end

function tanh_function_fixed(x, a)
    return 0.5 .* tanh.(a .* x) .+ 0.5
end

mag = 0.9
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0])
data_folder = "data/gpu/kappa_0.001/code_fixes"
stat = zeros(length(lambdas))
for l_indx in 1:length(lambdas)
    data_name = data_name = "mag_" * string(mag) * "_U_" * string(1.0) * "_lambda_" * string(lambdas[l_indx]) * ".jld2"

    # Concatenate the folder and file name to get the full path
    load_name = joinpath(data_folder, data_name)
    @load load_name c_mean flux_mean c_squared_mean gc
    stat[l_indx] = (mean(c_mean[:]) - (1-mag^2)^0.5)/(1 - (1-mag^2)^0.5)
    stat[l_indx] = (maximum(c_mean) - minimum(c_mean))/(2*mag)
end
# Sample data
#xdata = range(-5, 5, 100)
#ydata = 2 .* tanh.(2 .* (xdata .- 1)) .+ 0.5 .+ 0.1 .* randn(length(xdata))
xdata = log10.(lambdas)
ydata = stat

# Initial parameters guess [a, b, c, d]
#p0 = [1.0, 1.0, 1.0, 1.0]
#p0 = [1.0, 1.0, 0.0 ,0.0]
p0 = [1.0]

# Fitting the data to the tanh function
fit_result = curve_fit(tanh_function_fixed, xdata, ydata, p0)

# Extracting the fitted parameters
best_parameters = fit_result.param

# Generating the fitted curve using the best parameters
yfit = tanh_function_fixed(xdata, best_parameters)

# Plotting the original data and the fitted curve
fig = Figure(resolution = (3024, 1964))
ax = Axis(fig[1, 1])
#lines!(stat)

lines!(xdata, ydata, label="Data")
lines!(xdata, yfit, label="Fitted Curve")
xlabel!(ax, "x")
ylabel!(ax, "y")
#title!("Tanh Function Fitting")
axislegend()

# Display the plot