using ReactingTracers

mag = 0.7
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0])
data_folder = "data/gpu/kappa_0.001"
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

# Fitting the data to the tanh Function
fit_result = fit_tanh(xdata, ydata)

mat = rand(4, 8)
mat[1,:] = [1, 2, 3, 4, 5, 6, 7, 8]
mat
conc = [1, 2, 3, 4] .*ones(4, 8)
conc.*mat
mat[1, :].*conc
sum(mat[1,:].*conc, dims = 1)

mat[1, :]' .* mat

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

matrix = full_qmn_matrix(4)
c = ones(10, 4)
matrix[1, :]'.*c