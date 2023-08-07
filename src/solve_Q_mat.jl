using JuMP
using Ipopt
number_of_states = 10

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

matrix = zeros(number_of_states, number_of_states)
for m = 0:number_of_states-1
    for n = 0:number_of_states-1
        matrix[m+1, n+1] = create_qmn_matrix(m, n, number_of_states)
    end
end
matrix

b = zeros(number_of_states)

# Create a JuMP model
model = Model(optimizer_with_attributes(Ipopt.Optimizer)) #, "msg_lev" => GLPK.MSG_OFF))

# Define the variables x_i
n = size(matrix, 2)
@variable(model, x[1:n] >= 0)

# Set the sum constraint
@constraint(model, sum(x) == 1)

# Define the objective function: minimize ||Ax - b||
@objective(model, Min, sum((matrix * x - b).^2))

# Solve the optimization problem
optimize!(model)

# Get the optimal solution
x_optimal = value.(x)

println("Optimal solution x: ", x_optimal)
