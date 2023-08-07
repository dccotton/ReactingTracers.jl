using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2
using PerceptualColourMaps
using JuMP
using Ipopt

function specify_colours(num_colours)
  colours = cmap("Gouldian", N = num_colours)
  return colours
end


function adv_closure(x)
  ℱ = plan_fft(x, 1)
  ℱ⁻¹ = plan_ifft(x, 1)
  function adv(c,um,κ,k)
    ch = ℱ * c
    tmp = -im*k.*ch.*um-κ*k.*k.*ch
    dc=real.(ℱ⁻¹ * tmp) #d/dx --> =-ik
    return dc
  end
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

function steady_state_probabilities(number_of_states)
    matrix = full_qmn_matrix(number_of_states)
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
    return x_optimal
end

κ = 0.001     # "subgrid" kappa
dt = 8*2/(800*κ*1024^2)#1/5250
#dt = 4*2/(8*κ*1024^2)#1/5250 for £kappa = 0.001

# setup grid
x_length = 512
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

ox=ones(x_length,1);
velocities = [1.0] #, 10, 100, 0.1, 1] #, 1, 10]
magnitudes = [0.7] #[0.9, 0.7, 0.5, 0.1]
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
#lambdas = [100.0]
U_force = 1

state_list = [100]

for number_of_states in state_list
    adv2 = adv_closure(zeros(x_length, number_of_states))
    p_list = steady_state_probabilities(number_of_states)
for U_force in velocities
for mag in magnitudes
for λ in ProgressBar(lambdas)
  # initiate velocities
    ulist = u_list(number_of_states)
    u= randn(1, number_of_states) #[-1, +1] #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
    for m=1:number_of_states
        u[m] = ulist[m]
    end
    # initial concentration
    c= 1*ones(x_length,number_of_states) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
    dc = copy(c)
    # concentration bias
    Δconc = mag*cos.(x)
    #Δconc = Δconc*ones(1, N);
    # plotting parameters

    t=0
    tmax= 1000
    t_array = collect(t:dt:tmax);
    t_indices = round.(Int, collect(1:length(t_array)/tmax:length(t_array)))
    save_times = t_array[t_indices]

    cs=zeros(x_length, number_of_states, tmax);
    fs=zeros(x_length, number_of_states, tmax);
    for t in ProgressBar(t_array)
      if minimum(abs.(save_times .- t)) == 0
        if t > 0
          cs[:, :, round(Int, t)]= c
          if t > 10 && sum((c .- cs[:, :, round(Int, t) - 1]).^2) < 10^-5
            print(t)
            cs = cs[:, :, 1:round(Int, t)]
            break
          end
        end
      end
      c_old = copy(c)

      dc = adv2(c, U_force*u, κ, k)
      for m = 1:number_of_states
        c_qm_addition = zeros(x_length)
        for n = 1:number_of_states
            #print(create_qmn_matrix(m, n, number_of_states))
            c_qm_addition = c_qm_addition + create_qmn_matrix(m-1, n-1, number_of_states)*c_old[:, n]
        end
        c[:, m] = c[:, m] .+ dc[:, m]*dt .+ λ*dt*c[:,m] .- dt*λ*c[:,m].^2 ./((1 .+ Δconc)*p_list[m]) .+ c_qm_addition*dt
      end

      if any(isnan, c)
        print("nan")
        break
      end

    end
  
  save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_" * string(number_of_states) * ".jld2"
  @save save_name cs
end

end

end
end
mag  = 0.7
U_force = 1.0
λ = 100.0

data_folder = "data/gpu/kappa_0.001/100_state/"
N=100
save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * "_N_" * string(N) * ".jld2"
load_name = joinpath(data_folder, save_name)
@load load_name cs
colours = specify_colours(3)

# now plot the data
t_indx = Observable(1)
end_time = size(cs)[3] - 1

fig = lines(x, @lift(cs[:, 1, $t_indx]), linewidth = 4, color = colours[1], label = L"c_1",
axis = (xlabel = "x", title = @lift("t= " * string($t_indx))),)
lines!(fig.axis, x,  @lift(cs[:, 2, $t_indx]), color = colours[2], linewidth = 4, label = L"c_2")
lines!(fig.axis, x,  @lift(cs[:, 1, $t_indx] + cs[:, 2, $t_indx]), color = colours[3], linewidth = 4, label = L"c_1 + c_2")
axislegend()
#ylims!(minimum([minimum(cs), minimum(sum(cs, dims = 2))]), maximum([maximum(cs), maximum(sum(cs, dims = 2))]))

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

record(fig, "test.mp4", timestamps;
        framerate = framerate) do t
    t_indx[] = t
end

(cs[:, 1, end] + cs[:, 2, end])

c= 1*ones(512,3)