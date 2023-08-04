using Statistics
using FFTW, ProgressBars
using ReactingTracers
using JLD2
##
# timestep minimum
#1/\kappa k^2
##
# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
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

function rk4(h, c, u, u_t0ph_2, u_t0ph, κ, k, U_force)
    k1 = adv2(c, U_force*u, κ, k)
    k2 = adv2(c + h/2*k1, U_force*u_t0ph_2, κ, k)
    k3 = adv2(c + h/2*k2, U_force*u_t0ph_2, κ, k)
    k4 = adv2(c + h*k3, U_force*u_t0ph, κ, k)
    dc = h/6*(k1+2*k2+2*k3+k4)
    return dc
end

κ = 0.001     # "subgrid" kappa
dt = 4*2/(8*κ*1024^2)#1/5250

N=250
av=1/N

# setup grid
x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

adv2 = adv_closure(zeros(x_length, N))

U_force = 1.0 #, 10, 100, 0.1, 1] #, 1, 10]
mag = parse(Float64, ARGS[2]) #[0.9, 0.7, 0.5, 0.1]
λ = parse(Float64, ARGS[1])


# initiate velocities
u=randn(1,N) #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
ind = findall(u -> abs(u) >= 5, u)
u[ind] .= 5

# initial concentration
c= ones(x_length,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
dc = copy(c)
# concentration bias
Δconc = mag*cos.(x)
Δconc = Δconc*ones(1, N);
# plotting parameters

t=0
tmax= 2000
t_array = collect(t:dt:tmax);
t_indices = round.(Int, collect(1:length(t_array)/tmax:length(t_array)))
save_times = t_array[t_indices]

cs=zeros(x_length);
c²s = zeros(x_length);
fs=zeros(x_length);

i = 1
t = t_array[i]
for (i,t) in ProgressBar(enumerate(t_array))
    if minimum(abs.(save_times .- t)) == 0
    if t > 0
        if t > save_times[500]
            cs .+= mean(c, dims=2)
            c²s .+= mean(c .^ 2, dims=2)
            fs .+= (1 / N) * c * u'
            num_avg .+= 1
        end
        if i % 100 == 0
            save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * string(num_avg) * ".jld2"
            @save save_name cs/num_avg fs /num_avg c²s / num_avg real(ifft(im*k[:,1].*fft(c_mean)))
        end
    end
    u_t0ph_2 = u .- dt/2*u .+ sqrt(2*dt)*randn(1,N)
    u_t0ph = u_t0ph_2 .- dt/2*u_t0ph_2 .+ sqrt(2*dt)*randn(1,N)
    dc .= rk4(dt, c, u, u_t0ph_2, u_t0ph, κ, k, U_force)
    @. c = c +  dc + λ*dt*(c)*(1-c/(1 + Δconc))
    u = u_t0ph
    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5
    if any(isnan, c)
        print("nan")
        break
    end
end

c_mean = cs / num_avg; #<c>
flux_mean = fs /num_avg #<uc>
c_squared_mean = c²s / num_avg; #<c^2>
gc=real(ifft(im*k[:,1].*fft(c_mean)));

save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * ".jld2"
@save save_name c_mean flux_mean c_squared_mean gc
