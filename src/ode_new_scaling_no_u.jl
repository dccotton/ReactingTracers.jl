using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2

# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
function diff_closure(x)
  ℱ = plan_fft(x)
  ℱ⁻¹ = plan_ifft(x)
  function diff(c, κ, k, dt, Δ, λ)
    ch = ℱ * c
    c0 = 1 .+ Δ
    λc = λ * c .* (1 .- c./c0)

    ch = (ch/dt .+ ℱ * λc)./(1/dt .+ κ*k.^2)
    c_new = real.(ℱ⁻¹ * ch)
    return c_new
  end
end

x_length = 512
dt=2/(0.001*x_length^2) # need 1/dt > κ(x_length/2)^2
#dt = 2/(0.01*x_length^2) # if kappa = 0 can't have infinite timestep

x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

diff2 = diff_closure(zeros(x_length))

magnitudes = [0.5, 0.9] #[0.7, 0.1, 0.01] #[0.1, 0.01] #, 0.1, 0.1]
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
kappas = [1.0] #0.1, 0.01, 0.001, 0.025, 0.05, 0.15, 0.25, 0.5] # 1, 10, 

for κ in kappas
for mag in magnitudes
    for λ in ProgressBar(lambdas)
        
        # initial concentration
        c= ones(length(x)); #+ 0.001*randn(1024,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
        # concentration bias
        Δconc = mag*cos.(x)
        #Δconc = Δconc*ones(1, 10);
        
        # plotting parameters
        t=0
        tmax=3000
        t_array = collect(t:dt:tmax);
        t_indices = round.(Int, collect(1:length(t_array)/tmax:length(t_array)))
        save_times = t_array[t_indices]

        cs=zeros(x_length, tmax);
        fs=zeros(x_length, tmax);

        for t in ProgressBar(t_array)
        if minimum(abs.(save_times .- t)) == 0
            if t > 0
                cs[:, round(Int, t)]= c
                if any(isnan, c)
                    print("nan error")
                    break
                elseif t > 10 && sum(c-cs[:, round(Int, t)-1]).^2 < 10^(-10)
                    cs[:, round(Int, t)]= c
                    cs = cs[:, 1:round(Int, t)]
                    print(t)
                    break
                end
            end
            

        end
        c .= diff2(c, κ, k, dt, Δconc, λ);
        
        end
    cf=mean(cs[:,201:end],dims = 2);
    
    save_name = "mag_" * string(mag) * "_kappa_" * string(κ) * "_lambda_" * string(λ) * "_nou_FT.jld2"

    @save save_name cs cf c
    end

end
end

κ = 1.0
λ = 1.0
mag = 0.7
load_name = "mag_" * string(mag) * "_kappa_" * string(κ) * "_lambda_" * string(λ) * "_nou_FT.jld2"

@load load_name cs cf c
# now plot the data
t_indx = Observable(1)
end_time = size(cs)[2] - 1

fig = lines(x, @lift(cs[:, $t_indx]), linewidth = 4, color = colours[1], label = L"c_1",
axis = (xlabel = "x", title = @lift("t= " * string($t_indx))),)
axislegend()
#ylims!(minimum([minimum(cs), minimum(sum(cs, dims = 2))]), maximum([maximum(cs), maximum(sum(cs, dims = 2))]))

framerate = 10
timestamps = range(start = 1, stop = end_time, step=1)

record(fig, "test.mp4", timestamps;
        framerate = framerate) do t
    t_indx[] = t
end