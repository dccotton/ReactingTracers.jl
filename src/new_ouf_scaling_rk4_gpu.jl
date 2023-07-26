using Statistics
using FFTW, ProgressBars
using ReactingTracers
using JLD2
using CUDA
##
# timestep minimum
#1/\kappa k^2
##
# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
Array_Type = CuArray

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
dt = 4*2/(κ*1024^2)#1/5250

N=250
av=1/N

# setup grid
x_length = 1024
x = Array_Type(nodes(x_length, a = -pi, b = pi))
k  = Array_Type(wavenumbers(x_length))

adv2 = adv_closure(Array_Type(zeros(x_length, N)))

ox=Array_Type(ones(x_length,1));
velocities = [1.0] #, 10, 100, 0.1, 1] #, 1, 10]
magnitudes = [0.5] #[0.9, 0.7, 0.5, 0.1]
lambdas = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 2.5]
#lambdas = [1, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 2.5, 3, 5, 6, 7, 8, 9] #[0.01, 0.1, 1, 10, 100, 0.5, 1.5, 2]
U_force = 1

for U_force in velocities
for mag in magnitudes
for λ in ProgressBar(lambdas)
  # initiate velocities
    u=randn(1,N) #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5
    u = Array_Type(u)

    # initial concentration
    c= Array_Type(ones(x_length,N)) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
    dc = copy(c)
    # concentration bias
    Δconc = Array_Type(mag*cos.(x))
    Δconc = Δconc*Array_Type(ones(1, N));
    # plotting parameters

    t=0
    tmax= 200#1000
    t_array = collect(t:dt:tmax);
    t_indices = round.(Int, collect(1:length(t_array)/tmax:length(t_array)))
    save_times = t_array[t_indices]

    #fig = Figure()
    #ax = Axis(fig[1, 1], xlabel = L"x",
    #  xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    #  xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
    #lines!(x,mean(c,dims=2)[:], label = L"\overline{c}")
    #lines!(x,(1/N)*c*u'[:], label = L"\overline{uc}") #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
    #lines!(x,Δconc[:, 1], label = L"\Delta(x)")
    #axislegend()#position = :rt, bgcolor = (:grey90, 0.25));
    #fig

    cs=Array_Type(zeros(x_length));
    c²s = Array_Type(zeros(x_length))
    fs=Array_Type(zeros(x_length));
    i = 1
    t = t_array[i]
    for (i, t) in ProgressBar(enumerate(t_array))
      if minimum(abs.(save_times .- t)) == 0
        if t > 0
          #f2 = Figure()
          #ax = Axis(f2[1, 1])
          #lines!(ax, x,mean(c,dims=2)[:])
          #lines!(ax, x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
          #display(f2)
          if t > save_times[210]
            # accumulate locally averaged concentration and flux
            cs .+= mean(c, dims=2)
            c²s .+= mean(c .^ 2, dims=2)
            fs .+= (1 / N) * c * u'
            num_avg += 1
        end
        end
      end
      #dc .= rk4(dt, c, U_force*u, κ, k)
      𝒩 = Array_Type(randn(1, N))
      u_t0ph_2 = Array_Type(u .- dt/2*u .+ sqrt(2*dt)*𝒩)
      u_t0ph = Array_Type(u_t0ph_2 .- dt/2*u_t0ph_2 .+ sqrt(2*dt)*𝒩)
      dc .= rk4(dt, c, u, u_t0ph_2, u_t0ph, κ, k, U_force)
      #print(maximum(dc))
      #print("u_max = ", maximum(u_t0ph_2))
      @. c = c +  dc + λ*dt*(c)*(1-c/(1 + Δconc))
      u = u_t0ph

      #@. c = c +  dc*dt + λ*dt*(c)*(1-c/(1 + Δconc))
      if any(isnan, c)
        print("nan")
        break
      end

      #𝒩 = randn(1,N)
      #@. u = u - dt*u+sqrt(2)*𝒩
      ind = findall(u -> abs(u) >= 5, u)
      u[ind] .= 5
    end

    c_mean = Array(cs / num_avg) # mean(cs[:,201:end-1],dims = 2); #<c>
    flux_mean = Array(fs / num_avg)# mean(fs[:,201:end-1],dims = 2); #<uc>
    c_squared_mean = Array(c²s / num_avg)# mean(cs[:,201:end-1].^2,dims = 2); #<c>
    gc = real(ifft(im * Array(k)[:, 1] .* fft(c_mean)))

    #c_mean = mean(cs,dims = 2); #<c>
    #flux_mean =mean(fs, dims = 2); #<uc>
    #c_squared_mean = mean(cs.^2,dims = 2); #<c>
    #gc=real(ifft(im*k[:,1].*fft(c_mean)));

    #f3 = Figure()
    #ax = Axis(f3[1, 1])
    #lines!(ax, c_squared_mean[:])  
    #display(f3) 

    
  #f3 = Figure()
  #ax = Axis(f3[1, 1])
  #lines!(ax, gc[:],flux_mean[:])
  #display(f3)

  
  save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * ".jld2"
  @save save_name c_mean flux_mean c_squared_mean gc
end

end

end