using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2
using CUDA

Array_Type = Array
# timestep minimum
#1/\kappa k^2

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

function rk4(h, c, u,κ, k)
    k1 = adv2(c, u, κ, k)
    k2 = adv2(c + h/2*k1, u, κ, k)
    k3 = adv2(c + h/2*k2, u, κ, k)
    k4 = adv2(c + h*k3, u, κ, k)
    dc = h/6*(k1+2*k2+2*k3+k4)
    return dc
end

r = 0.2    # damping rate in OE
varu = r^2  # variance of u (ensures d/dx(uc) ~ 1)
κ = 0.02    # "subgrid" kappa
λ = 0.1    # relaxation to forcing
dt = 4*2/(κ*1024^2)#1/5250
rfac=sqrt(2*varu*dt*r)

N=2
av=1/N

# setup grid
x_length = 1024
x = Array_Type(nodes(x_length, a = -pi, b = pi))
k  = Array_Type(wavenumbers(x_length))

adv2 = adv_closure(Array_Type(zeros(x_length, N)))


ox=Array_Type(ones(x_length,1));
lambdas = [0.1, 1, 10]
magnitudes = [0.7, 0.5, 0.1, 0.9]
divisor = [25, 13, 6, 3, 1] #[1, 3, 6, 13, 25]#, 63, 125]

for λ in lambdas
for mag in magnitudes
for div in ProgressBar(divisor)
    # initiate velocities
    u = randn(1, N) * varu #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5
    u = Array_Type(u)

    # initial concentration
    c = Array_Type(ones(x_length, N))#array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
    dc = copy(c)
    # concentration bias
    Δconc = Array_Type(mag * cos.(div * x))
    Δconc = Array_Type(Δconc * ones(1, N))
    # plotting parameters

    t = 0
    tmax = 1000 # 1000
    t_array = collect(t:dt:tmax)
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

    cs = Array_Type(zeros(x_length))
    c²s = Array_Type(zeros(x_length))
    fs = Array_Type(zeros(x_length))
    num_avg = 0
    for (i, t) in ProgressBar(enumerate(t_array))
        if minimum(abs.(save_times .- t)) == 0
            if t > 0
                #f2 = Figure()
                #ax = Axis(f2[1, 1])
                #lines!(ax, x,mean(c,dims=2)[:])
                #lines!(ax, x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
                #display(f2)
                if t > t_array[210]
                    # accumulate locally averaged concentration and flux
                    cs .+= mean(c, dims=2)
                    c²s .+= mean(c .^ 2, dims=2)
                    fs .+= (1 / N) * c * u'
                    num_avg += 1
                end
            end
        end
        dc .= rk4(dt, c, ox * u, κ, k)
        @. c = c + dc + λ * dt * (c) * (1 - c / (1 + Δconc))

        #@. c = c +  dc*dt + λ*dt*(c)*(1-c/(1 + Δconc))
        if any(isnan, c)
            break
        end

        𝒩 = randn(1, N)
        @. u = u - r * dt * u + rfac * 𝒩
        ind = findall(u -> abs(u) >= 5, u)
        u[ind] .= 5
        u = Array_Type(u)
    end

    c_mean = Array(cs / num_avg) # mean(cs[:,201:end-1],dims = 2); #<c>
    flux_mean = Array(fs / num_avg)# mean(fs[:,201:end-1],dims = 2); #<uc>
    c_squared_mean = Array(c²s / num_avg)# mean(cs[:,201:end-1].^2,dims = 2); #<c>
    gc = real(ifft(im * k[:, 1] .* fft(c_mean)))

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


    save_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits=3)) * "_lambda_" * string(λ) * "_FT.jld2"
    @save save_name c_mean flux_mean c_squared_mean gc
end

end

end