using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2

function adv(c,um,kappa,k)
    ch=fft(c)#
    dc=real(ifft(-im*k.*ch.*um-kappa*k.*k.*ch)) #d/dx --> =-ik
    return dc
end

varu=0.1   # variance of u
r=0.2    # damping rate in OE
kappa=0.01    # "subgrid" kappa
la=0.05    # relaxation to forcing
dt=1/2048
rfac=sqrt(2*varu*dt*r)

N=10
av=1/N
uamp=0.1

# slighly different choice of x to Glenn
x_length = 1024
x = nodes(x_length)
k  = wavenumbers(x_length)

ox=ones(x_length,1);

# initiate velocities
u=randn(1,N)*varu #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
ind = findall(u -> abs(u) >= 5, u)
u[ind] .= 5

# initial concentration
c=x*zeros(1,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)

# concentration bias
mag = 0.1;
lengthscale = 1
Δconc = mag*sin.(2*pi/1*lengthscale);
Δconc = Δconc*ones(1, 10);

# plotting parameters
t=0
tmax=1000
tpl=1 # plots every timestep  = 1

f = Figure()
Axis(f[1, 1])
lines!(x,mean(c,dims=2)[:])
lines!(x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
#title(num2str(t));
f

cs=zeros(x_length, tmax);
fs=zeros(x_length, tmax);
c_p=c;
while t<tmax+dt/2
  if rem(t,tpl)==0
    if t > 0
      print(t)
      f2 = Figure()
      Axis(f2[1, 1])
      lines!(x,mean(c,dims=2)[:])
      lines!(x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
      #title(num2str(t));
      f2
      cs[:, Int(t)]= mean(c,dims = 2)
      fs[:, Int(t)]= (1/N)*c*u'
    end
  end
  tmp=c;c=1.5*c-0.5*c_p;c_p=c #not sure why glenn has added this line?
  dc=adv(c,ox*u,kappa,k)
  c = c +  dc*dt + la*dt*(-c-c .^2 + Δconc +c.*Δconc)./(1 .+ Δconc)
  #c = c +  dc*dt + la*dt*(1+c).*(1-abs(1+c)./(1+delta));
  global t = t + dt
  u = u - r*dt*u+rfac*randn(1,N)
  ind = findall(u -> abs(u) >= 5, u)
  u[ind] .= 5
end

#cf=mean(cs[:,2:end],dims = 2)
#ff=mean(fs[:,2:end],dims = 2)
cf=mean(cs[:,201:end],dims = 2);
ff=mean(fs[:,201:end],dims = 2);
gc=real(ifft(im*k[:,1].*fft(cf)));

f3 = Figure()
Axis(f3[1, 1])
lines!(gc[:],ff[:])
#title(num2str(t));
f3

save_name = "mag_" * string(mag) * "_k_" * string(lengthscale) * "_FT.jld2"

@save save_name cs fs ff cf gc