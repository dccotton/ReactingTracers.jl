using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers

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
delta = mag*sin.(2*pi/1*x);
delta = delta*ones(1, 10);

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

cs=[];
fs=[];
c_p=c;
while t<tmax+dt/2
  if rem(t,tpl)==0
    f = Figure()
    Axis(f[1, 1])
    lines!(x,mean(c,dims=2)[:])
    lines!(x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
    #title(num2str(t));
    f
    cs=[cs,mean(c,dims = 2)]
    fs=[fs,(1/N)*c*u']
  end
  tmp=c;c=1.5*c-0.5*c_p;c_p=c;
  dc=adv(c,ox*u,kappa,k);
  c = c +  dc*dt + la*dt*(-c-c.^2+delta+c.*delta)./(1+delta)
  #c = c +  dc*dt + la*dt*(1+c).*(1-abs(1+c)./(1+delta));
  t = t + dt;
  u = u - r*dt*u+rfac*randn(1,N);

    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5
end

cf=mean(cs[:,201:end],dims = 2);
ff=mean(fs[:,201:end],dims = 2);
gc=real(ifft(i*k[:,1].*fft(cf)));

#plot(gc,ff,[min(gc),max(gc)],[0,0],'--k',[0,0],[min(ff),max(ff)],'--k')