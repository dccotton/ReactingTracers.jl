using Statistics

a = [1, 2, 3]
b = [2, 2, 2]'
a*b
a*b + a*b
(a*b) ./ (a*b)
# test creating a matrix from one vector

(a*b) .^2

# test finding elements in array and changing them
a = Array(1:10)
indices = findall(a -> a >= 5, a)
a[indices] .= 5

a = Array(1:10)
indices = findall(a -> a >= 11, a)
a[indices] .= 5
a

Int(6.0)

mean(a*b, dims=1)
mean(a*b, dims=2)

b = zeros(2, 3)
b[1, :] = a
b

plus_one(2)

function plus_one(n)
    return n+1
end

# test loop
s = 10
t = 0
for i = 1:10
    t = s + i
    print(t)
end

f = Figure()

ax = f[1, 1] = Axis(f)

lines!(0..15, sin, label = L"\overline{c}", color = :blue)
lines!(0..15, cos, label = L"\overline{cu}", color = :red)
lines!(0..15, x -> -cos(x), label = L"\Delta(x)", color = :green)

f[1, 2] = Legend(f, ax, "Trig Functions", framevisible = false)

f

fig = Figure()
ax = fig[1,1] = Axis(fig, xlabel = L"x",
  xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
  xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
lines!(x,mean(c,dims=2)[:], label = L"\hat{c}")
lines!(x,(1/N)*c*u'[:], label = L"\hat{uc}") #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
lines!(x,Δconc[:, 1], label = L"\Delta(x)")
fig[1, 2] = Legend(fig, ax, "Trig Functions", framevisible = false)
