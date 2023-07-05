using FFTW, GLMakie
using ReactingTracers
N = 64
x = nodes(64)
k  = wavenumbers(64)
∂x = im .* k
Δ = ∂x .^2

scatter(x, sin.(x))

y = sin.(x)
ŷ = fft(y)
dy = ifft(∂x .* ŷ)

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, x, real.(dy))
lines!(ax, x, cos.(x), color = :red)
display(fig)

x2d = reshape(x, (64, 1))
y2d = reshape(x, (1, 64))
∂x2d = reshape(∂x, (64, 1))
∂y2d = reshape(∂x, (1, 64))

z = sin.(x2d) .* sin.(y2d)
dz = real.(ifft(∂y2d .* fft(z)))

fig = Figure()
ax1 = Axis(fig[1, 1])
heatmap!(ax1, x2d[:], y2d[:], z)
ax2 = Axis(fig[1, 2])
heatmap!(ax2, x2d[:], y2d[:], dz)
display(fig)