using FFTW, GLMakie
using ReactingTracers

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