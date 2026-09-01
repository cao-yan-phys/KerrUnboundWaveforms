include(joinpath(@__DIR__, "HorizonScattering.jl"))
using .HorizonScattering

cfg = HorizonModeConfig(
    a=0.9,
    energy=1.2,
    lz=8.0,
    carter_q=30.0,
    theta_infinity=pi / 3,
    phi_infinity=0.0,
    theta_sign=1.0,
    ell=2,
    m=2,
    omega=0.5,
)

result = solve_horizon_mode(cfg)
write_horizon_spectrum_csv(joinpath(@__DIR__, "output", "example_l2_m2.csv"), [result])
println("Z_H/mu = ", result.zh_over_mu)
println("(dE_H/domega)/mu^2 = ", result.one_sided_dE_horizon_domega_over_mu2)
