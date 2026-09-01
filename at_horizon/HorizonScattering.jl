module HorizonScattering

using KerrUnboundWaveforms
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using OrdinaryDiffEqVerner: AutoVern9
using SciMLBase: ODEProblem, solve
using SpinWeightedSpheroidalHarmonics
using StaticArrays: SVector
using Printf

const Root = KerrUnboundWaveforms
const Internal = Root.Internal
const KG = Root.KerrGeometry
const NS = Root.NativeSN
const NSL = Root.NativeSNLinear
const FR = Internal.FastReducedSourceWaveform
const SNG = FR.SNGreenAmplitude
const EQS = FR.EquatorialScatteringSource
const UG = EQS.UG

export HorizonModeConfig,
       HorizonSpectrumConfig,
       HorizonKernel,
       HorizonModeResult,
       horizon_flux_factor,
       starobinsky_abs2,
       teukolsky_horizon_conversion,
       solve_horizon_mode,
       solve_horizon_spectrum,
       write_horizon_spectrum_csv

Base.@kwdef struct HorizonModeConfig
    a::Float64 = 0.9
    ell::Int = 2
    m::Int = 2
    omega::Float64 = 0.5
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 30.0
    theta_infinity::Float64 = pi / 3
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    r_outer_min::Float64 = 1_000.0
    r_outer_floor::Float64 = 400.0
    nsteps_per_branch::Int = 32_000
    npoints::Int = 12_001
    asymptotic_match_phase::Float64 = 200.0
    source_tail_order::Int = 3
    green_tail_slow_order::Int = 3
    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_tolerance::Float64 = 1.0e-12
    horizon_expansion_order::Int = 8
    infinity_expansion_order::Int = 10
    integration_rule::String = "trapezoid"
    wronskian_tolerance::Float64 = 3.0e-8
    outer_asymptotic_tolerance::Float64 = 1.0e-7
end

Base.@kwdef struct HorizonSpectrumConfig
    a::Float64 = 0.9
    ell::Int = 2
    m::Int = 2
    omegas::Vector{Float64} = Float64[]
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 30.0
    theta_infinity::Float64 = pi / 3
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    r_outer_min::Float64 = 1_000.0
    r_outer_floor::Float64 = 400.0
    nsteps_per_branch::Int = 32_000
    npoints::Int = 12_001
    asymptotic_match_phase::Float64 = 200.0
    source_tail_order::Int = 3
    green_tail_slow_order::Int = 3
    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_tolerance::Float64 = 1.0e-12
    horizon_expansion_order::Int = 8
    infinity_expansion_order::Int = 10
    integration_rule::String = "trapezoid"
    wronskian_tolerance::Float64 = 3.0e-8
    outer_asymptotic_tolerance::Float64 = 1.0e-7
end

struct UpSolution
    mode::NS.SNMode
    rsin::Float64
    rsout::Float64
    rin::Float64
    rout::Float64
    numerical_solution
end

struct HorizonKernel
    cfg::HorizonModeConfig
    rstar::Vector{Float64}
    rsin::Float64
    rsout::Float64
    xin::Vector{ComplexF64}
    dxin::Vector{ComplexF64}
    xup::Vector{ComplexF64}
    dxup::Vector{ComplexF64}
    lambda::Float64
    c0::ComplexF64
    bref::ComplexF64
    binc::ComplexF64
    scaled_wronskian::ComplexF64
    wronskian_relative_error::Float64
    outer_asymptotic_relative_error::Float64
end

struct HorizonModeResult
    cfg::HorizonModeConfig
    lambda::Float64
    source_points::Int
    finite_projection::ComplexF64
    outer_tail_projection::ComplexF64
    total_projection::ComplexF64
    xh_over_mu::ComplexF64
    zh_over_mu::ComplexF64
    alpha::Float64
    one_sided_dE_horizon_domega_over_mu2::Float64
    p_horizon::Float64
    starobinsky_abs2::Float64
    wronskian_relative_error::Float64
    outer_asymptotic_relative_error::Float64
    elapsed_seconds::Float64
end

function validate_config(cfg::HorizonModeConfig)
    0 <= abs(cfg.a) < 1 || throw(ArgumentError("require |a| < 1"))
    cfg.ell >= 2 || throw(ArgumentError("require ell >= 2"))
    abs(cfg.m) <= cfg.ell || throw(ArgumentError("require |m| <= ell"))
    cfg.omega > 0 || throw(ArgumentError("this development program accepts omega > 0"))
    cfg.energy > 1 || throw(ArgumentError("scattering requires energy > 1"))
    cfg.nsteps_per_branch >= 2 || throw(ArgumentError("need at least two radial source steps"))
    cfg.npoints >= 3 || throw(ArgumentError("need at least three source points"))
    cfg.integration_rule in ("trapezoid", "simpson", "boole", "oscillatory") ||
        throw(ArgumentError("unknown integration rule"))
    return cfg
end

function mode_config(cfg::HorizonSpectrumConfig, omega::Real)
    return HorizonModeConfig(
        a=cfg.a, ell=cfg.ell, m=cfg.m, omega=Float64(omega),
        energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q,
        theta_infinity=cfg.theta_infinity, phi_infinity=cfg.phi_infinity,
        theta_sign=cfg.theta_sign, r_outer_min=cfg.r_outer_min,
        r_outer_floor=cfg.r_outer_floor, nsteps_per_branch=cfg.nsteps_per_branch,
        npoints=cfg.npoints, asymptotic_match_phase=cfg.asymptotic_match_phase,
        source_tail_order=cfg.source_tail_order,
        green_tail_slow_order=cfg.green_tail_slow_order,
        homogeneous_rsin=cfg.homogeneous_rsin,
        homogeneous_rsout_min=cfg.homogeneous_rsout_min,
        homogeneous_tolerance=cfg.homogeneous_tolerance,
        horizon_expansion_order=cfg.horizon_expansion_order,
        infinity_expansion_order=cfg.infinity_expansion_order,
        integration_rule=cfg.integration_rule,
        wronskian_tolerance=cfg.wronskian_tolerance,
        outer_asymptotic_tolerance=cfg.outer_asymptotic_tolerance,
    )
end

function fast_config(cfg::HorizonModeConfig)
    return FR.FastReducedWaveformConfig(
        a=cfg.a, ell=cfg.ell, m=cfg.m, omega=cfg.omega,
        energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q,
        theta_infinity=cfg.theta_infinity, phi_infinity=cfg.phi_infinity,
        theta_sign=cfg.theta_sign, orbit_kind="scattering",
        r_outer_min=cfg.r_outer_min, r_outer_floor=cfg.r_outer_floor,
        nsteps_per_branch=cfg.nsteps_per_branch,
        asymptotic_tail_correction=true,
        asymptotic_match_phase=cfg.asymptotic_match_phase,
        source_tail_order=cfg.source_tail_order,
        green_tail_correction=true,
        scattering_green_tail_slow_order=cfg.green_tail_slow_order,
        source_grid="table", npoints=cfg.npoints,
        homogeneous_rsin=cfg.homogeneous_rsin,
        homogeneous_rsout_min=cfg.homogeneous_rsout_min,
        homogeneous_tolerance=cfg.homogeneous_tolerance,
        integration_rule=cfg.integration_rule,
    )
end

function solve_up(mode::NS.SNMode;
                  rsin::Real,
                  rsout::Real,
                  tolerance::Real,
                  infinity_order::Int,
                  maxiters::Integer=100_000)
    rsin < rsout || throw(ArgumentError("require rsin < rsout"))
    rout = NS.r_from_rstar(mode, rsout)
    rin = NS.r_from_rstar(mode, rsin)
    factor, derivative = NS.infinity_factor(
        mode, rout, :outgoing; order=infinity_order,
    )
    D = NS.delta(mode, rout) / (rout^2 + mode.a^2)
    phase = cis(mode.omega * rsout)
    initial_x = phase * factor
    initial_dx = phase * (D * derivative + 1im * mode.omega * factor)
    problem = ODEProblem(
        NSL.rstar_rhs, SVector(initial_x, initial_dx), (Float64(rsout), Float64(rsin)), mode,
    )
    numerical_solution = solve(
        problem,
        AutoVern9(Rosenbrock23(autodiff=false));
        reltol=tolerance,
        abstol=tolerance,
        maxiters=maxiters,
    )
    return UpSolution(mode, Float64(rsin), Float64(rsout), rin, rout, numerical_solution)
end

function solution_values(solution, rstar::Vector{Float64}, index::Int)
    values = Vector{ComplexF64}(undef, length(rstar))
    solution.numerical_solution(values, rstar; idxs=index)
    return values
end

function build_horizon_kernel(rstar::Vector{Float64}, cfg::HorizonModeConfig, lambda::Real)
    rsin = min(rstar[1], cfg.homogeneous_rsin)
    rsout = max(rstar[end], cfg.homogeneous_rsout_min)
    mode = NS.SNMode(cfg.a, cfg.m, cfg.omega, lambda)
    in_solution = NSL.solve_in(
        mode;
        rsin=rsin,
        rsout=rsout,
        tolerance=cfg.homogeneous_tolerance,
        horizon_order=cfg.horizon_expansion_order,
    )
    amplitudes = NSL.match_infinity(
        in_solution; order=cfg.infinity_expansion_order,
    )
    up_solution = solve_up(
        mode;
        rsin=rsin,
        rsout=rsout,
        tolerance=cfg.homogeneous_tolerance,
        infinity_order=cfg.infinity_expansion_order,
    )
    xin = solution_values(in_solution, rstar, 1)
    dxin = solution_values(in_solution, rstar, 2)
    xup = solution_values(up_solution, rstar, 1)
    dxup = solution_values(up_solution, rstar, 2)
    c0 = ComplexF64(NS.eta_coefficient(mode, 0))
    binc = ComplexF64(amplitudes.binc)
    expected = 2im * cfg.omega * binc / c0
    eta_values = ComplexF64[NS.eta(mode, NS.r_from_rstar(mode, rs)) for rs in rstar]
    wronskians = (xin .* dxup .- dxin .* xup) ./ eta_values
    wronskian_error = maximum(abs.(wronskians .- expected)) /
        max(abs(expected), eps(Float64))
    wronskian_error <= cfg.wronskian_tolerance || error(
        "scaled SN Wronskian gate failed: relative error = $(wronskian_error)",
    )
    r_outer = NS.r_from_rstar(mode, rstar[end])
    f_out, _ = NS.infinity_factor(
        mode, r_outer, :outgoing; order=cfg.infinity_expansion_order,
    )
    outer_model = f_out * cis(cfg.omega * rstar[end])
    outer_error = abs(xup[end] - outer_model) / max(abs(xup[end]), eps(Float64))
    outer_error <= cfg.outer_asymptotic_tolerance || error(
        "Xup outer-asymptotic gate failed: relative error = $(outer_error)",
    )
    return HorizonKernel(
        cfg, rstar, rsin, rsout, xin, dxin, xup, dxup, Float64(lambda),
        c0, ComplexF64(amplitudes.bref), binc, expected,
        Float64(wronskian_error), Float64(outer_error),
    )
end

function auxiliary_green_kernel(kernel::HorizonKernel)
    gcfg = SNG.GreenAmplitudeConfig(
        a=kernel.cfg.a, spin=-2, ell=kernel.cfg.ell, m=kernel.cfg.m,
        omega=kernel.cfg.omega, lambda=kernel.lambda,
        energy_norm=kernel.cfg.energy,
        horizon_expansion_order=kernel.cfg.horizon_expansion_order,
        infinity_expansion_order=kernel.cfg.infinity_expansion_order,
        homogeneous_rsin=kernel.cfg.homogeneous_rsin,
        homogeneous_rsout_min=kernel.cfg.homogeneous_rsout_min,
        homogeneous_tolerance=kernel.cfg.homogeneous_tolerance,
        integration_rule=kernel.cfg.integration_rule,
    )
    return SNG.GreenKernel(
        gcfg, kernel.rstar, kernel.rsin, kernel.rsout, kernel.xup,
        ComplexF64(kernel.lambda), kernel.c0, kernel.bref, kernel.binc,
    )
end

function finite_projection(kernel::HorizonKernel, built, source, cfg::HorizonModeConfig)
    table = built.branch_summed_table
    inner_count = length(kernel.rstar) - length(table.r)
    inner_count >= 2 || error("scattering source must contain an inner rstar segment")
    r = table.r
    drstar_du =
        (r .^ 2 .+ cfg.a^2) ./ UG.delta.(Ref(UG.KerrParams(a=cfg.a)), r) .*
        table.dr_dx
    turn_points = [
        trajectory.points[argmin(getproperty.(trajectory.points, :r))]
        for trajectory in built.trajectories
    ]
    q2_turn = sum(
        UG.q_distribution_coefficients(
            built.kerr, built.constants, built.mode, point, cfg.omega, cfg.m,
            t_origin=built.t_origin, phi_origin=built.phi_origin,
        ).total.q2
        for point in turn_points
    )
    radial_prime = UG.radial_potential_derivative(
        built.kerr, built.constants, built.r_turn,
    )
    sigma_turn = UG.sigma(built.kerr, built.r_turn, turn_points[1].theta)
    source_phase_turn = cis(-UG.k_over_delta_antiderivative(
        built.kerr, built.r_turn, cfg.omega, cfg.m,
    ))
    endpoint_value =
        kernel.xup[inner_count] * built.particle_mass * source_phase_turn *
        2 * sigma_turn * q2_turn /
        (
            built.r_turn^2 * sqrt(built.r_turn^2 + cfg.a^2) * sqrt(radial_prime)
        )
    auxiliary = auxiliary_green_kernel(kernel)
    result = SNG.apply_green_kernel_partitioned(
        auxiliary, source, inner_count, table.integration_x, drstar_du, endpoint_value,
    )
    return ComplexF64(result.integral)
end

function outgoing_asymptotic_factor(coefficients, omega::Float64, radius::Float64)
    return sum(
        coefficients[order + 1] / (omega * radius)^order
        for order in 0:(length(coefficients) - 1);
        init=0.0 + 0.0im,
    )
end

function outer_tail_projection(kernel::HorizonKernel, built, cfg::HorizonModeConfig)
    length(built.tables) == 2 || error(
        "a scattering horizon tail requires incoming and outgoing source tables",
    )
    mode = NS.SNMode(cfg.a, cfg.m, cfg.omega, kernel.lambda)
    coefficients = NS.infinity_coefficients(
        mode, :outgoing; order=cfg.infinity_expansion_order,
    )
    tail = 0.0 + 0.0im
    for table in built.tables
        fit_count = EQS.asymptotic_fit_point_count(table.r)
        fit_start = length(table.r) - fit_count + 1
        r = @view table.r[fit_start:end]
        phase = @view table.phase_arguments[fit_start:end]
        source_rate = @view table.chi_prime_values[fit_start:end]
        W = @view(table.Wnn[fit_start:end]) .+
            @view(table.Wnmb[fit_start:end]) .+
            @view(table.Wmbmb[fit_start:end])
        smooth_W = W .* cis.(-phase)
        radial_phase = Float64[
            UG.k_over_delta_antiderivative(built.kerr, radius, cfg.omega, cfg.m)
            for radius in r
        ]
        orbit_phase = phase .- radial_phase
        rstar = KG.rstar_from_r.(Ref(cfg.a), r)
        total_phase = orbit_phase .+ cfg.omega .* rstar
        radial_phase_rate = UG.phase_integrand.(Ref(built.kerr), r, cfg.omega, cfg.m)
        orbit_rate = source_rate .- radial_phase_rate
        rstar_rate = (r .^ 2 .+ cfg.a^2) ./ UG.delta.(Ref(built.kerr), r)
        total_rate = orbit_rate .+ cfg.omega .* rstar_rate
        xup_factor = ComplexF64[
            outgoing_asymptotic_factor(coefficients, cfg.omega, radius) for radius in r
        ]
        radial_prefactor = inv.(r .^ 2 .* sqrt.(r .^ 2 .+ cfg.a^2))
        amplitude = built.particle_mass .* radial_prefactor .* smooth_W .* xup_factor
        tail += EQS.fitted_oscillatory_tail(
            r, amplitude .* cis.(total_phase), cis.(total_phase), total_rate;
            order=cfg.green_tail_slow_order,
            max_points=EQS.asymptotic_fit_point_count(r),
        )
    end
    return ComplexF64(tail)
end

function starobinsky_abs2(a::Real, m::Integer, omega::Real, lambda::Real)
    a = Float64(a)
    omega = Float64(omega)
    lambda = Float64(lambda)
    Q = lambda + 2
    value =
        (Q^2 + 4a * m * omega - 4a^2 * omega^2) *
        ((Q - 2)^2 + 36a * m * omega - 36a^2 * omega^2) +
        (2Q - 1) * (96a^2 * omega^2 - 48a * m * omega) +
        144omega^2 * (1 - a^2)
    return Float64(value)
end

function horizon_flux_factor(a::Real, m::Integer, omega::Real, lambda::Real)
    a = Float64(a)
    omega = Float64(omega)
    rp = 1 + sqrt(1 - a^2)
    p = omega - m * a / (2rp)
    epsilon_h = sqrt(1 - a^2) / (4rp)
    C2 = starobinsky_abs2(a, m, omega, lambda)
    C2 > 0 || error("nonpositive Starobinsky modulus squared")
    return Float64(
        256 * (2rp)^5 * p * (p^2 + 4epsilon_h^2) *
        (p^2 + 16epsilon_h^2) * omega^3 / C2,
    )
end

function teukolsky_horizon_conversion(a::Real, m::Integer, omega::Real)
    a = Float64(a)
    omega = Float64(omega)
    rp = 1 + sqrt(1 - a^2)
    bracket =
        (8 - 24im * omega - 16omega^2) * rp^2 +
        (12im * a * m - 16 + 16a * m * omega + 24im * omega) * rp +
        (-4a^2 * m^2 - 12im * a * m + 8)
    return ComplexF64(inv(sqrt(2rp) * bracket))
end

function solve_horizon_mode(cfg::HorizonModeConfig; orbit_cache=nothing)
    validate_config(cfg)
    started = time()
    fcfg = fast_config(cfg)
    cache = orbit_cache === nothing ? FR.build_fast_reduced_orbit_cache(fcfg) : orbit_cache
    built = EQS.build_scattering_source_from_cache(FR.source_config(fcfg), cache)
    rstar, source = FR.source_grid_values(fcfg, built)
    lambda = real(getproperty(built.harmonic, :lambda))
    kernel = build_horizon_kernel(rstar, cfg, lambda)
    finite = finite_projection(kernel, built, source, cfg)
    tail = outer_tail_projection(kernel, built, cfg)
    projection = finite + tail
    xh = projection / kernel.scaled_wronskian
    z_horizon = teukolsky_horizon_conversion(cfg.a, cfg.m, cfg.omega) * xh
    alpha = horizon_flux_factor(cfg.a, cfg.m, cfg.omega, lambda)
    energy = alpha * abs2(z_horizon) / (2 * cfg.omega^2)
    rp = 1 + sqrt(1 - cfg.a^2)
    p = cfg.omega - cfg.m * cfg.a / (2rp)
    return HorizonModeResult(
        cfg, lambda, length(rstar), finite, tail, projection, xh, z_horizon,
        alpha, Float64(energy), Float64(p),
        starobinsky_abs2(cfg.a, cfg.m, cfg.omega, lambda),
        kernel.wronskian_relative_error, kernel.outer_asymptotic_relative_error,
        time() - started,
    )
end

function solve_horizon_spectrum(cfg::HorizonSpectrumConfig)
    isempty(cfg.omegas) && throw(ArgumentError("omegas cannot be empty"))
    all(>(0), cfg.omegas) || throw(ArgumentError("all frequencies must be positive"))
    seed = mode_config(cfg, first(cfg.omegas))
    cache = FR.build_fast_reduced_orbit_cache(fast_config(seed))
    return HorizonModeResult[
        solve_horizon_mode(mode_config(cfg, omega); orbit_cache=cache)
        for omega in cfg.omegas
    ]
end

function write_horizon_spectrum_csv(path::AbstractString, results)
    directory = dirname(abspath(path))
    isdir(directory) || mkpath(directory)
    open(path, "w") do io
        println(io, "omega,ell,m,lambda,re_ZH_over_mu,im_ZH_over_mu,abs_ZH_over_mu,p_horizon,alpha,starobinsky_abs2,one_sided_dEH_domega_over_mu2,re_projection,im_projection,re_tail_projection,im_tail_projection,wronskian_relative_error,outer_asymptotic_relative_error,elapsed_seconds")
        for result in results
            @printf(
                io,
                "%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                result.cfg.omega, result.cfg.ell, result.cfg.m, result.lambda,
                real(result.zh_over_mu), imag(result.zh_over_mu), abs(result.zh_over_mu),
                result.p_horizon, result.alpha, result.starobinsky_abs2,
                result.one_sided_dE_horizon_domega_over_mu2,
                real(result.total_projection), imag(result.total_projection),
                real(result.outer_tail_projection), imag(result.outer_tail_projection),
                result.wronskian_relative_error,
                result.outer_asymptotic_relative_error, result.elapsed_seconds,
            )
        end
    end
    return abspath(path)
end

end
