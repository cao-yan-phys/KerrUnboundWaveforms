module KerrHorizonScattering

using KerrUnboundWaveforms
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using OrdinaryDiffEqVerner: AutoVern9
using SciMLBase: ODEProblem, solve, successful_retcode
using StaticArrays: SVector
using ForwardDiff
using SpecialFunctions: besselk
using SpinWeightedSpheroidalHarmonics: spin_weighted_spheroidal_harmonic

const Root = KerrUnboundWaveforms
const Internal = Root.Internal
const FR = Internal.FastReducedSourceWaveform
const EQS = FR.EquatorialScatteringSource
const UG = EQS.UG
const NativeSN = Root.NativeSN
const NativeSNLinear = Root.NativeSNLinear
const KerrGeometry = Root.KerrGeometry
const SSM = Internal.SSM

include(joinpath(@__DIR__, "CenteredCanonicalSource.jl"))

export HorizonScatteringConfig,
       HorizonSpectrumConfig,
       DirectHorizonOrbitCache,
       HorizonSNResult,
       HorizonTeukolskyResult,
       HorizonPhysicalResult,
       build_centered_scattering_source,
       solve_up,
       compute_horizon_sn_amplitude,
       compute_horizon_teukolsky_amplitude,
       horizon_physical_from_teukolsky,
       compute_horizon_physical,
       horizon_edge_scan,
       schwarzschild_pm_dE_domega,
       horizon_schwarzschild_pm_benchmark,
       horizon_direct_convergence,
       horizon_direct_outer_convergence,
       direct_cancellation_condition,
       horizon_frequency_spectrum,
       horizon_spectrum_diagnostics,
       write_horizon_spectrum_csv,
       horizon_frequency_grid,
       solve_horizon_spectrum,
       build_direct_orbit_cache,
       solve_horizon,
       photon_edge_adjacent_energy_ratio,
       horizon_axisymmetric_photon_edge_scan,
       weighted_wronskian_diagnostics

Base.@kwdef mutable struct HorizonScatteringConfig
    a::Float64 = 0.9
    ell::Int = 2
    m::Int = 2
    omega::Float64 = 0.6325395978733317
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 30.0
    theta_infinity::Float64 = pi / 3
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    particle_mass::Float64 = 1.0

    asymptotic_match_phase::Float64 = 600.0
    r_outer_floor::Float64 = 400.0
    nsteps_per_branch::Int = 32_000
    source_tail_order::Int = 3
    allow_theta_turns::Bool = true

    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_tolerance::Float64 = 1.0e-12
    horizon_expansion_order::Int = 8
    infinity_expansion_order::Int = 10

    green_tail_correction::Bool = true
    horizon_green_tail_order::Int = 3
    direct_tail_order::Int = 3
    direct_geodesic_tolerance::Float64 = 2.0e-12
    direct_geodesic_max_step::Float64 = 0.05
    direct_geodesic_maxiters::Int = 1_000_000
    direct_gauss_order::Int = 20
    direct_gauss_panels_per_600::Int = 2048
    direct_tail_fit_points::Int = 256
    direct_tail_window_fraction::Float64 = 0.025
end

struct DirectHorizonOrbitCache
    a::Float64
    energy::Float64
    lz::Float64
    carter_q::Float64
    theta_infinity::Float64
    phi_infinity::Float64
    theta_sign::Float64
    omega_abs::Float64
    asymptotic_match_phase::Float64
    r_outer_floor::Float64
    direct_geodesic_tolerance::Float64
    direct_geodesic_max_step::Float64
    r_outer::Float64
    r_turn::Float64
    geodesic
end

Base.@kwdef mutable struct HorizonSpectrumConfig
    base::HorizonScatteringConfig = HorizonScatteringConfig()
    frequencies::Vector{Float64} = Float64[]
    omega_min::Float64 = 0.02
    omega_max::Float64 = 0.7
    frequency_count::Int = 12
    frequency_grid::String = "logarithmic"
    ell_min::Int = 2
    auto_l_max::Int = 6
    shell_tolerance::Float64 = 1.0e-3
    consecutive_shells::Int = 2
    high_endpoint_fraction_tolerance::Float64 = 1.0e-2
    high_endpoint_consecutive_points::Int = 2
    cancellation_condition_limit::Float64 = 1.0e8
    stop_when_converged::Bool = true
    gauss_panels::Union{Nothing,Int} = nothing
end

struct UpSolution
    mode::NativeSN.SNMode
    rsin::Float64
    rsout::Float64
    rin::Float64
    rout::Float64
    numerical_solution
end

struct HorizonKernel
    mode::NativeSN.SNMode
    rsin::Float64
    rsout::Float64
    xin_solution
    xup_solution::UpSolution
    xin::Vector{ComplexF64}
    xup::Vector{ComplexF64}
    binc::ComplexF64
    bref::ComplexF64
    denominator::ComplexF64
    lambda::ComplexF64
    c0::ComplexF64
    wronskian_relative_spread::Float64
    wronskian_relative_offset::Float64
end

struct HorizonSNResult
    cfg::HorizonScatteringConfig
    source
    rstar::Vector{Float64}
    reduced_source::Vector{ComplexF64}
    xup::Vector{ComplexF64}
    integral_numeric::ComplexF64
    integral_tail::ComplexF64
    integral::ComplexF64
    amplitude_x::ComplexF64
    denominator::ComplexF64
    lambda::ComplexF64
    binc::ComplexF64
    bref::ComplexF64
    wronskian_relative_spread::Float64
    wronskian_relative_offset::Float64
end

struct HorizonTeukolskyResult
    cfg::HorizonScatteringConfig
    r_outer::Float64
    r_turn::Float64
    lambda::ComplexF64
    incoming_numeric::ComplexF64
    outgoing_numeric::ComplexF64
    incoming_tail::ComplexF64
    outgoing_tail::ComplexF64
    source_integral::ComplexF64
    z_h_minus2::ComplexF64
    btrans::ComplexF64
    binc_teukolsky::ComplexF64
    ctrans::ComplexF64
    binc_sn::ComplexF64
    wronskian_relative_spread::Float64
    wronskian_relative_offset::Float64
    gauss_panels::Int
    gauss_order::Int
    tail_fit_points::Int
    geodesic_incoming_steps::Int
    geodesic_outgoing_steps::Int
end

struct HorizonPhysicalResult
    teukolsky::HorizonTeukolskyResult
    starobinsky::ComplexF64
    starobinsky_abs2::Float64
    omega_h::Float64
    k::Float64
    epsilon::Float64
    kappa_h::Float64
    y_hole_plus2::ComplexF64
    psi0_hh::ComplexF64
    Psi_h::ComplexF64
    shear_h::ComplexF64
    absorption_factor::Float64
    dE_domega::Float64
    dE_domega_shear::Float64
    dJ_domega::Float64
    dA_domega::Float64
    flux_relative_difference::Float64
end

@inline _finite_complex(z) = isfinite(real(z)) && isfinite(imag(z))

function _waveform_config(cfg::HorizonScatteringConfig)
    momentum = sqrt(cfg.energy^2 - 1)
    momentum > 0 || error("ordinary unbound scattering requires E>1")
    phase_rate = abs(cfg.omega) / (momentum * (cfg.energy + momentum))
    r_outer = max(cfg.r_outer_floor, cfg.asymptotic_match_phase / phase_rate)
    return FR.FastReducedWaveformConfig(
        a=cfg.a,
        ell=cfg.ell,
        m=cfg.m,
        omega=cfg.omega,
        energy=cfg.energy,
        lz=cfg.lz,
        carter_q=cfg.carter_q,
        theta_infinity=cfg.theta_infinity,
        phi_infinity=cfg.phi_infinity,
        theta_sign=cfg.theta_sign,
        orbit_kind="scattering",
        r_outer_min=r_outer,
        r_outer_floor=cfg.r_outer_floor,
        nsteps_per_branch=cfg.nsteps_per_branch,
        asymptotic_tail_correction=true,
        asymptotic_match_phase=cfg.asymptotic_match_phase,
        source_tail_order=cfg.source_tail_order,
        green_tail_correction=cfg.green_tail_correction,
        allow_theta_turns=cfg.allow_theta_turns,
        source_grid="table",
        npoints=12_001,
        homogeneous_rsin=cfg.homogeneous_rsin,
        homogeneous_rsout_min=cfg.homogeneous_rsout_min,
        homogeneous_method="linear",
        homogeneous_tolerance=cfg.homogeneous_tolerance,
        integration_rule="trapezoid",
    )
end

function _centered_piece_table(kerr, constants, traj, mode, scfg, r_turn,
                               t_origin, phi_origin)
    points = EQS.table_points(traj, r_turn, scfg)
    n = length(points)
    r = Vector{Float64}(undef, n)
    f2 = Vector{ComplexF64}(undef, n)
    g1 = similar(f2)
    g2 = similar(f2)
    h0 = similar(f2)
    h1 = similar(f2)
    h2 = similar(f2)
    phase_values = similar(f2)
    phase_arguments = Vector{Float64}(undef, n)
    chi_prime_values = Vector{Float64}(undef, n)
    terms = UG.TeukolskyATermsBuffer()

    @inbounds for i in eachindex(points)
        point = points[i]
        r[i] = point.r
        radial_phase = UG.k_over_delta_antiderivative(
            kerr, point.r, scfg.omega, scfg.m,
        )
        orbital_argument = scfg.omega * (point.t - t_origin) -
                           scfg.m * (point.phi - phi_origin)
        orbital_phase = cis(orbital_argument)
        echi = cis(radial_phase)
        phase_arguments[i] = orbital_argument + radial_phase
        phase_values[i] = cis(phase_arguments[i])
        UG.fill_q_distribution_integrands!(
            f2, g1, g2, h0, h1, h2, i,
            terms, kerr, mode, point, scfg.omega, scfg.m,
            orbital_phase, echi,
        )
        chi_prime_values[i] = EQS.point_chi_prime_cached(
            kerr, point, scfg.omega, scfg.m,
        )
    end

    integration_x, dr_dx = EQS.channel_integration_grid(r, traj.piece.r_turn)
    nn = centered_mathcalW_channel(
        r, integration_x, dr_dx, nothing, nothing, f2,
        phase_values, phase_arguments, chi_prime_values;
        tail_order=scfg.asymptotic_tail_order,
    )
    nm = centered_mathcalW_channel(
        r, integration_x, dr_dx, nothing, g1, g2,
        phase_values, phase_arguments, chi_prime_values;
        tail_order=scfg.asymptotic_tail_order,
    )
    mm = centered_mathcalW_channel(
        r, integration_x, dr_dx, h0, h1, h2,
        phase_values, phase_arguments, chi_prime_values;
        tail_order=scfg.asymptotic_tail_order,
    )

    all(_finite_complex, nn.W) || error("non-finite centered Wnn")
    all(_finite_complex, nm.W) || error("non-finite centered Wnmb")
    all(_finite_complex, mm.W) || error("non-finite centered Wmbmb")

    table = (
        label=traj.piece.label,
        r=r,
        Wnn=nn.W,
        Wnmb=nm.W,
        Wmbmb=mm.W,
        inner_m0_nn=nn.inner_m0,
        inner_m1_nn=nn.inner_m1,
        inner_m0_nmb=nm.inner_m0,
        inner_m1_nmb=nm.inner_m1,
        inner_m0_mbmb=mm.inner_m0,
        inner_m1_mbmb=mm.inner_m1,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        chi_prime_values=chi_prime_values,
        chi_prime_outer=chi_prime_values[end],
        asymptotic_tail_correction=true,
    )

    return EQS.add_turn_moment_correction(
        table, kerr, constants, traj, points, mode, scfg,
        t_origin, phi_origin,
    )
end

function build_centered_scattering_source(cfg::HorizonScatteringConfig;
                                          orbit_cache=nothing)
    wcfg = _waveform_config(cfg)
    scfg = FR.source_config(wcfg)
    scfg.particle_mass = cfg.particle_mass
    cache = orbit_cache === nothing ? FR.build_fast_reduced_orbit_cache(wcfg) : orbit_cache
    EQS.validate_orbit_cache(scfg, cache)

    kerr = cache.kerr
    constants = cache.constants
    harmonic, mode = EQS.scattering_mode(kerr, scfg)
    t_origin, phi_origin = EQS.trajectory_phase_origin(cache.trajectories, scfg)

    tables = [
        _centered_piece_table(
            kerr, constants, traj, mode, scfg, cache.r_turn,
            t_origin, phi_origin,
        )
        for traj in cache.trajectories
    ]
    branch_summed_table = EQS.build_branch_summed_table(tables, scfg)
    table_support = (
        r_minimum=maximum(table.r[1] for table in tables),
        r_maximum=minimum(table.r[end] for table in tables),
    )

    function reduced_at_r(r)
        inside_turn = r < branch_summed_table.r[1]
        Wnn = inside_turn ?
              branch_summed_table.inner_m1_nn - r * branch_summed_table.inner_m0_nn :
              EQS.interp_complex(branch_summed_table.r, branch_summed_table.Wnn, r)
        Wnmb = inside_turn ?
               branch_summed_table.inner_m1_nmb - r * branch_summed_table.inner_m0_nmb :
               EQS.interp_complex(branch_summed_table.r, branch_summed_table.Wnmb, r)
        Wmbmb = inside_turn ?
                branch_summed_table.inner_m1_mbmb - r * branch_summed_table.inner_m0_mbmb :
                EQS.interp_complex(branch_summed_table.r, branch_summed_table.Wmbmb, r)
        total = Wnn + Wnmb + Wmbmb
        prefactor = UG.delta(kerr, r) / (r^2 * (r^2 + kerr.a^2)^(3 / 2))
        phase = exp(-1im * UG.k_over_delta_antiderivative(
            kerr, r, scfg.omega, scfg.m,
        ))
        return scfg.particle_mass * total * prefactor * phase
    end

    return (
        kerr=kerr,
        constants=constants,
        harmonic=harmonic,
        mode=mode,
        particle_mass=scfg.particle_mass,
        t_origin=Float64(t_origin),
        phi_origin=Float64(phi_origin),
        r_turn=cache.r_turn,
        r_minimum=KerrGeometry.r_from_rstar(kerr.a, cfg.homogeneous_rsin),
        r_maximum=table_support.r_maximum,
        table_r_maximum=table_support.r_maximum,
        pieces=cache.pieces,
        trajectories=cache.trajectories,
        tables=tables,
        branch_summed_table=branch_summed_table,
        table_r_minimum=table_support.r_minimum,
        reduced_at_r=reduced_at_r,
        waveform_config=wcfg,
        source_config=scfg,
        orbit_cache=cache,
    )
end

function _rstar_rhs(u, mode::NativeSN.SNMode, rs)
    r = NativeSN.r_from_rstar(mode, rs)
    F, U = NativeSN.potentials(mode, r)
    return SVector(u[2], F * u[2] + U * u[1])
end

function solve_up(mode::NativeSN.SNMode;
                  rsin::Real=-50.0,
                  rsout::Real=1000.0,
                  tolerance::Real=1.0e-12,
                  infinity_order::Int=10,
                  maxiters::Integer=200_000)
    rsin < rsout || error("require rsin < rsout")
    mode.omega != 0 || error("omega=0 requires a separate static solver")
    rin = NativeSN.r_from_rstar(mode, rsin)
    rout = NativeSN.r_from_rstar(mode, rsout)
    factor, derivative = NativeSN.infinity_factor(
        mode, rout, :outgoing; order=infinity_order,
    )
    D = NativeSN.delta(mode, rout) / (rout^2 + mode.a^2)
    phase = exp(1im * mode.omega * rsout)
    initial_X = phase * factor
    initial_Y = phase * (D * derivative + 1im * mode.omega * factor)
    problem = ODEProblem(
        _rstar_rhs,
        SVector(initial_X, initial_Y),
        (Float64(rsout), Float64(rsin)),
        mode,
    )
    numerical_solution = solve(
        problem,
        AutoVern9(Rosenbrock23(autodiff=false));
        reltol=tolerance,
        abstol=tolerance,
        maxiters=maxiters,
    )
    return UpSolution(
        mode, Float64(rsin), Float64(rsout), rin, rout, numerical_solution,
    )
end

function _evaluate_component!(out::Vector{ComplexF64}, solution, rs, idx::Int)
    solution(out, rs; idxs=idx)
    return out
end

function _weighted_wronskian(mode, yin, yup, r)
    c0 = NativeSN.eta_coefficient(mode, 0)
    rho = c0 / NativeSN.eta(mode, r)
    return rho * (yin[1] * yup[2] - yup[1] * yin[2])
end

function weighted_wronskian_diagnostics(mode::NativeSN.SNMode,
                                        xin_solution, up_solution::UpSolution,
                                        binc::ComplexF64;
                                        samples::Int=9)
    lo = max(xin_solution.rsin, up_solution.rsin)
    hi = min(xin_solution.rsout, up_solution.rsout)
    hi > lo || error("homogeneous solutions have no common interval")
    pad = min(20.0, 0.1 * (hi - lo))
    lo2 = lo + pad
    hi2 = hi - pad
    hi2 > lo2 || (lo2, hi2 = lo, hi)
    nodes = range(lo2, hi2; length=samples)
    values = ComplexF64[]
    for rs in nodes
        yin = xin_solution.numerical_solution(rs)
        yup = up_solution.numerical_solution(rs)
        r = NativeSN.r_from_rstar(mode, rs)
        push!(values, ComplexF64(_weighted_wronskian(mode, yin, yup, r)))
    end
    target = 2im * mode.omega * binc
    mean_value = sum(values) / length(values)
    spread = maximum(abs.(values .- mean_value)) / max(abs(mean_value), eps(Float64))
    offset = abs(mean_value - target) / max(abs(target), eps(Float64))
    return (
        values=values,
        mean=ComplexF64(mean_value),
        target=ComplexF64(target),
        relative_spread=Float64(spread),
        relative_offset=Float64(offset),
    )
end

function _build_horizon_kernel(rstar::Vector{Float64}, built,
                               cfg::HorizonScatteringConfig)
    lambda = real(getproperty(built.harmonic, :lambda))
    mode = NativeSN.SNMode(cfg.a, cfg.m, cfg.omega, lambda)
    rsin = min(first(rstar), cfg.homogeneous_rsin)
    rsout = max(last(rstar), cfg.homogeneous_rsout_min)

    xin_solution = NativeSNLinear.solve_in(
        mode;
        rsin=rsin,
        rsout=rsout,
        tolerance=cfg.homogeneous_tolerance,
        horizon_order=cfg.horizon_expansion_order,
    )
    amps = NativeSNLinear.match_infinity(
        xin_solution; order=cfg.infinity_expansion_order,
    )
    xup_solution = solve_up(
        mode;
        rsin=rsin,
        rsout=rsout,
        tolerance=cfg.homogeneous_tolerance,
        infinity_order=cfg.infinity_expansion_order,
    )

    xin = Vector{ComplexF64}(undef, length(rstar))
    xup = similar(xin)
    _evaluate_component!(xin, xin_solution.numerical_solution, rstar, 1)
    _evaluate_component!(xup, xup_solution.numerical_solution, rstar, 1)

    binc = ComplexF64(amps.binc)
    bref = ComplexF64(amps.bref)
    denominator = ComplexF64(2im * cfg.omega * binc)
    abs(denominator) > 0 || error("vanishing horizon Green denominator")
    wd = weighted_wronskian_diagnostics(
        mode, xin_solution, xup_solution, binc,
    )

    return HorizonKernel(
        mode,
        Float64(rsin), Float64(rsout),
        xin_solution, xup_solution,
        xin, xup,
        binc, bref, denominator,
        ComplexF64(lambda),
        ComplexF64(NativeSN.eta_coefficient(mode, 0)),
        wd.relative_spread,
        wd.relative_offset,
    )
end

function _trapezoid(xs, ys)
    length(xs) == length(ys) || error("integration arrays differ in length")
    total = 0.0 + 0.0im
    @inbounds for i in 2:length(xs)
        total += 0.5 * (xs[i] - xs[i-1]) * (ys[i-1] + ys[i])
    end
    return ComplexF64(total)
end

function _turn_endpoint_value(xup_at_turn, built, cfg::HorizonScatteringConfig)
    points = [
        trajectory.points[argmin(getproperty.(trajectory.points, :r))]
        for trajectory in built.trajectories
    ]
    q2_turn = sum(
        UG.q_distribution_coefficients(
            built.kerr, built.constants, built.mode, point,
            cfg.omega, cfg.m;
            t_origin=built.t_origin,
            phi_origin=built.phi_origin,
        ).total.q2
        for point in points
    )
    radial_prime = UG.radial_potential_derivative(
        built.kerr, built.constants, built.r_turn,
    )
    radial_prime > 0 || error("turning point has nonpositive R'(r_p)")
    sigma_turn = UG.sigma(
        built.kerr, built.r_turn, points[1].theta,
    )
    source_phase_turn = cis(-UG.k_over_delta_antiderivative(
        built.kerr, built.r_turn, cfg.omega, cfg.m,
    ))
    return xup_at_turn * built.particle_mass * source_phase_turn *
           2 * sigma_turn * q2_turn /
           (
               built.r_turn^2 *
               sqrt(built.r_turn^2 + cfg.a^2) *
               sqrt(radial_prime)
           )
end

function _partitioned_horizon_integral(rstar, source, xup, built,
                                       cfg::HorizonScatteringConfig)
    table = built.branch_summed_table
    inner_count = length(rstar) - length(table.r)
    2 <= inner_count < length(rstar) || error("invalid scattering source partition")
    integrand = xup .* source
    inner = _trapezoid(
        @view(rstar[1:inner_count]),
        @view(integrand[1:inner_count]),
    )

    r = table.r
    drstar_du =
        (r .^ 2 .+ cfg.a^2) ./ UG.delta.(Ref(UG.KerrParams(a=cfg.a)), r) .*
        table.dr_dx
    endpoint = _turn_endpoint_value(xup[inner_count], built, cfg)

    outer = 0.0 + 0.0im
    previous_x = 0.0
    previous_value = ComplexF64(endpoint)
    @inbounds for j in eachindex(table.integration_x)
        x = Float64(table.integration_x[j])
        value = integrand[inner_count + j] * drstar_du[j]
        outer += 0.5 * (x - previous_x) * (previous_value + value)
        previous_x = x
        previous_value = value
    end
    return ComplexF64(inner + outer)
end

function _asymptotic_factor(coefficients, omega, radius)
    total = 0.0 + 0.0im
    @inbounds for order in 0:(length(coefficients)-1)
        total += coefficients[order+1] / (omega * radius)^order
    end
    return ComplexF64(total)
end

function _horizon_green_tail(kernel::HorizonKernel, built,
                             cfg::HorizonScatteringConfig)
    cfg.green_tail_correction || return 0.0 + 0.0im
    coefficients = NativeSN.infinity_coefficients(
        kernel.mode, :outgoing; order=cfg.infinity_expansion_order,
    )
    total = 0.0 + 0.0im
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
            UG.k_over_delta_antiderivative(
                built.kerr, radius, cfg.omega, cfg.m,
            ) for radius in r
        ]
        orbit_phase = phase .- radial_phase
        rstar = KerrGeometry.rstar_from_r.(Ref(cfg.a), r)
        combined_phase = orbit_phase .+ cfg.omega .* rstar
        radial_phase_rate = UG.phase_integrand.(
            Ref(built.kerr), r, cfg.omega, cfg.m,
        )
        orbit_rate = source_rate .- radial_phase_rate
        rstar_rate = (r .^ 2 .+ cfg.a^2) ./ UG.delta.(Ref(built.kerr), r)
        combined_rate = orbit_rate .+ cfg.omega .* rstar_rate
        factors = ComplexF64[
            _asymptotic_factor(coefficients, cfg.omega, radius) for radius in r
        ]

        reconstructed = factors[end] * cis(cfg.omega * rstar[end])
        numerical = kernel.xup[end]
        reconstruction_error = abs(reconstructed - numerical) /
                               max(abs(numerical), eps(Float64))
        reconstruction_error < 5.0e-8 || error(
            "Xup asymptotic basis fails at the horizon Green-tail boundary: " *
            "relative error=$reconstruction_error",
        )

        radial_prefactor = inv.(r .^ 2 .* sqrt.(r .^ 2 .+ cfg.a^2))
        amplitude = built.particle_mass .* radial_prefactor .* smooth_W .* factors
        total += EQS.fitted_oscillatory_tail(
            r,
            amplitude .* cis.(combined_phase),
            cis.(combined_phase),
            combined_rate;
            order=cfg.horizon_green_tail_order,
            max_points=fit_count,
        )
    end
    return ComplexF64(total)
end

function compute_horizon_sn_amplitude(cfg::HorizonScatteringConfig;
                                      orbit_cache=nothing)
    built = build_centered_scattering_source(cfg; orbit_cache=orbit_cache)
    wcfg = built.waveform_config
    rstar, source = FR.source_grid_values(wcfg, built)
    rstar = Float64.(rstar)
    source = ComplexF64.(source)
    kernel = _build_horizon_kernel(rstar, built, cfg)

    numeric = _partitioned_horizon_integral(
        rstar, source, kernel.xup, built, cfg,
    )
    tail = _horizon_green_tail(kernel, built, cfg)
    integral = ComplexF64(numeric + tail)
    amplitude = ComplexF64(integral / kernel.denominator)
    _finite_complex(amplitude) || error("non-finite horizon SN amplitude")

    return HorizonSNResult(
        cfg,
        built,
        rstar,
        source,
        kernel.xup,
        numeric,
        tail,
        integral,
        amplitude,
        kernel.denominator,
        kernel.lambda,
        kernel.binc,
        kernel.bref,
        kernel.wronskian_relative_spread,
        kernel.wronskian_relative_offset,
    )
end



function _direct_problem_context(cfg::HorizonScatteringConfig)
    wcfg = _waveform_config(cfg)
    scfg = FR.source_config(wcfg)
    scfg.particle_mass = cfg.particle_mass
    kerr = UG.KerrParams(a=cfg.a)
    constants = UG.GeodesicConstants(
        energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q,
    )
    r_outer = wcfg.r_outer_min
    roots = UG.radial_turning_points(kerr, constants; r_max=r_outer, samples=5000)
    isempty(roots) && error("ordinary scattering requires an outer radial turning point")
    r_turn = last(roots)
    outer = UG.finite_outer_state_from_asymptote(
        kerr, constants, r_outer,
        cfg.theta_infinity, cfg.phi_infinity, cfg.theta_sign,
    )
    harmonic, angular_mode = EQS.scattering_mode(kerr, scfg)
    equatorial = abs(cfg.carter_q) <= 1e-13 &&
                 abs(cfg.theta_infinity - pi / 2) <= 1e-12
    theta_bounds = equatorial ? nothing :
        UG.theta_allowed_interval(kerr, constants, Float64(outer.theta))
    polar = equatorial ? nothing : UG.polar_phase_parameters(
        kerr, constants, Float64(outer.theta), Float64(outer.theta_sign), theta_bounds,
    )
    schwarzschild_equatorial = equatorial && abs(cfg.a) <= 1e-14
    z_outer = sqrt(r_outer - r_turn)
    radial_prime = UG.radial_potential_derivative(kerr, constants, r_turn)
    radial_prime > 0 || error("outer radial turning point must have positive R'(r_p)")
    return (
        waveform_config=wcfg, source_config=scfg,
        kerr=kerr, constants=constants, harmonic=harmonic,
        angular_mode=angular_mode, r_turn=Float64(r_turn),
        r_outer=Float64(r_outer), outer_state=outer,
        equatorial=equatorial, schwarzschild_equatorial=schwarzschild_equatorial,
        theta_bounds=theta_bounds, polar=polar,
        z_outer=Float64(z_outer), radial_prime=Float64(radial_prime),
    )
end

function _direct_dlambda_dz(context, radial_sign::Float64, z0::Float64, r::Float64)
    if context.schwarzschild_equatorial
        E = context.constants.energy
        L = context.constants.lz
        rp = context.r_turn
        p2 = E^2 - 1
        q = p2 * r^2 + (2 + p2 * rp) * r +
            (-L^2 + 2rp + p2 * rp^2)
        q > 0 || error("Schwarzschild radial quotient became nonpositive")
        return 2 / (radial_sign * sqrt(r * q))
    end

    E = context.constants.energy
    L = context.constants.lz
    Q = context.constants.carter_q
    a = context.kerr.a
    rp = context.r_turn
    p2 = E^2 - 1
    radial_r2_coefficient = a^2 * p2 - L^2 - Q
    R2 = 12p2 * rp^2 + 12rp + 2radial_r2_coefficient
    R3 = 24p2 * rp + 12
    u = z0^2
    quotient = ((p2 * u + R3 / 6) * u + R2 / 2) * u +
               context.radial_prime
    quotient > 0 || error(
        "generic Kerr radial-anomaly quotient became nonpositive: " *
        "z=$z0, r=$r, quotient=$quotient",
    )
    return 2 / (radial_sign * sqrt(quotient))
end


function _direct_geo_rhs(y, p, z)
    context, radial_sign = p
    z0 = max(Float64(z), 0.0)
    r = context.r_turn + z0^2
    if context.equatorial
        theta = pi / 2
        theta_sign = 1.0
        polar_phase = pi / 2
    else
        polar_phase = y[2]
        theta = UG.theta_from_polar_phase(polar_phase, context.polar)
        theta_sign = UG.polar_sign_from_phase(polar_phase)
    end
    state = UG.BLState(
        r=r, theta=theta, t=y[1], phi=y[3],
        ur_sign=radial_sign, utheta_sign=theta_sign,
    )
    velocity = UG.four_velocity(context.kerr, context.constants, state)
    dlambda_dz = _direct_dlambda_dz(context, radial_sign, z0, r)
    sigma = UG.sigma(context.kerr, r, theta)
    dt_dz = sigma * velocity.ut * dlambda_dz
    dphi_dz = sigma * velocity.uphi * dlambda_dz
    dpolar_dz = if context.equatorial
        0.0
    else
        polar_rate = sqrt(max(
            context.polar.q_over_zmax -
            context.polar.a2_one_minus_e2 * context.polar.zmax * sin(polar_phase)^2,
            0.0,
        ))
        -polar_rate * dlambda_dz
    end
    return SVector(dt_dz, dpolar_dz, dphi_dz)
end

function _solve_direct_geodesic(context, cfg::HorizonScatteringConfig)
    angular_coordinate = context.equatorial ? pi / 2 : context.polar.phase0
    y_outer = SVector(
        0.0, angular_coordinate, Float64(context.outer_state.phi),
    )
    incoming_problem = ODEProblem(
        _direct_geo_rhs, y_outer,
        (context.z_outer, 0.0), (context, -1.0),
    )
    incoming = solve(
        incoming_problem,
        AutoVern9(Rosenbrock23(autodiff=false));
        reltol=cfg.direct_geodesic_tolerance,
        abstol=cfg.direct_geodesic_tolerance,
        dtmax=cfg.direct_geodesic_max_step,
        maxiters=cfg.direct_geodesic_maxiters,
    )
    successful_retcode(incoming) ||
        error("incoming adaptive scattering geodesic failed: $(incoming.retcode)")
    turn = incoming(0.0)
    all(isfinite, turn) || error(
        "incoming direct geodesic reached a nonfinite turning-point state: $turn",
    )
    outgoing_problem = ODEProblem(
        _direct_geo_rhs, turn,
        (0.0, context.z_outer), (context, 1.0),
    )
    outgoing = solve(
        outgoing_problem,
        AutoVern9(Rosenbrock23(autodiff=false));
        reltol=cfg.direct_geodesic_tolerance,
        abstol=cfg.direct_geodesic_tolerance,
        dtmax=cfg.direct_geodesic_max_step,
        maxiters=cfg.direct_geodesic_maxiters,
    )
    successful_retcode(outgoing) ||
        error("outgoing adaptive scattering geodesic failed: $(outgoing.retcode)")
    outer_state = outgoing(context.z_outer)
    all(isfinite, outer_state) || error(
        "outgoing direct geodesic reached a nonfinite outer state: $outer_state",
    )
    return (
        incoming=incoming, outgoing=outgoing,
        t_origin=Float64(turn[1]), phi_origin=Float64(turn[3]),
        incoming_saved_steps=length(incoming.t),
        outgoing_saved_steps=length(outgoing.t),
    )
end

function build_direct_orbit_cache(cfg::HorizonScatteringConfig)
    abs(cfg.omega) > 0 || error("direct orbit cache requires nonzero omega")
    context = _direct_problem_context(cfg)
    geodesic = _solve_direct_geodesic(context, cfg)
    return DirectHorizonOrbitCache(
        cfg.a, cfg.energy, cfg.lz, cfg.carter_q,
        cfg.theta_infinity, cfg.phi_infinity, cfg.theta_sign,
        abs(cfg.omega), cfg.asymptotic_match_phase, cfg.r_outer_floor,
        cfg.direct_geodesic_tolerance, cfg.direct_geodesic_max_step,
        context.r_outer, context.r_turn, geodesic,
    )
end

@inline function _cache_close(a::Real, b::Real; rtol::Float64=5e-12, atol::Float64=5e-13)
    return abs(Float64(a) - Float64(b)) <= atol + rtol * max(abs(Float64(a)), abs(Float64(b)), 1.0)
end

function _validate_direct_orbit_cache(cache::DirectHorizonOrbitCache,
                                      context,
                                      cfg::HorizonScatteringConfig)
    checks = (
        (:a, cache.a, cfg.a),
        (:energy, cache.energy, cfg.energy),
        (:lz, cache.lz, cfg.lz),
        (:carter_q, cache.carter_q, cfg.carter_q),
        (:theta_infinity, cache.theta_infinity, cfg.theta_infinity),
        (:phi_infinity, cache.phi_infinity, cfg.phi_infinity),
        (:theta_sign, cache.theta_sign, cfg.theta_sign),
        (:omega_abs, cache.omega_abs, abs(cfg.omega)),
        (:asymptotic_match_phase, cache.asymptotic_match_phase, cfg.asymptotic_match_phase),
        (:r_outer_floor, cache.r_outer_floor, cfg.r_outer_floor),
        (:direct_geodesic_tolerance, cache.direct_geodesic_tolerance,
         cfg.direct_geodesic_tolerance),
        (:direct_geodesic_max_step, cache.direct_geodesic_max_step,
         cfg.direct_geodesic_max_step),
        (:r_outer, cache.r_outer, context.r_outer),
        (:r_turn, cache.r_turn, context.r_turn),
    )
    for (name, cached, requested) in checks
        _cache_close(cached, requested) || error(
            "direct orbit cache mismatch for $(name): cached=$(cached), requested=$(requested)",
        )
    end
    return cache
end

function _direct_state(context, branch_solution, z::Float64, radial_sign::Float64)
    y = branch_solution(z)
    r = context.r_turn + z^2
    if context.equatorial
        theta = pi / 2
        theta_sign = 1.0
    else
        polar_phase = y[2]
        theta = UG.theta_from_polar_phase(polar_phase, context.polar)
        theta_sign = UG.polar_sign_from_phase(polar_phase)
    end
    state = UG.BLState(
        r=r, theta=theta, t=y[1], phi=y[3],
        ur_sign=radial_sign, utheta_sign=theta_sign,
    )
    velocity = UG.four_velocity(context.kerr, context.constants, state)
    projection = UG.source_projections(context.kerr, state, velocity)
    return (
        r=Float64(r), theta=Float64(theta), t=Float64(y[1]), phi=Float64(y[3]),
        velocity=velocity, projection=projection,
    )
end

function _sn_inverse_coefficients(mode::NativeSN.SNMode, r::Float64)
    Delta = NativeSN.delta(mode, r)
    Deltap = 2r - 2
    radial = r^2 + mode.a^2
    root = sqrt(radial)
    alpha, alphap, _, beta, betap, betapp = NativeSN.alpha_beta_data(mode, r, Delta)
    determinant = NativeSN.eta(mode, r)
    determinantp = NativeSN.eta_prime(mode, r)

    h = Delta / root
    hp = Deltap / root - Delta * r / radial^(3 / 2)
    hpp = 2 / root - 2 * Deltap * r / radial^(3 / 2) -
           Delta / radial^(3 / 2) + 3 * Delta * r^2 / radial^(5 / 2)

    p = beta / Delta
    pp = betap / Delta - beta * Deltap / Delta^2
    A = alpha + betap / Delta
    Ap = alphap + betapp / Delta - betap * Deltap / Delta^2

    P = (A * h - p * hp) / determinant
    Pp = (Ap * h + (A - pp) * hp - p * hpp) / determinant -
         P * determinantp / determinant

    g = p * root
    gp = pp * root + p * r / root
    Q = -g / determinant
    Qp = -gp / determinant - Q * determinantp / determinant
    return P, Pp, Q, Qp
end

function _sn_to_teukolsky(mode::NativeSN.SNMode, r::Float64,
                           X::ComplexF64, Xstar::ComplexF64)
    Delta = NativeSN.delta(mode, r)
    radial = r^2 + mode.a^2
    D = Delta / radial
    F, U = NativeSN.potentials(mode, r)
    P, Pp, Q, Qp = _sn_inverse_coefficients(mode, r)

    R = P * X + Q * Xstar
    Rp = (Pp + Q * U / D) * X +
         (P / D + Qp + Q * F / D) * Xstar
    VT = NativeSN.teukolsky_potential(mode, r, Delta)
    Rpp = ((2r - 2) * Rp + VT * R) / Delta
    return ComplexF64(R), ComplexF64(Rp), ComplexF64(Rpp)
end

@inline _ctrans_minus2(mode::NativeSN.SNMode) =
    ComplexF64(-4 * mode.omega^2 / NativeSN.eta_coefficient(mode, 0))

@inline _binc_minus2(mode::NativeSN.SNMode, binc_sn::ComplexF64) =
    ComplexF64(-binc_sn / (4 * mode.omega^2))

function _regular_horizon_teukolsky_factor(mode::NativeSN.SNMode, x)
    omega = mode.omega
    rp = NativeSN.rplus(mode)
    rm = NativeSN.rminus(mode)
    gap = rp - rm
    r = rp + x
    Delta = x * (gap + x)
    radial = r^2 + mode.a^2
    k = omega - mode.m * mode.a / (2rp)

    hcoeff = NativeSN.horizon_coefficients(mode; order=2)
    h1 = hcoeff[2] * omega
    h2 = hcoeff[3] * omega^2
    H = 1 + h1 * x + h2 * x^2
    Hp = h1 + 2h2 * x

    n0, n1, n2, n3, n4, n5, n6 = mode.numerator_coefficients
    numerator = ((((((n6 * r + n5) * r + n4) * r + n3) * r + n2) * r + n1) * r + n0)

    b0, b1, b2, b3 = mode.bracket_coefficients
    bracket = ((b3 * r + b2) * r + b1) * r + b0
    bracketp = b1 + 2b2 * r + 3b3 * r^2

    p = 2 * bracket / r
    pp = 2 * (bracketp * r - bracket) / r^2
    regular = numerator / r^2 + Delta * pp + p * Delta * r / radial +
              im * k * p * radial

    return (regular * H - p * Delta * Hp) /
           (NativeSN.eta(mode, r) * sqrt(radial))
end

function _btrans_minus2(mode::NativeSN.SNMode)
    rp = NativeSN.rplus(mode)
    rm = NativeSN.rminus(mode)
    gap = rp - rm
    f(x) = _regular_horizon_teukolsky_factor(mode, x)

    f0 = ComplexF64(f(0.0))
    f1 = ComplexF64(
        ForwardDiff.derivative(x -> real(f(x)), 0.0),
        ForwardDiff.derivative(x -> imag(f(x)), 0.0),
    )
    abs(f0) <= 1e-9 && abs(f1) <= 1e-9 || error(
        "SN horizon transformation failed the Delta^2 regularity audit: " *
        "f0=$(f0), f1=$(f1)",
    )

    hreal = ForwardDiff.hessian(v -> real(f(v[1])), [0.0])[1, 1]
    himag = ForwardDiff.hessian(v -> imag(f(v[1])), [0.0])[1, 1]
    second = ComplexF64(hreal, himag)
    return second / (2 * gap^2)
end

function _direct_homogeneous_kernel(context, cfg::HorizonScatteringConfig)
    rout_rs = KerrGeometry.rstar_from_r(cfg.a, context.r_outer)
    stub = Float64[cfg.homogeneous_rsin, rout_rs]
    return _build_horizon_kernel(stub, context, cfg)
end

function _direct_teukolsky_value(sample, context, kernel::HorizonKernel,
                                  geodesic, cfg::HorizonScatteringConfig)
    rs = KerrGeometry.rstar_from_r(cfg.a, sample.r)
    X = kernel.xup_solution.numerical_solution(rs)
    R, Rp, Rpp = _sn_to_teukolsky(
        kernel.mode, sample.r, ComplexF64(X[1]), ComplexF64(X[2]),
    )
    terms = UG.teukolsky_a_terms_from_kinematics(
        context.kerr, sample.r, sample.theta,
        sample.projection.N, sample.projection.Mbar,
        context.angular_mode, cfg.omega,
    )
    A0 = terms.nn0 + terms.nm0 + terms.mm0
    A1 = terms.nm1 + terms.mm1
    A2 = terms.mm2
    phase = cis(
        cfg.omega * (sample.t - geodesic.t_origin) -
        cfg.m * (sample.phi - geodesic.phi_origin),
    )
    return ComplexF64(
        cfg.particle_mass * phase * (R * A0 - Rp * A1 + Rpp * A2),
    )
end

function _gauss_legendre_rule(order::Int)
    order >= 2 || error("Gauss-Legendre order must be at least two")
    diagonal = zeros(Float64, order)
    offdiagonal = Float64[
        j / sqrt(4j^2 - 1) for j in 1:(order - 1)
    ]
    decomposition = eigen(SymTridiagonal(diagonal, offdiagonal))
    nodes = Float64.(decomposition.values)
    weights = Float64.(2 .* decomposition.vectors[1, :].^2)
    return nodes, weights
end

function _direct_gauss_panels(cfg::HorizonScatteringConfig)
    base = cfg.direct_gauss_panels_per_600
    base >= 1 || error("direct_gauss_panels_per_600 must be positive")
    return max(64, ceil(Int, base * cfg.asymptotic_match_phase / 600.0))
end

function _direct_integrand_z(z::Float64, radial_sign::Float64,
                             branch_solution, context, kernel, geodesic,
                             cfg::HorizonScatteringConfig)
    sample = _direct_state(context, branch_solution, z, radial_sign)
    abs(sample.velocity.ur) > 0 || error("Gauss node landed on the radial turn")
    jacobian = 2z / abs(sample.velocity.ur)
    return _direct_teukolsky_value(sample, context, kernel, geodesic, cfg) * jacobian
end

function _direct_branch_numeric(branch_solution, radial_sign::Float64,
                                context, kernel, geodesic,
                                cfg::HorizonScatteringConfig;
                                panels::Int=_direct_gauss_panels(cfg))
    nodes, weights = _gauss_legendre_rule(cfg.direct_gauss_order)
    zmax = context.z_outer
    total = 0.0 + 0.0im
    @inbounds for panel in 1:panels
        za = zmax * (panel - 1) / panels
        zb = zmax * panel / panels
        center = (za + zb) / 2
        half = (zb - za) / 2
        local_sum = 0.0 + 0.0im
        for q in eachindex(nodes)
            z = center + half * nodes[q]
            local_sum += weights[q] * _direct_integrand_z(
                z, radial_sign, branch_solution,
                context, kernel, geodesic, cfg,
            )
        end
        total += half * local_sum
    end
    return ComplexF64(total)
end

function _direct_branch_tail(branch_solution, radial_sign::Float64,
                             context, kernel, geodesic,
                             cfg::HorizonScatteringConfig)
    n = cfg.direct_tail_fit_points
    n >= max(32, 2 * cfg.direct_tail_order + 2) ||
        error("direct_tail_fit_points is too small")
    0 < cfg.direct_tail_window_fraction < 0.2 ||
        error("direct_tail_window_fraction must lie between zero and 0.2")
    R = context.r_outer
    r0 = R - cfg.direct_tail_window_fraction * (R - context.r_turn)
    radii = collect(range(r0, R; length=n))
    values = Vector{ComplexF64}(undef, n)
    phases = Vector{ComplexF64}(undef, n)
    rates = Vector{ComplexF64}(undef, n)
    @inbounds for i in eachindex(radii)
        r = radii[i]
        z = sqrt(r - context.r_turn)
        sample = _direct_state(context, branch_solution, z, radial_sign)
        abs(sample.velocity.ur) > 0 || error("outer tail encountered a radial turn")
        direct = _direct_teukolsky_value(sample, context, kernel, geodesic, cfg) /
                 abs(sample.velocity.ur)
        orbit_argument = cfg.omega * (sample.t - geodesic.t_origin) -
                         cfg.m * (sample.phi - geodesic.phi_origin)
        phase_argument = orbit_argument + cfg.omega * KerrGeometry.rstar_from_r(cfg.a, r)
        rate = cfg.omega * sample.velocity.ut / sample.velocity.ur -
               cfg.m * sample.velocity.uphi / sample.velocity.ur +
               cfg.omega * (r^2 + cfg.a^2) / UG.delta(context.kerr, r)
        values[i] = direct
        phases[i] = cis(phase_argument)
        rates[i] = ComplexF64(rate)
    end
    return ComplexF64(_ccs_fitted_tail(
        Float64.(radii), values, phases, rates;
        order=cfg.direct_tail_order, centered=false,
    ))
end

function compute_horizon_teukolsky_amplitude(cfg::HorizonScatteringConfig;
                                              orbit_cache=nothing,
                                              gauss_panels=nothing)
    context = _direct_problem_context(cfg)
    geodesic = if orbit_cache === nothing
        _solve_direct_geodesic(context, cfg)
    else
        orbit_cache isa DirectHorizonOrbitCache ||
            error("orbit_cache must be a DirectHorizonOrbitCache")
        _validate_direct_orbit_cache(orbit_cache, context, cfg)
        orbit_cache.geodesic
    end
    kernel = _direct_homogeneous_kernel(context, cfg)
    panels = gauss_panels === nothing ? _direct_gauss_panels(cfg) : Int(gauss_panels)
    panels >= 1 || error("gauss_panels must be positive")

    incoming = _direct_branch_numeric(
        geodesic.incoming, -1.0, context, kernel, geodesic, cfg; panels=panels,
    )
    outgoing = _direct_branch_numeric(
        geodesic.outgoing, 1.0, context, kernel, geodesic, cfg; panels=panels,
    )
    incoming_tail = _direct_branch_tail(
        geodesic.incoming, -1.0, context, kernel, geodesic, cfg,
    )
    outgoing_tail = _direct_branch_tail(
        geodesic.outgoing, 1.0, context, kernel, geodesic, cfg,
    )
    integral = ComplexF64(incoming + outgoing + incoming_tail + outgoing_tail)

    btrans = _btrans_minus2(kernel.mode)
    binc_teuk = _binc_minus2(kernel.mode, kernel.binc)
    ctrans = _ctrans_minus2(kernel.mode)
    denominator = 2im * cfg.omega * binc_teuk * ctrans
    abs(denominator) > 0 || error("vanishing Teukolsky horizon Green denominator")
    z_h = ComplexF64(btrans * integral / denominator)
    _finite_complex(z_h) || error("non-finite direct Teukolsky horizon amplitude")

    return HorizonTeukolskyResult(
        cfg, context.r_outer, context.r_turn, kernel.lambda,
        incoming, outgoing, incoming_tail, outgoing_tail, integral, z_h,
        btrans, binc_teuk, ctrans, kernel.binc,
        kernel.wronskian_relative_spread, kernel.wronskian_relative_offset,
        panels, cfg.direct_gauss_order, cfg.direct_tail_fit_points,
        geodesic.incoming_saved_steps, geodesic.outgoing_saved_steps,
    )
end

function horizon_direct_convergence(cfg::HorizonScatteringConfig;
                                    panel_factors=(1, 2, 4))
    base = _direct_gauss_panels(cfg)
    results = HorizonTeukolskyResult[]
    for factor in panel_factors
        factor > 0 || error("panel factors must be positive")
        push!(results, compute_horizon_teukolsky_amplitude(
            cfg; gauss_panels=Int(round(base * factor)),
        ))
    end
    errors = Float64[]
    for i in 1:(length(results) - 1)
        a = results[i].z_h_minus2
        b = results[i + 1].z_h_minus2
        push!(errors, Float64(abs(a - b) / max(abs(b), eps(Float64))))
    end
    return (results=results, relative_successive_errors=errors)
end


function _starobinsky_abs2_minus2(mode::NativeSN.SNMode)
    Q = mode.lambda + 2
    a = mode.a
    m = mode.m
    omega = mode.omega
    return Float64(
        (Q^2 + 4a * omega * m - 4a^2 * omega^2) *
        ((Q - 2)^2 + 36a * omega * m - 36a^2 * omega^2) +
        (2Q - 1) * (96a^2 * omega^2 - 48a * omega * m) +
        144omega^2 * (1 - a^2)
    )
end

function _starobinsky_constant_minus2(mode::NativeSN.SNMode)
    abs2C = _starobinsky_abs2_minus2(mode)
    imagC = 12 * mode.omega
    real2 = abs2C - imagC^2
    tol = 256eps(Float64) * max(abs2C, imagC^2, 1.0)
    real2 >= -tol || error("negative Starobinsky Re(C)^2: $real2")
    realC = sqrt(max(real2, 0.0))
    C = ComplexF64(realC, imagC)
    abs(abs2(C) - abs2C) <= 1e-11 * max(abs2C, 1.0) ||
        error("phaseful Starobinsky constant fails |C|^2 audit")
    return C, abs2C
end

function horizon_physical_from_teukolsky(result::HorizonTeukolskyResult)
    cfg = result.cfg
    mode = NativeSN.SNMode(cfg.a, cfg.m, cfg.omega, real(result.lambda))
    C, Cabs2 = _starobinsky_constant_minus2(mode)

    rp = NativeSN.rplus(mode)
    root = sqrt(1 - cfg.a^2)
    omega_h = cfg.a / (2rp)
    k = cfg.omega - cfg.m * omega_h
    epsilon = root / (4rp)
    kappa_h = 2epsilon
    z_h = result.z_h_minus2

    y_hole = 64 * (2rp)^4 * (im * k) * (k^2 + 4epsilon^2) *
             (-im * k + 4epsilon) * z_h / C

    psi0_hh = ComplexF64(y_hole / (4 * (2rp)^2))
    Psi_h = -psi0_hh
    shear_h = ComplexF64(-psi0_hh / (im * k + 2epsilon))

    alpha = Float64(
        128 * cfg.omega * k * (k^2 + 4epsilon^2) * (k^2 + 16epsilon^2) *
        (2rp)^5 / Cabs2
    )
    dE = Float64(alpha * abs2(z_h))

    dE_shear = if abs(k) <= 1e-13
        0.0
    else
        Float64(cfg.omega * rp / k * abs2(shear_h))
    end
    flux_rel = abs(dE - dE_shear) / max(abs(dE), abs(dE_shear), eps(Float64))
    flux_rel <= 2e-10 || error(
        "Teukolsky and Hawking--Hartle horizon flux normalizations disagree: $flux_rel",
    )

    dJ = Float64(cfg.m / cfg.omega * dE)
    dA = Float64(8pi / kappa_h * (dE - omega_h * dJ))
    area_tol = 1e-11 * max(abs(8pi / kappa_h * dE), 1e-300)
    dA >= -area_tol || error("negative horizon area spectrum: $dA")
    dA = max(dA, 0.0)

    return HorizonPhysicalResult(
        result, C, Cabs2, omega_h, k, epsilon, kappa_h,
        ComplexF64(y_hole), psi0_hh, Psi_h, shear_h, alpha,
        dE, dE_shear, dJ, dA, Float64(flux_rel),
    )
end

function compute_horizon_physical(cfg::HorizonScatteringConfig)
    return horizon_physical_from_teukolsky(compute_horizon_teukolsky_amplitude(cfg))
end



function horizon_edge_scan(cfg::HorizonScatteringConfig;
                           k_offsets=(-0.002, -0.001, 0.0, 0.001, 0.002),
                           gauss_panels=nothing)
    rp = KerrGeometry.rplus(cfg.a)
    omega_h = cfg.a / (2rp)
    edge = cfg.m * omega_h
    rows = NamedTuple[]
    for offset in k_offsets
        local_cfg = deepcopy(cfg)
        local_cfg.omega = Float64(edge + offset)
        local_cfg.omega != 0 || error("edge scan encountered omega=0")
        radial = compute_horizon_teukolsky_amplitude(
            local_cfg; gauss_panels=gauss_panels,
        )
        physical = horizon_physical_from_teukolsky(radial)
        k = physical.k
        push!(rows, (
            k=Float64(k),
            omega=Float64(local_cfg.omega),
            result=physical,
            psi_over_k=abs(k) <= 1e-13 ? nothing : physical.Psi_h / k,
            shear_over_k=abs(k) <= 1e-13 ? nothing : physical.shear_h / k,
            dE_over_k=abs(k) <= 1e-13 ? nothing : physical.dE_domega / k,
            dA_over_k2=abs(k) <= 1e-13 ? nothing : physical.dA_domega / k^2,
        ))
    end
    return (omega_h=Float64(omega_h), edge=Float64(edge), rows=rows)
end


function schwarzschild_pm_dE_domega(energy::Real, b::Real, u::Real;
                                     particle_mass::Real=1.0)
    E = Float64(energy)
    impact = Float64(b)
    uu = Float64(u)
    mu = Float64(particle_mass)
    E > 1 || error("Schwarzschild PM benchmark requires E>1")
    impact > 0 || error("impact parameter must be positive")
    uu > 0 || error("dimensionless frequency u must be positive")
    p = sqrt(E^2 - 1)
    K0 = besselk(0, uu)
    K1 = besselk(1, uu)
    bracket =
        uu^2 * (1 - 2E^2 + 2E^4) * K0^2 +
        uu * (1 - 8E^2 + 8E^4) * K0 * K1 +
        (1 - 8E^2 + 8E^4 + uu^2 * (2E^2 - 1)) * K1^2
    dE_du = 128 * mu^2 * uu^4 * p * bracket / (45pi * impact^7)
    return Float64((impact / p) * dE_du)
end

function horizon_schwarzschild_pm_benchmark(;
        energy::Real=1.2,
        b_values=(50.0, 100.0, 200.0),
        u_values=(1.0, 2.0),
        particle_mass::Real=1.0,
        asymptotic_match_phase::Real=250.0,
        gauss_panels=nothing,
    )
    E = Float64(energy)
    p = sqrt(E^2 - 1)
    rows = NamedTuple[]
    for u in u_values
        uu = Float64(u)
        for b in b_values
            impact = Float64(b)
            omega = uu * p / impact
            one_helicity = 0.0
            modes = NamedTuple[]
            for m in -2:2
                cfg = HorizonScatteringConfig(
                    a=0.0,
                    ell=2,
                    m=m,
                    omega=omega,
                    energy=E,
                    lz=impact * p,
                    carter_q=0.0,
                    theta_infinity=pi / 2,
                    phi_infinity=0.0,
                    theta_sign=1.0,
                    particle_mass=Float64(particle_mass),
                    asymptotic_match_phase=Float64(asymptotic_match_phase),
                    r_outer_floor=max(400.0, 4impact),
                    allow_theta_turns=false,
                )
                radial = compute_horizon_teukolsky_amplitude(
                    cfg; gauss_panels=gauss_panels,
                )
                physical = horizon_physical_from_teukolsky(radial)
                one_helicity += physical.dE_domega
                push!(modes, (m=m, result=physical))
            end
            analytic = schwarzschild_pm_dE_domega(
                E, impact, uu; particle_mass=particle_mass,
            )
            helicity_completed = 2 * one_helicity
            push!(rows, (
                u=uu,
                b=impact,
                omega=omega,
                numerical_one_helicity=Float64(one_helicity),
                numerical_helicity_completed=Float64(helicity_completed),
                analytic_leading_pm=Float64(analytic),
                ratio=Float64(helicity_completed / analytic),
                modes=modes,
            ))
        end
    end
    return (energy=E, p_infinity=p, rows=rows)
end



function direct_cancellation_condition(result::HorizonTeukolskyResult)
    numerator = abs(result.incoming_numeric) + abs(result.outgoing_numeric) +
                abs(result.incoming_tail) + abs(result.outgoing_tail)
    return Float64(numerator / max(abs(result.source_integral), eps(Float64)))
end


function horizon_frequency_spectrum(
    cfg::HorizonScatteringConfig;
    frequencies,
    ell_min::Int=2,
    ell_max::Int=4,
    gauss_panels=nothing,
)
    2 <= ell_min <= ell_max || error("require 2 <= ell_min <= ell_max")
    rows = NamedTuple[]
    for omega_value in frequencies
        omega = Float64(omega_value)
        omega > 0 || error("horizon_frequency_spectrum expects positive frequencies")
        cache_cfg = deepcopy(cfg)
        cache_cfg.omega = omega
        orbit_cache = build_direct_orbit_cache(cache_cfg)
        shells = NamedTuple[]
        total_E = 0.0
        total_J = 0.0
        total_A = 0.0
        total_abs_E = 0.0
        condition_weighted_numerator = 0.0
        max_condition = 0.0
        for ell in ell_min:ell_max
            modes = NamedTuple[]
            shell_E = 0.0
            shell_J = 0.0
            shell_A = 0.0
            shell_abs_E = 0.0
            shell_condition_weighted_numerator = 0.0
            shell_max_condition = 0.0
            for m in -ell:ell
                local_cfg = deepcopy(cfg)
                local_cfg.ell = ell
                local_cfg.m = m
                local_cfg.omega = omega
                radial = compute_horizon_teukolsky_amplitude(
                    local_cfg; orbit_cache=orbit_cache, gauss_panels=gauss_panels,
                )
                physical = horizon_physical_from_teukolsky(radial)
                condition = direct_cancellation_condition(radial)
                shell_E += physical.dE_domega
                shell_J += physical.dJ_domega
                shell_A += physical.dA_domega
                energy_weight = abs(physical.dE_domega)
                shell_abs_E += energy_weight
                shell_condition_weighted_numerator += energy_weight * condition
                shell_max_condition = max(shell_max_condition, condition)
                push!(modes, (
                    ell=ell, m=m, omega=omega,
                    result=physical,
                    cancellation_condition=condition,
                ))
            end
            total_E += shell_E
            total_J += shell_J
            total_A += shell_A
            total_abs_E += shell_abs_E
            condition_weighted_numerator += shell_condition_weighted_numerator
            max_condition = max(max_condition, shell_max_condition)
            shell_weighted_condition = shell_condition_weighted_numerator /
                                       max(shell_abs_E, eps(Float64))
            push!(shells, (
                ell=ell,
                dE_domega=Float64(shell_E),
                dJ_domega=Float64(shell_J),
                dA_domega=Float64(shell_A),
                absolute_signed_energy_norm=Float64(shell_abs_E),
                energy_weighted_cancellation_condition=Float64(shell_weighted_condition),
                max_cancellation_condition=Float64(shell_max_condition),
                modes=modes,
            ))
        end
        weighted_condition = condition_weighted_numerator /
                             max(total_abs_E, eps(Float64))
        push!(rows, (
            omega=omega,
            signed_positive_frequency_dE_domega=Float64(total_E),
            signed_positive_frequency_dJ_domega=Float64(total_J),
            signed_positive_frequency_dA_domega=Float64(total_A),
            reality_completed_one_sided_dE_domega=Float64(2 * total_E),
            energy_weighted_cancellation_condition=Float64(weighted_condition),
            max_cancellation_condition=Float64(max_condition),
            shells=shells,
        ))
    end
    return (
        orbit=(
            a=cfg.a, energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q,
            theta_infinity=cfg.theta_infinity, phi_infinity=cfg.phi_infinity,
            theta_sign=cfg.theta_sign, particle_mass=cfg.particle_mass,
        ),
        ell_min=ell_min, ell_max=ell_max, rows=rows,
    )
end

function horizon_spectrum_diagnostics(
    result;
    shell_tolerance::Real=1.0e-3,
    consecutive_shells::Int=2,
    high_endpoint_fraction_tolerance::Real=1.0e-2,
)
    shell_tol = Float64(shell_tolerance)
    endpoint_tol = Float64(high_endpoint_fraction_tolerance)
    shell_tol > 0 || error("shell_tolerance must be positive")
    consecutive_shells >= 1 || error("consecutive_shells must be positive")
    0 < endpoint_tol < 1 || error(
        "high_endpoint_fraction_tolerance must lie between zero and one",
    )
    isempty(result.rows) && error("spectrum result has no frequency rows")

    frequency_rows = NamedTuple[]
    all_multipole_converged = true
    for row in result.rows
        cumulative = 0.0
        small_count = 0
        shell_rows = NamedTuple[]
        for shell in row.shells
            shell_norm = sum(abs(mode.result.dE_domega) for mode in shell.modes)
            cumulative += shell_norm
            fraction = shell_norm / max(cumulative, eps(Float64))
            small_count = fraction < shell_tol ? small_count + 1 : 0
            push!(shell_rows, (
                ell=shell.ell,
                absolute_signed_energy_norm=Float64(shell_norm),
                cumulative_absolute_signed_energy_norm=Float64(cumulative),
                shell_fraction=Float64(fraction),
                below_tolerance=fraction < shell_tol,
                consecutive_below=small_count,
            ))
        end
        multipole_converged = small_count >= consecutive_shells
        all_multipole_converged &= multipole_converged
        weighted_numerator = sum(
            abs(mode.result.dE_domega) * mode.cancellation_condition
            for shell in row.shells for mode in shell.modes
        )
        weighted_denominator = sum(
            abs(mode.result.dE_domega)
            for shell in row.shells for mode in shell.modes
        )
        weighted_condition = weighted_numerator /
                             max(weighted_denominator, eps(Float64))
        push!(frequency_rows, (
            omega=row.omega,
            multipole_converged=multipole_converged,
            shell_diagnostics=shell_rows,
            energy_weighted_cancellation_condition=Float64(weighted_condition),
            max_cancellation_condition=row.max_cancellation_condition,
        ))
    end

    totals = Float64[
        abs(row.reality_completed_one_sided_dE_domega) for row in result.rows
    ]
    peak = maximum(totals)
    endpoint_fraction = totals[end] / max(peak, eps(Float64))
    high_endpoint_pass = endpoint_fraction <= endpoint_tol

    slope = NaN
    tail_integral = NaN
    if length(result.rows) >= 2
        row0 = result.rows[end - 1]
        row1 = result.rows[end]
        e0 = row0.reality_completed_one_sided_dE_domega
        e1 = row1.reality_completed_one_sided_dE_domega
        domega = row1.omega - row0.omega
        if domega > 0 && e0 != 0 && e1 != 0 && signbit(e0) == signbit(e1) &&
           abs(e1) < abs(e0)
            slope = -log(abs(e1 / e0)) / domega
            slope > 0 && (tail_integral = abs(e1) / slope)
        end
    end

    return (
        frequencies=frequency_rows,
        multipole_convergence_pass=all_multipole_converged,
        high_endpoint_fraction=Float64(endpoint_fraction),
        high_endpoint_fraction_tolerance=endpoint_tol,
        high_endpoint_pass=high_endpoint_pass,
        local_high_frequency_log_slope=Float64(slope),
        local_exponential_tail_integral_diagnostic=Float64(tail_integral),
        max_energy_weighted_cancellation_condition=maximum(
            row.energy_weighted_cancellation_condition for row in frequency_rows
        ),
        max_cancellation_condition=maximum(
            row.max_cancellation_condition for row in result.rows
        ),
    )
end

"""
    write_horizon_spectrum_csv(path, result)

Write the frequency-domain diagnostics from `solve_horizon_spectrum` as a
flat, one-row-per-`(omega, ell, m)` CSV.  The solver itself remains no-I/O;
call this explicitly when a persistent diagnostic artifact is required.
"""
function write_horizon_spectrum_csv(path::AbstractString, result)
    hasproperty(result, :spectrum) ||
        error("write_horizon_spectrum_csv expects solve_horizon_spectrum output")
    hasproperty(result, :diagnostics) ||
        error("spectrum result does not contain diagnostics")

    spectrum = result.spectrum
    diagnostics = result.diagnostics
    length(spectrum.rows) == length(diagnostics.frequencies) ||
        error("spectrum and diagnostic frequency rows are inconsistent")

    directory = dirname(abspath(path))
    isdir(directory) || mkpath(directory)
    open(path, "w") do io
        println(io,
            "omega,ell,m,re_lambda,im_lambda,re_Z_H_minus2,im_Z_H_minus2," *
            "re_Psi_H,im_Psi_H,re_shear_H,im_shear_H,omega_h,k," *
            "starobinsky_abs2,absorption_factor,dE_H_domega," *
            "dE_H_domega_shear,dJ_H_domega,dA_H_domega," *
            "flux_relative_difference,direct_cancellation_condition," *
            "wronskian_relative_spread,wronskian_relative_offset," *
            "shell_dE_H_domega,shell_dJ_H_domega,shell_dA_H_domega," *
            "shell_absolute_signed_energy_norm,shell_fraction," *
            "shell_below_tolerance,shell_consecutive_below," *
            "shell_energy_weighted_cancellation_condition," *
            "shell_max_cancellation_condition," *
            "frequency_signed_positive_dE_H_domega," *
            "frequency_reality_completed_one_sided_dE_H_domega," *
            "frequency_multipole_converged," *
            "frequency_energy_weighted_cancellation_condition," *
            "frequency_max_cancellation_condition,high_endpoint_fraction," *
            "high_endpoint_fraction_tolerance,high_endpoint_pass," *
            "endpoint_sequence_pass,accepted_direct_omega_max," *
            "first_conditioning_failure_omega,status",
        )
        for (frequency_row, diagnostic_row) in zip(
            spectrum.rows, diagnostics.frequencies,
        )
            length(frequency_row.shells) == length(diagnostic_row.shell_diagnostics) ||
                error("shell diagnostics are inconsistent at omega=$(frequency_row.omega)")
            for (shell, shell_diagnostic) in zip(
                frequency_row.shells, diagnostic_row.shell_diagnostics,
            )
                for mode in shell.modes
                    physical = mode.result
                    radial = physical.teukolsky
                    fields = (
                        mode.omega, mode.ell, mode.m,
                        real(radial.lambda), imag(radial.lambda),
                        real(radial.z_h_minus2), imag(radial.z_h_minus2),
                        real(physical.Psi_h), imag(physical.Psi_h),
                        real(physical.shear_h), imag(physical.shear_h),
                        physical.omega_h, physical.k, physical.starobinsky_abs2,
                        physical.absorption_factor, physical.dE_domega,
                        physical.dE_domega_shear, physical.dJ_domega,
                        physical.dA_domega, physical.flux_relative_difference,
                        mode.cancellation_condition,
                        radial.wronskian_relative_spread,
                        radial.wronskian_relative_offset,
                        shell.dE_domega, shell.dJ_domega, shell.dA_domega,
                        shell_diagnostic.absolute_signed_energy_norm,
                        shell_diagnostic.shell_fraction,
                        shell_diagnostic.below_tolerance,
                        shell_diagnostic.consecutive_below,
                        shell.energy_weighted_cancellation_condition,
                        shell.max_cancellation_condition,
                        frequency_row.signed_positive_frequency_dE_domega,
                        frequency_row.reality_completed_one_sided_dE_domega,
                        diagnostic_row.multipole_converged,
                        diagnostic_row.energy_weighted_cancellation_condition,
                        diagnostic_row.max_cancellation_condition,
                        diagnostics.high_endpoint_fraction,
                        diagnostics.high_endpoint_fraction_tolerance,
                        diagnostics.high_endpoint_pass,
                        result.endpoint_sequence_pass,
                        result.accepted_direct_omega_max,
                        result.first_conditioning_failure_omega,
                        result.status,
                    )
                    println(io, join(string.(fields), ','))
                end
            end
        end
    end
    return abspath(path)
end


function _validate_horizon_spectrum_config(cfg::HorizonSpectrumConfig)
    cfg.base.energy > 1 || error("ordinary unbound scattering requires energy > 1")
    0 <= abs(cfg.base.a) < 1 || error("require |a| < 1")
    cfg.frequency_grid in ("logarithmic", "linear") ||
        error("frequency_grid must be logarithmic or linear")
    cfg.frequency_count >= 2 || error("frequency_count must be at least two")
    cfg.omega_min > 0 || error("omega_min must be positive")
    cfg.omega_max > cfg.omega_min || error("omega_max must exceed omega_min")
    2 <= cfg.ell_min <= cfg.auto_l_max || error("require 2 <= ell_min <= auto_l_max")
    cfg.shell_tolerance > 0 || error("shell_tolerance must be positive")
    cfg.consecutive_shells >= 1 || error("consecutive_shells must be positive")
    0 < cfg.high_endpoint_fraction_tolerance < 1 || error(
        "high_endpoint_fraction_tolerance must lie between zero and one",
    )
    cfg.high_endpoint_consecutive_points >= 1 ||
        error("high_endpoint_consecutive_points must be positive")
    cfg.cancellation_condition_limit > 1 ||
        error("cancellation_condition_limit must exceed one")
    cfg.gauss_panels === nothing || cfg.gauss_panels > 0 ||
        error("gauss_panels must be positive")
    return cfg
end

function horizon_frequency_grid(cfg::HorizonSpectrumConfig)
    _validate_horizon_spectrum_config(cfg)
    if !isempty(cfg.frequencies)
        values = sort!(unique(Float64.(cfg.frequencies)))
        isempty(values) && error("explicit frequency list is empty")
        all(>(0), values) || error("all frequencies must be positive")
        return values
    end
    if cfg.frequency_grid == "linear"
        return collect(range(cfg.omega_min, cfg.omega_max; length=cfg.frequency_count))
    end
    logs = range(log(cfg.omega_min), log(cfg.omega_max); length=cfg.frequency_count)
    return Float64[exp(x) for x in logs]
end

function _adaptive_horizon_frequency_row(cfg::HorizonSpectrumConfig, omega::Float64)
    cache_cfg = deepcopy(cfg.base)
    cache_cfg.omega = omega
    orbit_cache = build_direct_orbit_cache(cache_cfg)
    shells = NamedTuple[]
    total_E = 0.0
    total_J = 0.0
    total_A = 0.0
    cumulative_abs_E = 0.0
    condition_weighted_numerator = 0.0
    small_shells = 0
    max_condition = 0.0

    for ell in cfg.ell_min:cfg.auto_l_max
        modes = NamedTuple[]
        shell_E = 0.0
        shell_J = 0.0
        shell_A = 0.0
        shell_abs_E = 0.0
        shell_condition_weighted_numerator = 0.0
        shell_max_condition = 0.0
        for m in -ell:ell
            local_cfg = deepcopy(cfg.base)
            local_cfg.ell = ell
            local_cfg.m = m
            local_cfg.omega = omega
            radial = compute_horizon_teukolsky_amplitude(
                local_cfg; orbit_cache=orbit_cache, gauss_panels=cfg.gauss_panels,
            )
            physical = horizon_physical_from_teukolsky(radial)
            condition = direct_cancellation_condition(radial)
            shell_E += physical.dE_domega
            shell_J += physical.dJ_domega
            shell_A += physical.dA_domega
            energy_weight = abs(physical.dE_domega)
            shell_abs_E += energy_weight
            shell_condition_weighted_numerator += energy_weight * condition
            shell_max_condition = max(shell_max_condition, condition)
            push!(modes, (
                ell=ell, m=m, omega=omega, result=physical,
                cancellation_condition=Float64(condition),
            ))
        end

        total_E += shell_E
        total_J += shell_J
        total_A += shell_A
        cumulative_abs_E += shell_abs_E
        condition_weighted_numerator += shell_condition_weighted_numerator
        fraction = shell_abs_E / max(cumulative_abs_E, eps(Float64))
        small_shells = fraction < cfg.shell_tolerance ? small_shells + 1 : 0
        max_condition = max(max_condition, shell_max_condition)
        shell_weighted_condition = shell_condition_weighted_numerator /
                                   max(shell_abs_E, eps(Float64))
        push!(shells, (
            ell=ell,
            dE_domega=Float64(shell_E),
            dJ_domega=Float64(shell_J),
            dA_domega=Float64(shell_A),
            absolute_signed_energy_norm=Float64(shell_abs_E),
            cumulative_absolute_signed_energy_norm=Float64(cumulative_abs_E),
            shell_fraction=Float64(fraction),
            below_tolerance=fraction < cfg.shell_tolerance,
            consecutive_below=small_shells,
            energy_weighted_cancellation_condition=Float64(shell_weighted_condition),
            max_cancellation_condition=Float64(shell_max_condition),
            modes=modes,
        ))
        small_shells >= cfg.consecutive_shells && break
    end

    multipole_converged = small_shells >= cfg.consecutive_shells
    weighted_condition = condition_weighted_numerator /
                         max(cumulative_abs_E, eps(Float64))
    return (
        omega=omega,
        signed_positive_frequency_dE_domega=Float64(total_E),
        signed_positive_frequency_dJ_domega=Float64(total_J),
        signed_positive_frequency_dA_domega=Float64(total_A),
        reality_completed_one_sided_dE_domega=Float64(2 * total_E),
        multipole_converged=multipole_converged,
        ell_max_computed=shells[end].ell,
        energy_weighted_cancellation_condition=Float64(weighted_condition),
        max_cancellation_condition=Float64(max_condition),
        within_conditioning_limit=weighted_condition <= cfg.cancellation_condition_limit,
        shells=shells,
    )
end

function solve_horizon_spectrum(cfg::HorizonSpectrumConfig)
    _validate_horizon_spectrum_config(cfg)
    frequencies = horizon_frequency_grid(cfg)
    explicit_grid = !isempty(cfg.frequencies)
    rows = NamedTuple[]
    peak = 0.0
    below_endpoint_count = 0
    endpoint_closed_at = nothing
    first_conditioning_failure = nothing

    for omega in frequencies
        row = _adaptive_horizon_frequency_row(cfg, Float64(omega))
        push!(rows, row)
        value = abs(row.reality_completed_one_sided_dE_domega)
        peak = max(peak, value)

        if row.within_conditioning_limit && first_conditioning_failure === nothing
            endpoint_small = value <= cfg.high_endpoint_fraction_tolerance *
                                      max(peak, eps(Float64))
            decreasing = length(rows) >= 2 && value <
                         abs(rows[end - 1].reality_completed_one_sided_dE_domega)
            below_endpoint_count = endpoint_small && decreasing ?
                                   below_endpoint_count + 1 : 0
            if below_endpoint_count >= cfg.high_endpoint_consecutive_points &&
               endpoint_closed_at === nothing
                endpoint_closed_at = length(rows)
                if !explicit_grid && cfg.stop_when_converged
                    break
                end
            end
        elseif !row.within_conditioning_limit && first_conditioning_failure === nothing
            first_conditioning_failure = length(rows)
            if !explicit_grid && endpoint_closed_at === nothing
                break
            end
        end
    end

    raw = (
        orbit=(
            a=cfg.base.a, energy=cfg.base.energy, lz=cfg.base.lz,
            carter_q=cfg.base.carter_q, theta_infinity=cfg.base.theta_infinity,
            phi_infinity=cfg.base.phi_infinity, theta_sign=cfg.base.theta_sign,
            particle_mass=cfg.base.particle_mass,
        ),
        ell_min=cfg.ell_min,
        ell_max=maximum(row.ell_max_computed for row in rows),
        rows=rows,
    )
    diagnostics = horizon_spectrum_diagnostics(
        raw;
        shell_tolerance=cfg.shell_tolerance,
        consecutive_shells=cfg.consecutive_shells,
        high_endpoint_fraction_tolerance=cfg.high_endpoint_fraction_tolerance,
    )

    endpoint_sequence_pass = endpoint_closed_at !== nothing
    acceptance_end = endpoint_sequence_pass ? endpoint_closed_at : length(rows)
    trusted_prefix = rows[1:acceptance_end]
    multipole_pass = all(row.multipole_converged for row in trusted_prefix)
    conditioning_pass_to_endpoint = all(row.within_conditioning_limit for row in trusted_prefix)
    conditioning_failed_before_endpoint = first_conditioning_failure !== nothing &&
        (!endpoint_sequence_pass || first_conditioning_failure <= endpoint_closed_at)

    status = if endpoint_sequence_pass && multipole_pass && conditioning_pass_to_endpoint
        :accepted_direct_band
    elseif conditioning_failed_before_endpoint
        :needs_high_frequency_fallback
    elseif !multipole_pass
        :needs_more_multipoles
    else
        :needs_larger_frequency_window
    end

    return (
        config=cfg,
        status=status,
        candidate_frequencies=frequencies,
        computed_frequencies=Float64[row.omega for row in rows],
        accepted_direct_omega_max=endpoint_sequence_pass ?
            Float64(rows[endpoint_closed_at].omega) : NaN,
        first_conditioning_failure_omega=first_conditioning_failure === nothing ?
            NaN : Float64(rows[first_conditioning_failure].omega),
        spectrum=raw,
        diagnostics=diagnostics,
        endpoint_sequence_pass=endpoint_sequence_pass,
        conditioning_failed_before_endpoint=conditioning_failed_before_endpoint,
        high_frequency_fallback_required=status == :needs_high_frequency_fallback,
    )
end



const ANGULAR_TS_PHASE_CACHE = Dict{Tuple{Int,Int,UInt64}, NamedTuple}()

function solve_horizon(
    base::HorizonScatteringConfig;
    spectrum_config=nothing,
)
    spectrum_cfg = spectrum_config === nothing ?
        HorizonSpectrumConfig(base=deepcopy(base)) : deepcopy(spectrum_config)
    spectrum_cfg.base = deepcopy(base)
    spectrum = solve_horizon_spectrum(spectrum_cfg)
    accepted = spectrum.status == :accepted_direct_band
    return (
        status=accepted ? :accepted_horizon_spectrum : spectrum.status,
        production_accepted=accepted,
        base=deepcopy(base),
        spectrum=spectrum,
    )
end


@inline function _log1pexp(x::Float64)
    return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
end

function photon_edge_adjacent_energy_ratio(
    ell::Integer,
    omega::Real;
    eta_ph::Real,
    g::Real,
    gamma_ph::Real,
)
    ell >= 2 || error("require ell >= 2")
    omega_value = Float64(omega)
    eta_value = Float64(eta_ph)
    g_value = Float64(g)
    gamma_value = Float64(gamma_ph)
    omega_value > 0 || error("require omega > 0")
    gamma_value > 0 || error("require gamma_ph > 0")

    J = Float64(ell) + 0.5
    x = omega_value * eta_value
    y0 = 2 * gamma_value * (J - x)
    y1 = 2 * gamma_value * (J + 1 - x)
    log_ratio = 2 * g_value - (_log1pexp(y1) - _log1pexp(y0))
    return (
        ell=Int(ell),
        omega=omega_value,
        J=J,
        edge_offset=J - x,
        lattice_spacing_eta=Float64(1 / omega_value),
        log_ratio=Float64(log_ratio),
        ratio=Float64(exp(log_ratio)),
    )
end

function horizon_axisymmetric_photon_edge_scan(
    cfg::HorizonScatteringConfig;
    frequencies,
    ell_min::Int=2,
    ell_max::Int=4,
    gauss_panels=nothing,
    eta_ph=nothing,
    g=nothing,
    gamma_ph=nothing,
)
    2 <= ell_min <= ell_max || error("require 2 <= ell_min <= ell_max")
    supplied = (eta_ph !== nothing, g !== nothing, gamma_ph !== nothing)
    all(supplied) || !any(supplied) ||
        error("supply eta_ph, g and gamma_ph together, or none of them")
    use_theory = all(supplied)

    rows = NamedTuple[]
    for omega_value in frequencies
        omega = Float64(omega_value)
        omega > 0 || error("axisymmetric photon-edge scan expects omega > 0")
        cache_cfg = deepcopy(cfg)
        cache_cfg.omega = omega
        orbit_cache = build_direct_orbit_cache(cache_cfg)
        modes = NamedTuple[]
        for ell in ell_min:ell_max
            local_cfg = deepcopy(cfg)
            local_cfg.ell = ell
            local_cfg.m = 0
            local_cfg.omega = omega
            radial = compute_horizon_teukolsky_amplitude(
                local_cfg; orbit_cache=orbit_cache, gauss_panels=gauss_panels,
            )
            physical = horizon_physical_from_teukolsky(radial)
            condition = direct_cancellation_condition(radial)
            push!(modes, (
                ell=ell,
                m=0,
                omega=omega,
                J=ell + 0.5,
                eta=(ell + 0.5) / omega,
                z_h_minus2=radial.z_h_minus2,
                Psi_h=physical.Psi_h,
                shear_h=physical.shear_h,
                dE_domega=physical.dE_domega,
                dA_domega=physical.dA_domega,
                cancellation_condition=condition,
            ))
        end

        adjacent = NamedTuple[]
        for i in 1:(length(modes) - 1)
            low = modes[i]
            high = modes[i + 1]
            numerical_ratio = high.dE_domega / low.dE_domega
            theory = use_theory ? photon_edge_adjacent_energy_ratio(
                low.ell, omega;
                eta_ph=eta_ph, g=g, gamma_ph=gamma_ph,
            ) : nothing
            push!(adjacent, (
                ell_low=low.ell,
                ell_high=high.ell,
                numerical_energy_ratio=Float64(numerical_ratio),
                numerical_log_ratio=Float64(log(numerical_ratio)),
                theory=theory,
            ))
        end

        push!(rows, (
            omega=omega,
            lattice_spacing_eta=Float64(1 / omega),
            modes=modes,
            adjacent=adjacent,
        ))
    end

    return (
        orbit=(
            a=cfg.a, energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q,
            theta_infinity=cfg.theta_infinity, phi_infinity=cfg.phi_infinity,
            theta_sign=cfg.theta_sign, particle_mass=cfg.particle_mass,
        ),
        ell_min=ell_min,
        ell_max=ell_max,
        theory=use_theory ? (
            eta_ph=Float64(eta_ph),
            g=Float64(g),
            gamma_ph=Float64(gamma_ph),
        ) : nothing,
        rows=rows,
    )
end



function horizon_direct_outer_convergence(cfg::HorizonScatteringConfig;
                                          match_phases=(600.0, 1800.0))
    length(match_phases) >= 2 || error("need at least two outer matching phases")
    results = HorizonTeukolskyResult[]
    for phase in match_phases
        local_cfg = deepcopy(cfg)
        local_cfg.asymptotic_match_phase = Float64(phase)
        push!(results, compute_horizon_teukolsky_amplitude(local_cfg))
    end
    errors = Float64[]
    for i in 1:(length(results) - 1)
        a = results[i].z_h_minus2
        b = results[i + 1].z_h_minus2
        push!(errors, Float64(abs(a - b) / max(abs(b), eps(Float64))))
    end
    return (results=results, relative_successive_errors=errors)
end

end # module
