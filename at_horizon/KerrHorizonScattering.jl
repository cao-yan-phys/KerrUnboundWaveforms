module KerrHorizonScattering

using KerrUnboundWaveforms
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using OrdinaryDiffEqVerner: AutoVern9
using SciMLBase: ODEProblem, solve, successful_retcode
using StaticArrays: SVector
using LinearAlgebra
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
       HorizonTimeDomainConfig,
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
       horizon_frequency_grid,
       solve_horizon_spectrum,
       build_direct_orbit_cache,
       angular_ts_phase_diagnostic,
       horizon_spherical_field_block,
       horizon_spherical_field_mode,
       horizon_spherical_field_convergence,
       quadrupole_horizon_endpoint_nonanalytic,
       horizon_time_frequency_grid,
       horizon_soft_completed_frequency_bins,
       reconstruct_horizon_time_mode,
       solve_horizon_time_mode,
       solve_horizon_time_domain,
       horizon_time_multipole_diagnostics,
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

Base.@kwdef mutable struct HorizonTimeDomainConfig
    base::HorizonScatteringConfig = HorizonScatteringConfig()
    frequencies::Vector{Float64} = Float64[]
    omega_min::Float64 = 0.02
    soft_frequency_min_fraction::Float64 = 0.25
    soft_frequency_max_fraction::Float64 = 0.50
    soft_frequency_count::Int = 3
    frequency_grid::String = "fft_sparse"
    fft_points::Int = 1024
    fft_dense_low_bin_count::Int = 8
    fft_bin_stride::Int = 4
    spherical_l_max::Int = 6
    spheroidal_buffer::Int = 2
    spheroidal_l_max::Int = 10
    mixing_tolerance::Float64 = 2.0e-4
    angular_ts_tolerance::Float64 = 2.0e-8
    cancellation_condition_limit::Float64 = 1.0e8
    soft_fit_points::Int = 3
    soft_polynomial_degree::Int = 1
    use_quadrupole_endpoint::Bool = true
    high_taper_start_fraction::Float64 = 0.9
    waveform_completion_tolerance::Float64 = 2.0e-2
    time_shell_tolerance::Float64 = 5.0e-3
    time_consecutive_shells::Int = 2
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

function _angular_ldag_A_derivatives(theta::Float64,
                                     m::Int,
                                     c::Float64,
                                     n::Int)
    st = sin(theta)
    ct = cos(theta)
    abs(st) > 1e-8 || error("angular TS audit point is too close to a pole")
    csc = inv(st)
    cot = ct * csc
    csc2 = csc^2

    A0 = -m * csc + c * st + n * cot
    A1 = m * csc * cot + c * ct - n * csc2
    A2 = m * (-csc * cot^2 - csc^3) - c * st +
         2n * csc2 * cot
    A3 = m * (csc * cot^3 + 5csc^3 * cot) - c * ct -
         4n * csc2 * cot^2 - 2n * csc^4
    return (Float64(A0), Float64(A1), Float64(A2), Float64(A3))
end

function _apply_angular_ldag_jet(jet::Vector{ComplexF64},
                                 theta::Float64,
                                 m::Int,
                                 c::Float64,
                                 n::Int)
    length(jet) >= 2 || error("angular operator jet needs at least two derivatives")
    max_order = length(jet) - 2
    Ader = _angular_ldag_A_derivatives(theta, m, c, n)
    out = Vector{ComplexF64}(undef, length(jet) - 1)
    for r in 0:max_order
        value = jet[r + 2]
        for k in 0:r
            value += binomial(r, k) * Ader[r - k + 1] * jet[k + 1]
        end
        out[r + 1] = ComplexF64(value)
    end
    return out
end

function _angular_ts_minus_to_plus_value(harmonic_minus,
                                         theta::Float64,
                                         m::Int,
                                         c::Float64)
    jet = ComplexF64[
        harmonic_minus(theta, 0.0; theta_derivative=j) for j in 0:4
    ]
    for n in (2, 1, 0, -1)
        jet = _apply_angular_ldag_jet(jet, theta, m, c, n)
    end
    length(jet) == 1 || error("angular TS operator did not reduce to a scalar")
    return jet[1]
end

function angular_ts_phase_diagnostic(ell::Integer,
                                     m::Integer,
                                     c::Real;
                                     tolerance::Real=2.0e-8)
    ell_value = Int(ell)
    m_value = Int(m)
    ell_value >= 2 || error("require ell >= 2")
    abs(m_value) <= ell_value || error("require |m| <= ell")
    c_value = Float64(c)
    tol = Float64(tolerance)
    tol > 0 || error("angular TS tolerance must be positive")
    key = (ell_value, m_value, reinterpret(UInt64, c_value))
    cached = get(ANGULAR_TS_PHASE_CACHE, key, nothing)
    if cached !== nothing
        cached.relative_residual <= tol || error(
            "cached angular TS residual $(cached.relative_residual) exceeds requested tolerance $tol",
        )
        return cached
    end

    minus = spin_weighted_spheroidal_harmonic(
        -2, ell_value, m_value, c_value,
    )
    plus = spin_weighted_spheroidal_harmonic(
        +2, ell_value, m_value, c_value,
    )
    thetas = (0.47, 0.71, 0.96, 1.23, 1.57, 1.88, 2.17, 2.43, 2.67)
    transformed = ComplexF64[
        _angular_ts_minus_to_plus_value(minus, theta, m_value, c_value)
        for theta in thetas
    ]
    plus_values = ComplexF64[plus(theta, 0.0) for theta in thetas]
    denominator = sum(abs2, plus_values)
    denominator > 0 || error("normalized s=+2 spheroidal harmonic vanished on audit grid")
    Cang = sum(conj(plus_values[i]) * transformed[i] for i in eachindex(thetas)) /
           denominator
    abs(Cang) > 0 || error("angular Teukolsky--Starobinsky constant vanished")
    residual = sqrt(
        sum(abs2(transformed[i] - Cang * plus_values[i]) for i in eachindex(thetas)) /
        max(sum(abs2, transformed), eps(Float64))
    )
    residual <= tol || error(
        "angular Teukolsky--Starobinsky identity failed: relative residual=$residual",
    )
    if abs(c_value) <= 1.0e-14
        expected = Float64(factorial(big(ell_value + 2)) / factorial(big(ell_value - 2)))
        magnitude_error = abs(abs(Cang) - expected) / expected
        magnitude_error <= 50tol || error(
            "spherical angular TS normalization failed: |C_ang|=$(abs(Cang)), expected=$expected",
        )
    end
    result = (
        ell=ell_value,
        m=m_value,
        c=c_value,
        constant=ComplexF64(Cang),
        magnitude=Float64(abs(Cang)),
        phase=ComplexF64(Cang / abs(Cang)),
        relative_residual=Float64(residual),
    )
    ANGULAR_TS_PHASE_CACHE[key] = result
    return result
end

function _plus2_spheroidal_to_spherical_coefficient(spheroidal_l::Int,
                                                     spherical_l::Int,
                                                     m::Int,
                                                     c::Float64)
    rows = SSM.spheroidal_expansion_coefficients(
        +2, spheroidal_l, m, c; coefficient_cutoff=0.0,
    )
    coefficient = 0.0 + 0.0im
    for row in rows
        row.spherical_l == spherical_l || continue
        coefficient += row.coefficient
    end
    return ComplexF64(coefficient)
end


function horizon_spherical_field_block(
    cfg::HorizonScatteringConfig,
    m::Integer,
    omega::Real;
    spherical_l_min::Int=max(2, abs(Int(m))),
    spherical_l_max::Int=6,
    spheroidal_buffer::Int=2,
    spheroidal_l_max::Int=max(10, spherical_l_max + 4),
    mixing_tolerance::Real=2.0e-4,
    angular_ts_tolerance::Real=2.0e-8,
    orbit_cache=nothing,
    gauss_panels=nothing,
)
    m_value = Int(m)
    omega_value = Float64(omega)
    omega_value != 0 || error("spherical horizon field block requires nonzero omega")
    ell_min = max(2, abs(m_value))
    ell_min <= spherical_l_min <= spherical_l_max ||
        error("invalid spherical-l range")
    spheroidal_buffer >= 1 || error("spheroidal_buffer must be positive")
    start_max = max(spherical_l_max + spheroidal_buffer, ell_min)
    spheroidal_l_max >= start_max || error(
        "spheroidal_l_max=$spheroidal_l_max is below required minimum $start_max",
    )
    mix_tol = Float64(mixing_tolerance)
    mix_tol > 0 || error("mixing_tolerance must be positive")

    local_cache = orbit_cache
    if local_cache === nothing
        cache_cfg = deepcopy(cfg)
        cache_cfg.omega = omega_value
        local_cache = build_direct_orbit_cache(cache_cfg)
    end

    spherical_ls = collect(spherical_l_min:spherical_l_max)
    Psi = Dict(ell => 0.0 + 0.0im for ell in spherical_ls)
    shear = Dict(ell => 0.0 + 0.0im for ell in spherical_ls)
    last_relative = Dict(ell => Inf for ell in spherical_ls)
    spheroidal_rows = NamedTuple[]
    converged = false
    used_max = ell_min

    for ell_sph in ell_min:spheroidal_l_max
        local_cfg = deepcopy(cfg)
        local_cfg.ell = ell_sph
        local_cfg.m = m_value
        local_cfg.omega = omega_value
        radial = compute_horizon_teukolsky_amplitude(
            local_cfg; orbit_cache=local_cache, gauss_panels=gauss_panels,
        )
        physical = horizon_physical_from_teukolsky(radial)
        c = local_cfg.a * omega_value
        angular = angular_ts_phase_diagnostic(
            ell_sph, m_value, c; tolerance=angular_ts_tolerance,
        )
        expansion = SSM.spheroidal_expansion_coefficients(
            +2, ell_sph, m_value, c; coefficient_cutoff=0.0,
        )
        package_coefficients = Dict{Int,ComplexF64}()
        for row in expansion
            row.spherical_l in spherical_ls || continue
            package_coefficients[row.spherical_l] =
                get(package_coefficients, row.spherical_l, 0.0 + 0.0im) +
                ComplexF64(row.coefficient)
        end

        contributions = NamedTuple[]
        max_relative = 0.0
        for ell_spherical in spherical_ls
            package_mix = get(package_coefficients, ell_spherical, 0.0 + 0.0im)
            mixing = angular.phase * package_mix
            dPsi = mixing * physical.Psi_h
            dshear = mixing * physical.shear_h
            Psi[ell_spherical] += dPsi
            shear[ell_spherical] += dshear
            relative = max(
                abs(dPsi) / max(abs(Psi[ell_spherical]), eps(Float64)),
                abs(dshear) / max(abs(shear[ell_spherical]), eps(Float64)),
            )
            last_relative[ell_spherical] = Float64(relative)
            max_relative = max(max_relative, relative)
            push!(contributions, (
                spherical_l=ell_spherical,
                package_mixing_coefficient=ComplexF64(package_mix),
                physical_mixing_coefficient=ComplexF64(mixing),
                Psi_contribution=ComplexF64(dPsi),
                shear_contribution=ComplexF64(dshear),
                relative_contribution=Float64(relative),
            ))
        end
        used_max = ell_sph
        push!(spheroidal_rows, (
            spheroidal_l=ell_sph,
            m=m_value,
            omega=omega_value,
            angular_ts_constant=angular.constant,
            angular_ts_phase=angular.phase,
            angular_ts_relative_residual=angular.relative_residual,
            physical=physical,
            cancellation_condition=direct_cancellation_condition(radial),
            spherical_contributions=contributions,
            max_relative_contribution=Float64(max_relative),
        ))
        ell_sph < start_max && continue
        if max_relative <= mix_tol
            converged = true
            break
        end
    end

    modes = Dict(
        ell => (
            spherical_l=ell,
            m=m_value,
            omega=omega_value,
            Psi_h=ComplexF64(Psi[ell]),
            shear_h=ComplexF64(shear[ell]),
            last_relative_contribution=Float64(last_relative[ell]),
        )
        for ell in spherical_ls
    )
    return (
        m=m_value,
        omega=omega_value,
        spherical_l_min=spherical_l_min,
        spherical_l_max=spherical_l_max,
        spheroidal_l_max=used_max,
        mixing_converged=converged,
        mixing_tolerance=mix_tol,
        modes=modes,
        spheroidal_rows=spheroidal_rows,
    )
end

function horizon_spherical_field_mode(
    cfg::HorizonScatteringConfig,
    spherical_l::Integer,
    m::Integer,
    omega::Real;
    spheroidal_buffer::Int=2,
    spheroidal_l_max::Int=max(8, Int(spherical_l) + 4),
    mixing_tolerance::Real=2.0e-4,
    angular_ts_tolerance::Real=2.0e-8,
    orbit_cache=nothing,
    gauss_panels=nothing,
)
    spherical_l_value = Int(spherical_l)
    block = horizon_spherical_field_block(
        cfg, m, omega;
        spherical_l_min=spherical_l_value,
        spherical_l_max=spherical_l_value,
        spheroidal_buffer=spheroidal_buffer,
        spheroidal_l_max=spheroidal_l_max,
        mixing_tolerance=mixing_tolerance,
        angular_ts_tolerance=angular_ts_tolerance,
        orbit_cache=orbit_cache,
        gauss_panels=gauss_panels,
    )
    mode = block.modes[spherical_l_value]
    return merge(mode, (
        spheroidal_l_max=block.spheroidal_l_max,
        mixing_converged=block.mixing_converged,
        mixing_tolerance=block.mixing_tolerance,
        spheroidal_rows=block.spheroidal_rows,
    ))
end


@inline function _complex_symmetric_relative(a::Complex, b::Complex)
    return Float64(2 * abs(a - b) / max(abs(a) + abs(b), eps(Float64)))
end

function horizon_spherical_field_convergence(
    cfg::HorizonScatteringConfig,
    spherical_l::Integer,
    m::Integer,
    omega::Real;
    spheroidal_buffer::Int=2,
    spheroidal_l_max::Int=max(8, Int(spherical_l) + 4),
    mixing_tolerance::Real=2.0e-4,
    angular_ts_tolerance::Real=2.0e-8,
    panel_factor::Real=2.0,
    outer_phase_factor::Real=3.0,
)
    panel_factor > 1 || error("panel_factor must exceed one")
    outer_phase_factor > 1 || error("outer_phase_factor must exceed one")
    signed_omega = Float64(omega)
    signed_omega != 0 || error("field convergence requires nonzero omega")

    baseline_cfg = deepcopy(cfg)
    baseline_cfg.omega = signed_omega
    baseline_cache = build_direct_orbit_cache(baseline_cfg)
    baseline = horizon_spherical_field_mode(
        baseline_cfg, spherical_l, m, signed_omega;
        spheroidal_buffer=spheroidal_buffer,
        spheroidal_l_max=spheroidal_l_max,
        mixing_tolerance=mixing_tolerance,
        angular_ts_tolerance=angular_ts_tolerance,
        orbit_cache=baseline_cache,
    )
    baseline.mixing_converged || error("baseline spherical mixing did not converge")

    base_panels = _direct_gauss_panels(baseline_cfg)
    refined_panels = max(base_panels + 1, ceil(Int, panel_factor * base_panels))
    quadrature = horizon_spherical_field_mode(
        baseline_cfg, spherical_l, m, signed_omega;
        spheroidal_buffer=spheroidal_buffer,
        spheroidal_l_max=spheroidal_l_max,
        mixing_tolerance=mixing_tolerance,
        angular_ts_tolerance=angular_ts_tolerance,
        orbit_cache=baseline_cache,
        gauss_panels=refined_panels,
    )
    quadrature.mixing_converged || error("quadrature-refined spherical mixing did not converge")

    outer_cfg = deepcopy(cfg)
    outer_cfg.omega = signed_omega
    outer_cfg.asymptotic_match_phase *= Float64(outer_phase_factor)
    outer_cache = build_direct_orbit_cache(outer_cfg)
    outer = horizon_spherical_field_mode(
        outer_cfg, spherical_l, m, signed_omega;
        spheroidal_buffer=spheroidal_buffer,
        spheroidal_l_max=spheroidal_l_max,
        mixing_tolerance=mixing_tolerance,
        angular_ts_tolerance=angular_ts_tolerance,
        orbit_cache=outer_cache,
    )
    outer.mixing_converged || error("outer-refined spherical mixing did not converge")

    condition = maximum(row.cancellation_condition for row in baseline.spheroidal_rows)
    return (
        baseline=baseline,
        quadrature=quadrature,
        outer=outer,
        baseline_max_cancellation=Float64(condition),
        baseline_panels=base_panels,
        refined_panels=refined_panels,
        baseline_match_phase=Float64(baseline_cfg.asymptotic_match_phase),
        refined_match_phase=Float64(outer_cfg.asymptotic_match_phase),
        quadrature_Psi_relative=_complex_symmetric_relative(baseline.Psi_h, quadrature.Psi_h),
        quadrature_shear_relative=_complex_symmetric_relative(baseline.shear_h, quadrature.shear_h),
        outer_Psi_relative=_complex_symmetric_relative(baseline.Psi_h, outer.Psi_h),
        outer_shear_relative=_complex_symmetric_relative(baseline.shear_h, outer.shear_h),
    )
end



@inline _unit_radial(theta::Real, phi::Real) = (
    sin(theta) * cos(phi),
    sin(theta) * sin(phi),
    cos(theta),
)

function _rotate_xy(vector, angle::Real)
    c = cos(angle)
    s = sin(angle)
    return (
        c * vector[1] - s * vector[2],
        s * vector[1] + c * vector[2],
        vector[3],
    )
end

function _direct_asymptotic_velocity_data(cfg::HorizonScatteringConfig,
                                          cache::DirectHorizonOrbitCache)
    context = _direct_problem_context(cfg)
    _validate_direct_orbit_cache(cache, context, cfg)
    beta = sqrt(cfg.energy^2 - 1) / cfg.energy
    incoming_direction = _unit_radial(cfg.theta_infinity, cfg.phi_infinity)
    incoming_raw = Tuple(-beta * value for value in incoming_direction)

    y = cache.geodesic.outgoing(context.z_outer)
    if context.equatorial
        theta_out = pi / 2
        theta_sign_out = 1.0
    else
        theta_out = UG.theta_from_polar_phase(y[2], context.polar)
        theta_sign_out = UG.polar_sign_from_phase(y[2])
    end
    outgoing_tail = UG.asymptote_from_finite_outer_state(
        context.kerr, context.constants, context.r_outer,
        theta_out, Float64(y[3]), 1.0, theta_sign_out,
    )
    outgoing_direction = _unit_radial(outgoing_tail.theta, outgoing_tail.phi)
    outgoing_raw = Tuple(beta * value for value in outgoing_direction)

    phi_origin = cache.geodesic.phi_origin
    return (
        v_in=_rotate_xy(incoming_raw, -phi_origin),
        v_out=_rotate_xy(outgoing_raw, -phi_origin),
        beta=Float64(beta),
        phi_origin=Float64(phi_origin),
        outgoing_theta=Float64(outgoing_tail.theta),
        outgoing_phi=Float64(outgoing_tail.phi),
    )
end

function _unit_velocity_direction(vector)
    norm = sqrt(sum(abs2, vector))
    norm > 0 || error("asymptotic velocity direction is undefined")
    return Tuple(Float64(component / norm) for component in vector)
end

function _quadrupole_endpoint_q(cfg::HorizonScatteringConfig,
                                m::Int,
                                vhat)
    abs(m) <= 2 || error("quadrupole endpoint requires |m| <= 2")
    p_inf = sqrt(cfg.energy^2 - 1)
    prefactor = cfg.particle_mass / p_inf^3
    vx, vy, vz = vhat
    if m == 0
        return ComplexF64(prefactor * (3vz^2 - 1))
    elseif m == 1
        return ComplexF64(-3prefactor * vz * (vx - im * vy))
    elseif m == -1
        return ComplexF64(-3prefactor * vz * (vx + im * vy))
    elseif m == 2
        return ComplexF64(-3prefactor * (vx - im * vy)^2)
    end
    return ComplexF64(-3prefactor * (vx + im * vy)^2)
end

function quadrupole_horizon_endpoint_nonanalytic(
    cfg::HorizonScatteringConfig,
    m::Integer,
    omega::Real,
    velocities,
)
    m_value = Int(m)
    abs(m_value) <= 2 || error("quadrupole endpoint requires |m| <= 2")
    omega_value = Float64(omega)
    if omega_value == 0
        return (Psi_h=0.0 + 0.0im, shear_h=0.0 + 0.0im,
                q_minus=0.0 + 0.0im, q_plus=0.0 + 0.0im)
    end

    vhat_minus = _unit_velocity_direction(velocities.v_in)
    vhat_plus = _unit_velocity_direction(velocities.v_out)
    qminus = _quadrupole_endpoint_q(cfg, m_value, vhat_minus)
    qplus = _quadrupole_endpoint_q(cfg, m_value, vhat_plus)
    logomega = log(abs(omega_value))
    s = sqrt(1 - cfg.a^2)

    Psi_tilde = if m_value == 0 || abs(cfg.a) <= 1.0e-14
        D0 = s^3 / (6 * (1 + s))
        ComplexF64(
            im * D0 / 2 * (qplus + qminus) * omega_value^3 * logomega -
            pi * D0 / 4 * (qminus - qplus) * omega_value^2 * abs(omega_value)
        )
    else
        gamma = cfg.a / (2s)
        Ah = -(im / 6) * m_value * gamma *
             (1 + im * m_value * gamma) *
             (1 + 4m_value^2 * gamma^2)
        transfer = s^4 / (1 + s)^2 * Ah
        ComplexF64(
            transfer * (
                (qplus + qminus) / 2 * omega_value^2 * logomega +
                im * pi / 4 * (qminus - qplus) *
                omega_value * abs(omega_value)
            )
        )
    end

    Psi = ComplexF64(Psi_tilde / (2pi))

    rp = KerrGeometry.rplus(cfg.a)
    omega_h = cfg.a / (2rp)
    kappa_h = s / (2rp)
    denominator = kappa_h + im * (omega_value - m_value * omega_h)
    abs(denominator) > 0 || error("vanishing Hawking--Hartle shear denominator")
    shear = ComplexF64(Psi / denominator)
    return (
        Psi_h=Psi,
        shear_h=shear,
        q_minus=qminus,
        q_plus=qplus,
    )
end

function _complex_polynomial_fit(xs, ys, degree::Int)
    length(xs) == length(ys) || error("polynomial fit arrays differ in length")
    degree >= 0 || error("polynomial degree must be nonnegative")
    length(xs) >= degree + 1 || error("not enough points for polynomial fit")
    A = Matrix{ComplexF64}(undef, length(xs), degree + 1)
    for i in eachindex(xs)
        power = 1.0
        for j in 0:degree
            A[i, j + 1] = power
            power *= xs[i]
        end
    end
    values = ComplexF64.(ys)
    coeff = A \ values
    fitted = A * coeff
    residual = sqrt(
        sum(abs2(values[i] - fitted[i]) for i in eachindex(values)) /
        max(sum(abs2, values), eps(Float64))
    )
    return ComplexF64.(coeff), Float64(residual)
end

@inline function _complex_polynomial_value(coefficients, x::Real)
    value = 0.0 + 0.0im
    for coefficient in Iterators.reverse(coefficients)
        value = value * x + coefficient
    end
    return ComplexF64(value)
end


function horizon_time_frequency_grid(cfg::HorizonTimeDomainConfig,
                                     omega_max::Real)
    omega_max_value = Float64(omega_max)
    omega_max_value > 0 || error("omega_max must be positive")
    ispow2(cfg.fft_points) || error("fft_points must be a power of two")
    cfg.fft_points >= 64 || error("fft_points must be at least 64")
    cfg.omega_min > 0 || error("time-domain omega_min must be positive")
    cfg.omega_min < omega_max_value || error("time-domain omega_min must be below omega_max")
    if !isempty(cfg.frequencies)
        values = sort!(unique(Float64.(cfg.frequencies)))
        all(>(0), values) || error("explicit time-domain frequencies must be positive")
        values[end] <= omega_max_value * (1 + 1e-12) || error(
            "explicit time-domain frequency exceeds the accepted spectrum endpoint",
        )
        return values
    end
    cfg.frequency_grid in ("fft_sparse", "fft_aligned") ||
        error("frequency_grid must be fft_sparse or fft_aligned when frequencies are not explicit")
    0 < cfg.soft_frequency_min_fraction < cfg.soft_frequency_max_fraction <= 1 ||
        error("require 0 < soft_frequency_min_fraction < soft_frequency_max_fraction <= 1")
    cfg.soft_frequency_count >= 3 || error("soft_frequency_count must be at least three")

    soft = collect(range(
        cfg.soft_frequency_min_fraction * cfg.omega_min,
        cfg.soft_frequency_max_fraction * cfg.omega_min;
        length=cfg.soft_frequency_count,
    ))

    domega = 2omega_max_value / cfg.fft_points
    maximum_bin = div(cfg.fft_points, 2)
    first_bin = max(1, ceil(Int, cfg.omega_min / domega))
    first_bin <= maximum_bin || error("omega_min lies above the FFT endpoint")
    fft_nodes = if cfg.frequency_grid == "fft_aligned"
        Float64[domega * k for k in first_bin:maximum_bin]
    else
        cfg.fft_dense_low_bin_count >= 1 ||
            error("fft_dense_low_bin_count must be positive")
        cfg.fft_bin_stride >= 1 || error("fft_bin_stride must be positive")
        dense_last = min(maximum_bin, first_bin + cfg.fft_dense_low_bin_count - 1)
        bins = collect(first_bin:dense_last)
        if dense_last < maximum_bin
            append!(bins, collect((dense_last + 1):cfg.fft_bin_stride:maximum_bin))
            last(bins) == maximum_bin || push!(bins, maximum_bin)
        end
        Float64[domega * k for k in unique(bins)]
    end
    return sort!(unique(vcat(Float64.(soft), fft_nodes)))
end

function horizon_soft_completed_frequency_bins(
    frequencies,
    positive_values,
    negative_values,
    n::Int;
    endpoint_term=(omega -> 0.0 + 0.0im),
    soft_fit_points::Int=3,
    soft_polynomial_degree::Int=1,
    high_taper_start_fraction::Real=0.9,
    interpolation_coordinate::String="logarithmic",
    zero_frequency_value=nothing,
)
    ispow2(n) || error("fft_points must be a power of two")
    n >= 64 || error("fft_points must be at least 64")
    length(frequencies) == length(positive_values) == length(negative_values) ||
        error("signed horizon field sample arrays have different lengths")
    length(frequencies) >= 2 || error("need at least two positive-frequency samples")
    freq = Float64.(frequencies)
    issorted(freq) || error("frequencies must be sorted")
    all(>(0), freq) || error("frequencies must be positive")
    length(unique(freq)) == length(freq) || error("frequencies must be distinct")
    0 < high_taper_start_fraction < 1 ||
        error("high_taper_start_fraction must lie between zero and one")
    p = min(soft_fit_points, length(freq))
    p >= soft_polynomial_degree + 1 ||
        error("soft_fit_points is too small for the requested polynomial degree")

    positive = ComplexF64.(positive_values)
    negative = ComplexF64.(negative_values)
    pos_endpoint = ComplexF64[endpoint_term(w) for w in freq]
    neg_endpoint = ComplexF64[endpoint_term(-w) for w in freq]
    pos_regular = positive .- pos_endpoint
    neg_regular = negative .- neg_endpoint

    pos_coeff, pos_residual = _complex_polynomial_fit(
        freq[1:p], pos_regular[1:p], soft_polynomial_degree,
    )
    neg_coeff, neg_residual = _complex_polynomial_fit(
        -freq[1:p], neg_regular[1:p], soft_polynomial_degree,
    )
    pos_coeff[1] += pos_regular[1] -
                    _complex_polynomial_value(pos_coeff, freq[1])
    neg_coeff[1] += neg_regular[1] -
                    _complex_polynomial_value(neg_coeff, -freq[1])

    pos_zero = _complex_polynomial_value(pos_coeff, 0.0)
    neg_zero = _complex_polynomial_value(neg_coeff, 0.0)
    common_zero = zero_frequency_value === nothing ?
        ComplexF64((pos_zero + neg_zero) / 2) : ComplexF64(zero_frequency_value)
    zero_mismatch = abs(pos_zero - neg_zero) /
                    max(abs(common_zero), abs(pos_zero), abs(neg_zero), eps(Float64))
    omega_min = freq[1]
    omega_max = freq[end]
    domega = 2omega_max / n
    taper_start = Float64(high_taper_start_fraction) * omega_max
    high_taper(omega) = omega <= taper_start ? 1.0 :
        0.5 * (1 + cos(pi * (omega - taper_start) / (omega_max - taper_start)))

    function low_regular(coeff, branch_zero, signed_omega)
        raw = _complex_polynomial_value(coeff, signed_omega)
        u = abs(signed_omega) / omega_min
        correction = (common_zero - branch_zero) * max(0.0, 1 - u)
        return ComplexF64(raw + correction)
    end

    function positive_value(omega)
        if omega < omega_min
            return low_regular(pos_coeff, pos_zero, omega) + endpoint_term(omega)
        end
        return Internal.interpolate_complex(
            freq, positive, (omega,); coordinate=interpolation_coordinate,
        )[1]
    end
    function negative_value(omega_abs)
        if omega_abs < omega_min
            signed = -omega_abs
            return low_regular(neg_coeff, neg_zero, signed) + endpoint_term(signed)
        end
        return Internal.interpolate_complex(
            freq, negative, (omega_abs,); coordinate=interpolation_coordinate,
        )[1]
    end

    bins = zeros(ComplexF64, n)
    bins[1] = common_zero
    for k in 1:(n >> 1)
        omega = k * domega
        omega <= omega_max || continue
        bins[k + 1] = positive_value(omega) * high_taper(omega)
    end
    for k in 1:((n >> 1) - 1)
        omega = k * domega
        omega <= omega_max || continue
        bins[n - k + 1] = negative_value(omega) * high_taper(omega)
    end
    return (
        bins=bins,
        domega=Float64(domega),
        omega_max=omega_max,
        positive_regular_coefficients=pos_coeff,
        negative_regular_coefficients=neg_coeff,
        positive_fit_residual=pos_residual,
        negative_fit_residual=neg_residual,
        zero_frequency_value=common_zero,
        zero_frequency_constrained=zero_frequency_value !== nothing,
        zero_branch_relative_mismatch=Float64(zero_mismatch),
        high_taper_start=Float64(taper_start),
    )
end

function _inverse_horizon_bins(bins::Vector{ComplexF64}, domega::Float64)
    n = length(bins)
    transformed = domega .* Internal.radix2_negative_dft(bins)
    shifted = vcat(
        transformed[(n >> 1) + 1:end],
        transformed[1:(n >> 1)],
    )
    dt = 2pi / (n * domega)
    times = Float64[(index - 1 - (n >> 1)) * dt for index in 1:n]
    return times, ComplexF64.(shifted)
end

function reconstruct_horizon_time_mode(
    frequencies,
    positive_Psi,
    negative_Psi,
    positive_shear,
    negative_shear;
    fft_points::Int=1024,
    endpoint_Psi=(omega -> 0.0 + 0.0im),
    endpoint_shear=(omega -> 0.0 + 0.0im),
    soft_fit_points::Int=3,
    soft_polynomial_degree::Int=1,
    high_taper_start_fraction::Real=0.9,
    zero_frequency_Psi=nothing,
    zero_frequency_shear=nothing,
)
    Psi_bins = horizon_soft_completed_frequency_bins(
        frequencies, positive_Psi, negative_Psi, fft_points;
        endpoint_term=endpoint_Psi,
        soft_fit_points=soft_fit_points,
        soft_polynomial_degree=soft_polynomial_degree,
        high_taper_start_fraction=high_taper_start_fraction,
        zero_frequency_value=zero_frequency_Psi,
    )
    shear_bins = horizon_soft_completed_frequency_bins(
        frequencies, positive_shear, negative_shear, fft_points;
        endpoint_term=endpoint_shear,
        soft_fit_points=soft_fit_points,
        soft_polynomial_degree=soft_polynomial_degree,
        high_taper_start_fraction=high_taper_start_fraction,
        zero_frequency_value=zero_frequency_shear,
    )
    times, Psi_time = _inverse_horizon_bins(Psi_bins.bins, Psi_bins.domega)
    shear_times, shear_time = _inverse_horizon_bins(
        shear_bins.bins, shear_bins.domega,
    )
    times == shear_times || error("curvature and shear FFT grids disagree")

    alternatives = NamedTuple[]
    for degree in (0, 1, 2)
        degree == soft_polynomial_degree && continue
        length(frequencies) >= degree + 1 || continue
        p = max(soft_fit_points, degree + 1)
        p <= length(frequencies) || continue
        altPsi = horizon_soft_completed_frequency_bins(
            frequencies, positive_Psi, negative_Psi, fft_points;
            endpoint_term=endpoint_Psi,
            soft_fit_points=p,
            soft_polynomial_degree=degree,
            high_taper_start_fraction=high_taper_start_fraction,
            zero_frequency_value=zero_frequency_Psi,
        )
        _, altPsiTime = _inverse_horizon_bins(altPsi.bins, altPsi.domega)
        altShear = horizon_soft_completed_frequency_bins(
            frequencies, positive_shear, negative_shear, fft_points;
            endpoint_term=endpoint_shear,
            soft_fit_points=p,
            soft_polynomial_degree=degree,
            high_taper_start_fraction=high_taper_start_fraction,
            zero_frequency_value=zero_frequency_shear,
        )
        _, altShearTime = _inverse_horizon_bins(altShear.bins, altShear.domega)
        push!(alternatives, (
            kind=:soft_degree,
            value=degree,
            max_Psi_difference=Float64(maximum(abs.(altPsiTime .- Psi_time))),
            max_shear_difference=Float64(maximum(abs.(altShearTime .- shear_time))),
            Psi_l2_difference=Float64(norm(altPsiTime .- Psi_time)),
            shear_l2_difference=Float64(norm(altShearTime .- shear_time)),
        ))
    end
    for fraction in (0.85, 0.95)
        abs(fraction - high_taper_start_fraction) <= 1e-12 && continue
        altPsi = horizon_soft_completed_frequency_bins(
            frequencies, positive_Psi, negative_Psi, fft_points;
            endpoint_term=endpoint_Psi,
            soft_fit_points=soft_fit_points,
            soft_polynomial_degree=soft_polynomial_degree,
            high_taper_start_fraction=fraction,
            zero_frequency_value=zero_frequency_Psi,
        )
        _, altPsiTime = _inverse_horizon_bins(altPsi.bins, altPsi.domega)
        altShear = horizon_soft_completed_frequency_bins(
            frequencies, positive_shear, negative_shear, fft_points;
            endpoint_term=endpoint_shear,
            soft_fit_points=soft_fit_points,
            soft_polynomial_degree=soft_polynomial_degree,
            high_taper_start_fraction=fraction,
            zero_frequency_value=zero_frequency_shear,
        )
        _, altShearTime = _inverse_horizon_bins(altShear.bins, altShear.domega)
        push!(alternatives, (
            kind=:high_taper_fraction,
            value=fraction,
            max_Psi_difference=Float64(maximum(abs.(altPsiTime .- Psi_time))),
            max_shear_difference=Float64(maximum(abs.(altShearTime .- shear_time))),
            Psi_l2_difference=Float64(norm(altPsiTime .- Psi_time)),
            shear_l2_difference=Float64(norm(altShearTime .- shear_time)),
        ))
    end

    return (
        times=times,
        Psi_h=Psi_time,
        shear_h=shear_time,
        Psi_frequency_bins=Psi_bins,
        shear_frequency_bins=shear_bins,
        completion_alternatives=alternatives,
    )
end

function solve_horizon_time_mode(cfg::HorizonTimeDomainConfig,
                                 spherical_l::Integer,
                                 m::Integer)
    frequencies = sort!(unique(Float64.(cfg.frequencies)))
    length(frequencies) >= 2 ||
        error("HorizonTimeDomainConfig.frequencies needs at least two positive nodes")
    all(>(0), frequencies) || error("time-domain frequencies must be positive")
    ispow2(cfg.fft_points) || error("fft_points must be a power of two")
    cfg.fft_points >= 64 || error("fft_points must be at least 64")
    spherical_l_value = Int(spherical_l)
    m_value = Int(m)
    spherical_l_value >= max(2, abs(m_value)) || error("invalid spherical mode")
    spherical_l_value <= cfg.spherical_l_max ||
        error("requested spherical_l exceeds spherical_l_max")

    positive_Psi = ComplexF64[]
    negative_Psi = ComplexF64[]
    positive_shear = ComplexF64[]
    negative_shear = ComplexF64[]
    frequency_rows = NamedTuple[]
    velocities = nothing

    for omega in frequencies
        cache_cfg = deepcopy(cfg.base)
        cache_cfg.omega = omega
        orbit_cache = build_direct_orbit_cache(cache_cfg)
        velocities === nothing &&
            (velocities = _direct_asymptotic_velocity_data(cache_cfg, orbit_cache))

        positive = horizon_spherical_field_mode(
            cfg.base, spherical_l_value, m_value, omega;
            spheroidal_buffer=cfg.spheroidal_buffer,
            spheroidal_l_max=cfg.spheroidal_l_max,
            mixing_tolerance=cfg.mixing_tolerance,
            angular_ts_tolerance=cfg.angular_ts_tolerance,
            orbit_cache=orbit_cache,
            gauss_panels=cfg.gauss_panels,
        )
        negative = horizon_spherical_field_mode(
            cfg.base, spherical_l_value, m_value, -omega;
            spheroidal_buffer=cfg.spheroidal_buffer,
            spheroidal_l_max=cfg.spheroidal_l_max,
            mixing_tolerance=cfg.mixing_tolerance,
            angular_ts_tolerance=cfg.angular_ts_tolerance,
            orbit_cache=orbit_cache,
            gauss_panels=cfg.gauss_panels,
        )
        positive.mixing_converged || error(
            "positive-frequency spherical mixing failed at omega=$omega",
        )
        negative.mixing_converged || error(
            "negative-frequency spherical mixing failed at omega=$omega",
        )
        positive_condition = maximum(
            row.cancellation_condition for row in positive.spheroidal_rows
        )
        negative_condition = maximum(
            row.cancellation_condition for row in negative.spheroidal_rows
        )
        positive_condition_flagged = positive_condition > cfg.cancellation_condition_limit
        negative_condition_flagged = negative_condition > cfg.cancellation_condition_limit
        push!(positive_Psi, positive.Psi_h)
        push!(negative_Psi, negative.Psi_h)
        push!(positive_shear, positive.shear_h)
        push!(negative_shear, negative.shear_h)
        push!(frequency_rows, (
            omega=omega,
            positive=positive,
            negative=negative,
            positive_max_cancellation=Float64(positive_condition),
            negative_max_cancellation=Float64(negative_condition),
            positive_condition_flagged=positive_condition_flagged,
            negative_condition_flagged=negative_condition_flagged,
        ))
    end

    use_endpoint = cfg.use_quadrupole_endpoint && spherical_l_value == 2
    endpoint_Psi = use_endpoint ?
        (omega -> quadrupole_horizon_endpoint_nonanalytic(
            cfg.base, m_value, omega, velocities,
        ).Psi_h) :
        (omega -> 0.0 + 0.0im)
    endpoint_shear = use_endpoint ?
        (omega -> quadrupole_horizon_endpoint_nonanalytic(
            cfg.base, m_value, omega, velocities,
        ).shear_h) :
        (omega -> 0.0 + 0.0im)

    synchronous_zero = m_value == 0 || abs(cfg.base.a) <= 1.0e-14
    zero_Psi = synchronous_zero ? (0.0 + 0.0im) : nothing
    zero_shear = synchronous_zero ? (0.0 + 0.0im) : nothing

    time = reconstruct_horizon_time_mode(
        frequencies,
        positive_Psi, negative_Psi,
        positive_shear, negative_shear;
        fft_points=cfg.fft_points,
        endpoint_Psi=endpoint_Psi,
        endpoint_shear=endpoint_shear,
        soft_fit_points=cfg.soft_fit_points,
        soft_polynomial_degree=cfg.soft_polynomial_degree,
        high_taper_start_fraction=cfg.high_taper_start_fraction,
        zero_frequency_Psi=zero_Psi,
        zero_frequency_shear=zero_shear,
    )
    return (
        spherical_l=spherical_l_value,
        m=m_value,
        frequencies=frequencies,
        frequency_rows=frequency_rows,
        endpoint_completion=use_endpoint ? :quadrupole_exact_nonanalytic :
            :regular_with_higher_order_endpoint_omitted,
        endpoint_nonanalytic_order=synchronous_zero ? spherical_l_value + 1 : spherical_l_value,
        asymptotic_velocities=velocities,
        time_domain=time,
    )
end




function horizon_time_multipole_diagnostics(
    modes;
    shell_tolerance::Real=5.0e-3,
    consecutive_shells::Int=2,
    completion_tolerance::Real=2.0e-2,
)
    shell_tol = Float64(shell_tolerance)
    completion_tol = Float64(completion_tolerance)
    shell_tol > 0 || error("time shell tolerance must be positive")
    completion_tol > 0 || error("waveform completion tolerance must be positive")
    consecutive_shells >= 1 || error("time consecutive shells must be positive")
    isempty(modes) && error("time-domain mode dictionary is empty")

    ell_values = sort!(unique(Int[key[1] for key in keys(modes)]))
    shell_rows = NamedTuple[]
    cumulative_Psi2 = 0.0
    cumulative_shear2 = 0.0
    small_shells = 0
    for ell in ell_values
        shell_keys = sort!(Tuple{Int,Int}[key for key in keys(modes) if key[1] == ell])
        expected = 2ell + 1
        length(shell_keys) == expected || error(
            "incomplete spherical L=$ell shell: expected $expected m modes, found $(length(shell_keys))",
        )
        shell_Psi2 = sum(sum(abs2, modes[key].time_domain.Psi_h) for key in shell_keys)
        shell_shear2 = sum(sum(abs2, modes[key].time_domain.shear_h) for key in shell_keys)
        cumulative_Psi2 += shell_Psi2
        cumulative_shear2 += shell_shear2
        Psi_fraction = sqrt(shell_Psi2 / max(cumulative_Psi2, eps(Float64)))
        shear_fraction = sqrt(shell_shear2 / max(cumulative_shear2, eps(Float64)))
        shell_fraction = max(Psi_fraction, shear_fraction)
        below = shell_fraction <= shell_tol
        small_shells = below ? small_shells + 1 : 0
        push!(shell_rows, (
            ell=ell,
            Psi_l2=Float64(sqrt(shell_Psi2)),
            shear_l2=Float64(sqrt(shell_shear2)),
            cumulative_Psi_l2=Float64(sqrt(cumulative_Psi2)),
            cumulative_shear_l2=Float64(sqrt(cumulative_shear2)),
            Psi_shell_fraction=Float64(Psi_fraction),
            shear_shell_fraction=Float64(shear_fraction),
            shell_fraction=Float64(shell_fraction),
            below_tolerance=below,
            consecutive_below=small_shells,
        ))
    end
    multipole_converged = small_shells >= consecutive_shells

    baseline_Psi2 = sum(sum(abs2, mode.time_domain.Psi_h) for mode in values(modes))
    baseline_shear2 = sum(sum(abs2, mode.time_domain.shear_h) for mode in values(modes))
    alternative_keys = Set{Tuple{Symbol,Float64}}()
    for mode in values(modes), alt in mode.time_domain.completion_alternatives
        push!(alternative_keys, (alt.kind, Float64(alt.value)))
    end
    completion_rows = NamedTuple[]
    for altkey in sort!(collect(alternative_keys); by=x -> (String(x[1]), x[2]))
        kind, value = altkey
        Psi_diff2 = 0.0
        shear_diff2 = 0.0
        for mode in values(modes)
            match = findfirst(alt -> alt.kind == kind && Float64(alt.value) == value,
                              mode.time_domain.completion_alternatives)
            match === nothing && continue
            alt = mode.time_domain.completion_alternatives[match]
            Psi_diff2 += alt.Psi_l2_difference^2
            shear_diff2 += alt.shear_l2_difference^2
        end
        Psi_relative = sqrt(Psi_diff2 / max(baseline_Psi2, eps(Float64)))
        shear_relative = sqrt(shear_diff2 / max(baseline_shear2, eps(Float64)))
        push!(completion_rows, (
            kind=kind,
            value=value,
            Psi_relative=Float64(Psi_relative),
            shear_relative=Float64(shear_relative),
            field_relative=Float64(max(Psi_relative, shear_relative)),
            pass=max(Psi_relative, shear_relative) <= completion_tol,
        ))
    end
    soft_rows = [row for row in completion_rows if row.kind == :soft_degree]
    taper_rows = [row for row in completion_rows if row.kind == :high_taper_fraction]
    max_soft = isempty(soft_rows) ? 0.0 : maximum(row.field_relative for row in soft_rows)
    max_taper = isempty(taper_rows) ? 0.0 : maximum(row.field_relative for row in taper_rows)
    return (
        shell_tolerance=shell_tol,
        consecutive_shells=consecutive_shells,
        multipole_converged=multipole_converged,
        shells=shell_rows,
        completion_tolerance=completion_tol,
        completion_rows=completion_rows,
        max_soft_completion_relative=Float64(max_soft),
        max_taper_completion_relative=Float64(max_taper),
        soft_completion_pass=max_soft <= completion_tol,
        taper_completion_pass=max_taper <= completion_tol,
    )
end


function solve_horizon_time_domain(cfg::HorizonTimeDomainConfig;
                                   spectrum_result=nothing)
    if spectrum_result !== nothing
        hasproperty(spectrum_result, :status) ||
            error("spectrum_result does not look like solve_horizon_spectrum output")
        spectrum_result.status == :accepted_direct_band || error(
            "time-domain production requires an accepted direct-band spectrum certificate; " *
            "received $(spectrum_result.status)",
        )
    end
    spectrum_result === nothing && isempty(cfg.frequencies) &&
        error("supply explicit cfg.frequencies or an accepted spectrum_result")
    accepted_omega_max = if spectrum_result === nothing
        maximum(cfg.frequencies)
    elseif hasproperty(spectrum_result, :accepted_direct_omega_max) &&
           isfinite(spectrum_result.accepted_direct_omega_max)
        Float64(spectrum_result.accepted_direct_omega_max)
    else
        maximum(Float64.(spectrum_result.computed_frequencies))
    end
    frequencies = horizon_time_frequency_grid(cfg, accepted_omega_max)
    length(frequencies) >= 2 || error("time-domain solver needs at least two frequencies")
    all(>(0), frequencies) || error("time-domain frequencies must be positive")
    ispow2(cfg.fft_points) || error("fft_points must be a power of two")
    cfg.fft_points >= 64 || error("fft_points must be at least 64")
    cfg.spherical_l_max >= 2 || error("spherical_l_max must be at least two")
    cfg.spheroidal_l_max >= cfg.spherical_l_max + cfg.spheroidal_buffer ||
        error("spheroidal_l_max is too small for the requested spherical block")
    cfg.waveform_completion_tolerance > 0 ||
        error("waveform_completion_tolerance must be positive")
    cfg.time_shell_tolerance > 0 || error("time_shell_tolerance must be positive")
    cfg.time_consecutive_shells >= 1 ||
        error("time_consecutive_shells must be positive")

    keys = Tuple{Int,Int}[
        (ell, m)
        for ell in 2:cfg.spherical_l_max
        for m in -ell:ell
    ]
    positive_Psi = Dict(key => ComplexF64[] for key in keys)
    negative_Psi = Dict(key => ComplexF64[] for key in keys)
    positive_shear = Dict(key => ComplexF64[] for key in keys)
    negative_shear = Dict(key => ComplexF64[] for key in keys)
    frequency_diagnostics = NamedTuple[]
    velocities = nothing

    for omega in frequencies
        cache_cfg = deepcopy(cfg.base)
        cache_cfg.omega = omega
        orbit_cache = build_direct_orbit_cache(cache_cfg)
        velocities === nothing &&
            (velocities = _direct_asymptotic_velocity_data(cache_cfg, orbit_cache))
        m_rows = NamedTuple[]

        for m in -cfg.spherical_l_max:cfg.spherical_l_max
            spherical_l_min = max(2, abs(m))
            spherical_l_min <= cfg.spherical_l_max || continue
            positive = horizon_spherical_field_block(
                cfg.base, m, omega;
                spherical_l_min=spherical_l_min,
                spherical_l_max=cfg.spherical_l_max,
                spheroidal_buffer=cfg.spheroidal_buffer,
                spheroidal_l_max=cfg.spheroidal_l_max,
                mixing_tolerance=cfg.mixing_tolerance,
                angular_ts_tolerance=cfg.angular_ts_tolerance,
                orbit_cache=orbit_cache,
                gauss_panels=cfg.gauss_panels,
            )
            negative = horizon_spherical_field_block(
                cfg.base, m, -omega;
                spherical_l_min=spherical_l_min,
                spherical_l_max=cfg.spherical_l_max,
                spheroidal_buffer=cfg.spheroidal_buffer,
                spheroidal_l_max=cfg.spheroidal_l_max,
                mixing_tolerance=cfg.mixing_tolerance,
                angular_ts_tolerance=cfg.angular_ts_tolerance,
                orbit_cache=orbit_cache,
                gauss_panels=cfg.gauss_panels,
            )
            positive.mixing_converged || error(
                "positive spherical field block failed mixing convergence at omega=$omega, m=$m",
            )
            negative.mixing_converged || error(
                "negative spherical field block failed mixing convergence at omega=$omega, m=$m",
            )
            positive_condition = maximum(
                row.cancellation_condition for row in positive.spheroidal_rows
            )
            negative_condition = maximum(
                row.cancellation_condition for row in negative.spheroidal_rows
            )
            positive_condition_flagged = positive_condition > cfg.cancellation_condition_limit
            negative_condition_flagged = negative_condition > cfg.cancellation_condition_limit

            for ell in spherical_l_min:cfg.spherical_l_max
                key = (ell, m)
                push!(positive_Psi[key], positive.modes[ell].Psi_h)
                push!(negative_Psi[key], negative.modes[ell].Psi_h)
                push!(positive_shear[key], positive.modes[ell].shear_h)
                push!(negative_shear[key], negative.modes[ell].shear_h)
            end
            push!(m_rows, (
                m=m,
                positive_spheroidal_l_max=positive.spheroidal_l_max,
                negative_spheroidal_l_max=negative.spheroidal_l_max,
                positive_max_cancellation=Float64(positive_condition),
                negative_max_cancellation=Float64(negative_condition),
                positive_condition_flagged=positive_condition_flagged,
                negative_condition_flagged=negative_condition_flagged,
                positive_max_angular_ts_residual=maximum(
                    row.angular_ts_relative_residual for row in positive.spheroidal_rows
                ),
                negative_max_angular_ts_residual=maximum(
                    row.angular_ts_relative_residual for row in negative.spheroidal_rows
                ),
            ))
        end
        push!(frequency_diagnostics, (
            omega=omega,
            geodesic_incoming_steps=orbit_cache.geodesic.incoming_saved_steps,
            geodesic_outgoing_steps=orbit_cache.geodesic.outgoing_saved_steps,
            m_blocks=m_rows,
        ))
    end

    modes = Dict{Tuple{Int,Int},Any}()
    for key in keys
        ell, m = key
        use_endpoint = cfg.use_quadrupole_endpoint && ell == 2
        endpoint_Psi = use_endpoint ?
            (omega -> quadrupole_horizon_endpoint_nonanalytic(
                cfg.base, m, omega, velocities,
            ).Psi_h) :
            (omega -> 0.0 + 0.0im)
        endpoint_shear = use_endpoint ?
            (omega -> quadrupole_horizon_endpoint_nonanalytic(
                cfg.base, m, omega, velocities,
            ).shear_h) :
            (omega -> 0.0 + 0.0im)
        synchronous_zero = m == 0 || abs(cfg.base.a) <= 1.0e-14
        zero_Psi = synchronous_zero ? (0.0 + 0.0im) : nothing
        zero_shear = synchronous_zero ? (0.0 + 0.0im) : nothing
        reconstructed = reconstruct_horizon_time_mode(
            frequencies,
            positive_Psi[key], negative_Psi[key],
            positive_shear[key], negative_shear[key];
            fft_points=cfg.fft_points,
            endpoint_Psi=endpoint_Psi,
            endpoint_shear=endpoint_shear,
            soft_fit_points=cfg.soft_fit_points,
            soft_polynomial_degree=cfg.soft_polynomial_degree,
            high_taper_start_fraction=cfg.high_taper_start_fraction,
            zero_frequency_Psi=zero_Psi,
            zero_frequency_shear=zero_shear,
        )
        modes[key] = (
            spherical_l=ell,
            m=m,
            endpoint_completion=use_endpoint ? :quadrupole_exact_nonanalytic :
                :regular_with_higher_order_endpoint_omitted,
            endpoint_nonanalytic_order=synchronous_zero ? ell + 1 : ell,
            positive_Psi=positive_Psi[key],
            negative_Psi=negative_Psi[key],
            positive_shear=positive_shear[key],
            negative_shear=negative_shear[key],
            time_domain=reconstructed,
        )
    end
    times = first(values(modes)).time_domain.times
    multipole_diagnostics = horizon_time_multipole_diagnostics(
        modes;
        shell_tolerance=cfg.time_shell_tolerance,
        consecutive_shells=cfg.time_consecutive_shells,
        completion_tolerance=cfg.waveform_completion_tolerance,
    )
    status = if !multipole_diagnostics.multipole_converged
        :needs_more_spherical_multipoles
    elseif !multipole_diagnostics.soft_completion_pass
        :needs_soft_completion_refinement
    elseif !multipole_diagnostics.taper_completion_pass
        :needs_high_frequency_completion_refinement
    else
        :accepted_time_domain
    end
    return (
        status=status,
        frequencies=frequencies,
        times=times,
        spherical_l_max=cfg.spherical_l_max,
        modes=modes,
        asymptotic_velocities=velocities,
        frequency_diagnostics=frequency_diagnostics,
        spectrum_certificate=spectrum_result,
        multipole_diagnostics=multipole_diagnostics,
        note=cfg.spherical_l_max > 2 ?
            "L>=3 endpoint nonanalyticity begins at the theorem-controlled higher soft order recorded per mode; its unresolved contribution is accepted only through the global waveform completion gate." :
            "All retained modes use the exact quadrupolar endpoint nonanalytic completion.",
    )
end


function solve_horizon(
    base::HorizonScatteringConfig;
    spectrum_config=nothing,
    time_config=nothing,
    compute_time_domain::Bool=true,
)
    spectrum_cfg = spectrum_config === nothing ?
        HorizonSpectrumConfig(base=deepcopy(base)) : deepcopy(spectrum_config)
    spectrum_cfg.base = deepcopy(base)
    spectrum = solve_horizon_spectrum(spectrum_cfg)
    if spectrum.status != :accepted_direct_band
        return (
            status=spectrum.status,
            production_accepted=false,
            base=deepcopy(base),
            spectrum=spectrum,
            time_domain=nothing,
        )
    end
    if !compute_time_domain
        return (
            status=:accepted_spectrum_only,
            production_accepted=false,
            base=deepcopy(base),
            spectrum=spectrum,
            time_domain=nothing,
        )
    end

    time_cfg = time_config === nothing ? HorizonTimeDomainConfig(base=deepcopy(base)) :
                                         deepcopy(time_config)
    time_cfg.base = deepcopy(base)
    if time_config === nothing
        time_cfg.spherical_l_max = max(time_cfg.spherical_l_max, spectrum.spectrum.ell_max + 2)
        time_cfg.spheroidal_l_max = max(time_cfg.spheroidal_l_max,
                                        time_cfg.spherical_l_max + 4)
    end
    time_domain = solve_horizon_time_domain(time_cfg; spectrum_result=spectrum)
    accepted = time_domain.status == :accepted_time_domain
    return (
        status=accepted ? :accepted_horizon_solution : time_domain.status,
        production_accepted=accepted,
        base=deepcopy(base),
        spectrum=spectrum,
        time_domain=time_domain,
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
