module EquatorialScatteringSource

include(joinpath(@__DIR__, "UnboundGeodesicSNSources.jl"))

using .UnboundGeodesicSNSources
import ....KerrGeometry
import ....NativeSN
using LinearAlgebra
using Printf
using SpinWeightedSpheroidalHarmonics

const UG = UnboundGeodesicSNSources

export EquatorialScatteringConfig,
       ScatteringOrbitCache,
       scattering_source_r_outer,
       build_scattering_orbit_cache,
       build_scattering_source,
       build_scattering_source_from_cache,
       build_direct_source_from_trajectories,
       write_scattering_source_csv,
       write_scattering_diagnostics_csv,
       write_scattering_outputs,
       csv_value

Base.@kwdef mutable struct EquatorialScatteringConfig
    a::Float64 = 0.9
    ell::Int = 2
    m::Int = 2
    omega::Float64 = 0.1
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 0.0
    theta_infinity::Float64 = pi / 2
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    particle_mass::Float64 = 1.0
    source_kind::String = "reduced"
    orbit_kind::String = "scattering"
    r_outer_min::Float64 = 80.0
    source_tail_factor::Float64 = 0.0
    turn_buffer::Float64 = 1e-3
    regularize_turn::Bool = true
    asymptotic_tail_correction::Bool = true
    asymptotic_match_phase::Float64 = 100.0
    asymptotic_tail_order::Int = 3
    allow_theta_turns::Bool = false
    npoints::Int = 1001
    nsteps_per_branch::Int = 1800
    out::String = joinpath(@__DIR__, "..", "output", "equatorial_scattering", "source.csv")
    diagnostics_out::String = ""
end

struct ScatteringOrbitCache
    kerr::KerrParams
    constants::GeodesicConstants
    orbit_kind::Symbol
    r_outer::Float64
    r_turn::Float64
    pieces::Vector{OrbitPiece}
    trajectories::Vector{PieceTrajectory}
end

csv_value(x::AbstractFloat) = isnan(x) ? "NaN" : repr(Float64(x))
csv_value(x::Integer) = string(x)
csv_value(x::Bool) = string(x)
function csv_value(x::AbstractString)
    escaped = replace(x, "\"" => "\"\"")
    if occursin(",", escaped) || occursin("\"", escaped) ||
       occursin("\n", escaped) || occursin("\r", escaped)
        return "\"$escaped\""
    end
    return escaped
end

function validate_source_config(cfg::EquatorialScatteringConfig)
    cfg.source_kind in ("reduced", "sn") ||
        error("source_kind must be reduced or sn")
    cfg.omega == 0 && error("omega=0 is not supported by the external Green driver")
    cfg.npoints >= 2 || error("npoints must be at least 2")
    cfg.nsteps_per_branch >= 10 || error("nsteps_per_branch must be at least 10")
    cfg.turn_buffer > 0 || error("turn_buffer must be positive")
    cfg.asymptotic_match_phase > 0 ||
        error("asymptotic_match_phase must be positive")
    cfg.asymptotic_tail_order >= 0 ||
        error("asymptotic_tail_order must be nonnegative")
    return cfg
end

function validate_config(cfg::EquatorialScatteringConfig)
    validate_source_config(cfg)
    cfg.orbit_kind in ("scattering", "plunge") ||
        error("orbit_kind must be scattering or plunge")
    cfg.energy >= 1 || error("energy must be at least 1 for unbound motion")
    theta_potential(KerrParams(a=cfg.a),
                    GeodesicConstants(energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q),
                    cfg.theta_infinity) < -1e-10 &&
        error("theta_infinity lies outside the allowed theta region")
    return cfg
end

function scattering_source_r_outer(cfg::EquatorialScatteringConfig)
    validate_config(cfg)
    tail = cfg.source_tail_factor > 0 ? cfg.source_tail_factor * pi / abs(cfg.omega) : 0.0
    return max(cfg.r_outer_min, tail)
end

effective_asymptotic_tail_correction(cfg::EquatorialScatteringConfig) =
    cfg.asymptotic_tail_correction

function scattering_constants(cfg::EquatorialScatteringConfig)
    return GeodesicConstants(energy=cfg.energy, lz=cfg.lz, carter_q=cfg.carter_q)
end

function scattering_mode(kerr::KerrParams, cfg::EquatorialScatteringConfig)
    harmonic = spin_weighted_spheroidal_harmonic(
        -2,
        cfg.ell,
        cfg.m,
        kerr.a * cfg.omega;
        method="auto",
    )
    mode = swsh_angular_mode(harmonic, cfg.m, kerr.a * cfg.omega; theta_step=1e-5)
    return harmonic, mode
end

function source_phase(kerr::KerrParams, point, cfg::EquatorialScatteringConfig,
                      t_origin, phi_origin)
    return cis(
        cfg.omega * (point.t - t_origin) -
        cfg.m * (point.phi - phi_origin) +
        k_over_delta_antiderivative(kerr, point.r, cfg.omega, cfg.m),
    )
end

function plunge_composite_radial_nodes(kerr::KerrParams,
                                       piece::OrbitPiece,
                                       nsteps::Int;
                                       energy::Float64=1.0)
    piece.radial_sign < 0 || error("plunge composite nodes require an incoming piece")
    piece.r_start > piece.r_stop || error("plunge radii must decrease")
    r_cut = min(200.0, piece.r_start)
    if piece.r_start <= 200.0
        rs_start = KerrGeometry.rstar_from_r(kerr.a, piece.r_start)
        rs_stop = KerrGeometry.rstar_from_r(kerr.a, piece.r_stop)
        required_steps = ceil(Int, (rs_start - rs_stop) / 0.1)
        nsteps >= required_steps || error(
            "plunge orbit needs at least $required_steps steps to keep " *
            "Delta rstar <= 0.1 through r=$(piece.r_start); received $nsteps",
        )
        nodes = Float64[
            KerrGeometry.r_from_rstar(kerr.a, rs)
            for rs in range(rs_start, rs_stop; length=nsteps + 1)
        ]
        nodes[1] = piece.r_start
        nodes[end] = piece.r_stop
        return nodes
    end
    r_cut > piece.r_stop ||
        return collect(range(piece.r_start, piece.r_stop; length=nsteps + 1))

    rs_cut = KerrGeometry.rstar_from_r(kerr.a, r_cut)
    rs_stop = KerrGeometry.rstar_from_r(kerr.a, piece.r_stop)
    inner_steps = max(2, ceil(Int, (rs_cut - rs_stop) / 0.1))
    outer_steps = nsteps - inner_steps
    outer_steps >= 2 || error(
        "plunge orbit needs at least $(inner_steps + 2) steps to keep " *
        "Delta rstar <= 0.1 through r=200M; received $nsteps",
    )

    outer = if energy == 1.0
        q_start = piece.r_start^(3 / 2)
        q_cut = r_cut^(3 / 2)
        collect(range(q_start, q_cut; length=outer_steps + 1)).^(2 / 3)
    else
        collect(range(piece.r_start, r_cut; length=outer_steps + 1))
    end
    inner_rstar = range(rs_cut, rs_stop; length=inner_steps + 1)
    inner = Float64[KerrGeometry.r_from_rstar(kerr.a, rs) for rs in inner_rstar]
    outer[1] = piece.r_start
    outer[end] = r_cut
    inner[1] = r_cut
    inner[end] = piece.r_stop
    return vcat(outer, @view(inner[2:end]))
end

function build_scattering_orbit_cache(cfg::EquatorialScatteringConfig)
    validate_config(cfg)
    kerr = KerrParams(a=cfg.a)
    constants = scattering_constants(cfg)
    r_outer = scattering_source_r_outer(cfg)
    kind = Symbol(cfg.orbit_kind)
    roots = radial_turning_points(kerr, constants; r_max=r_outer, samples=5000)
    if kind === :scattering && isempty(roots)
        error("no scattering turning point found for E=$(cfg.energy), Lz=$(cfg.lz), Q=$(cfg.carter_q)")
    end
    r_turn = kind === :scattering ? roots[end] : NaN
    outer_state = finite_outer_state_from_asymptote(
        kerr,
        constants,
        r_outer,
        cfg.theta_infinity,
        cfg.phi_infinity,
        cfg.theta_sign,
    )

    pieces = orbit_pieces(
        kerr,
        constants;
        kind=kind,
        r_outer=r_outer,
        turn_buffer=cfg.turn_buffer,
        regularized_turn=cfg.regularize_turn && kind === :scattering,
        theta_sign=outer_state.theta_sign,
    )
    radial_nodes = kind === :plunge ?
        [plunge_composite_radial_nodes(
             kerr, piece, cfg.nsteps_per_branch; energy=cfg.energy,
         )
         for piece in pieces] : nothing
    trajectories = integrate_orbit_pieces(
        kerr,
        constants,
        pieces,
        0.0,
        0;
        theta0=outer_state.theta,
        phi0=outer_state.phi,
        nsteps_per_piece=cfg.nsteps_per_branch,
        regularize_turns=cfg.regularize_turn && kind === :scattering,
        allow_theta_turns=cfg.allow_theta_turns,
        radial_nodes_per_piece=radial_nodes,
    )
    return ScatteringOrbitCache(
        kerr,
        constants,
        kind,
        r_outer,
        r_turn,
        pieces,
        trajectories,
    )
end

function validate_orbit_cache(cfg::EquatorialScatteringConfig,
                              cache::ScatteringOrbitCache)
    cfg.a == cache.kerr.a ||
        error("cached orbit spin a does not match source config")
    constants = scattering_constants(cfg)
    cfg.energy == cache.constants.energy ||
        error("cached orbit energy does not match source config")
    cfg.lz == cache.constants.lz ||
        error("cached orbit lz does not match source config")
    cfg.carter_q == cache.constants.carter_q ||
        error("cached orbit carter_q does not match source config")
    Symbol(cfg.orbit_kind) === cache.orbit_kind ||
        error("cached orbit kind does not match source config")
    required_outer = scattering_source_r_outer(cfg)
    required_outer <= cache.r_outer * (1 + 1e-12) ||
        error("cached orbit r_outer=$(cache.r_outer) is smaller than required r_outer=$required_outer")
    theta_potential(cache.kerr, constants, cfg.theta_infinity) < -1e-10 &&
        error("theta_infinity lies outside the allowed theta region")
    return true
end

function clip_trajectory_to_outer(traj::PieceTrajectory, r_outer::Float64)
    supports_radius(traj, r_outer) ||
        error("cached trajectory does not reach requested r_outer=$r_outer")
    tol = 1e-10 * max(1.0, abs(r_outer))
    points = TrajectoryPoint[p for p in traj.points if p.r <= r_outer + tol]
    isempty(points) && error("no cached trajectory points remain after clipping")
    boundary = interpolated_point(traj, r_outer)
    if traj.piece.radial_sign < 0
        if abs(points[1].r - r_outer) > tol
            pushfirst!(points, boundary)
        end
        piece = OrbitPiece(
            label=traj.piece.label,
            r_start=r_outer,
            r_stop=traj.piece.r_stop,
            radial_sign=traj.piece.radial_sign,
            theta_sign=traj.piece.theta_sign,
            r_turn=traj.piece.r_turn,
        )
        return PieceTrajectory(piece, points)
    end
    if abs(points[end].r - r_outer) > tol
        push!(points, boundary)
    end
    piece = OrbitPiece(
        label=traj.piece.label,
        r_start=traj.piece.r_start,
        r_stop=r_outer,
        radial_sign=traj.piece.radial_sign,
        theta_sign=traj.piece.theta_sign,
        r_turn=traj.piece.r_turn,
    )
    return PieceTrajectory(piece, points)
end

function clipped_trajectories(cache::ScatteringOrbitCache,
                              cfg::EquatorialScatteringConfig)
    required_outer = scattering_source_r_outer(cfg)
    if required_outer >= cache.r_outer * (1 - 1e-12)
        return cache.trajectories
    end
    return PieceTrajectory[
        clip_trajectory_to_outer(traj, required_outer)
        for traj in cache.trajectories
    ]
end

function point_with_shifted_origin(kerr::KerrParams,
                                   cfg::EquatorialScatteringConfig,
                                   point::TrajectoryPoint,
                                   t_origin::Float64,
                                   phi_origin::Float64)
    shifted_t = point.t - t_origin
    shifted_phi = point.phi - phi_origin
    chi = cfg.omega * shifted_t - cfg.m * shifted_phi +
          k_over_delta_antiderivative(kerr, point.r, cfg.omega, cfg.m) -
          0.0
    return TrajectoryPoint(
        point.label,
        point.r,
        point.theta,
        shifted_t,
        shifted_phi,
        chi,
        point.ur_sign,
        point.utheta_sign,
        point.ut,
        point.ur,
        point.utheta,
        point.uphi,
        point.radial_potential,
        point.theta_potential,
        point.N,
        point.Mbar,
    )
end

function rephased_trajectories(cache::ScatteringOrbitCache,
                               cfg::EquatorialScatteringConfig)
    clipped = clipped_trajectories(cache, cfg)
    return rephase_trajectories(cache.kerr, clipped, cfg)
end

function rephase_trajectories(kerr::KerrParams,
                              trajectories::Vector{PieceTrajectory},
                              cfg::EquatorialScatteringConfig)
    isempty(trajectories) && error("cannot rephase an empty trajectory list")
    t_origin, phi_origin = if cfg.orbit_kind == "scattering"
        all_points = Iterators.flatten(traj.points for traj in trajectories)
        origin_point = reduce(
            (left, right) -> left.r <= right.r ? left : right,
            all_points,
        )
        (origin_point.t, origin_point.phi)
    else
        (0.0, 0.0)
    end
    return PieceTrajectory[
        PieceTrajectory(
            traj.piece,
            TrajectoryPoint[
                point_with_shifted_origin(
                    kerr, cfg, point, t_origin, phi_origin,
                )
                for point in traj.points
            ],
        )
        for traj in trajectories
    ]
end

function trajectory_phase_origin(trajectories, cfg::EquatorialScatteringConfig)
    cfg.orbit_kind == "scattering" || return (0.0, 0.0)
    all_points = Iterators.flatten(traj.points for traj in trajectories)
    origin_point = reduce(
        (left, right) -> left.r <= right.r ? left : right,
        all_points,
    )
    return origin_point.t, origin_point.phi
end

function plunge_tail_extension_radii(cfg::EquatorialScatteringConfig,
                                     outer_point,
                                     inner_radial_step::Float64)
    radius0 = outer_point.r
    rate0 = abs(point_chi_prime_cached(
        KerrParams(a=cfg.a), outer_point, cfg.omega, cfg.m,
    ))
    abs(cfg.omega) > 0 || error("a plunge asymptotic tail requires nonzero frequency")
    momentum = sqrt(cfg.energy^2 - 1)
    parabolic = momentum == 0
    slow_rate_infinity = parabolic ? NaN :
                          abs(cfg.omega) /
                          (momentum * (cfg.energy + momentum))
    !parabolic && slow_rate_infinity > 0 || parabolic ||
        error("degenerate incoming asymptotic phase")
    worldline_stop = parabolic ?
        (3cfg.asymptotic_match_phase / (sqrt(2.0) * abs(cfg.omega)))^(2 / 3) :
        cfg.asymptotic_match_phase / slow_rate_infinity
    radial_stop = 100.0 / abs(cfg.omega)
    radius_stop = max(
        radius0,
        worldline_stop,
        radial_stop,
    )
    radius0 * rate0 >= cfg.asymptotic_match_phase &&
        radius0 >= radial_stop && return Float64[]
    radius_stop > radius0 * (1 + 1e-12) || return Float64[]

    phase_step = 0.05
    relative_step = 0.01
    step = max(inner_radial_step, eps(Float64) * radius0)
    radii = Float64[radius0]
    while radii[end] < radius_stop
        local_phase_rate = parabolic ? abs(cfg.omega) * sqrt(2.0 * radii[end]) :
                           slow_rate_infinity
        step = min(
            1.2 * step,
            relative_step * radii[end],
            phase_step / local_phase_rate,
            radius_stop - radii[end],
        )
        step > 0 || error("plunge asymptotic extension grid stalled")
        push!(radii, radii[end] + step)
        length(radii) <= 1_000_001 ||
            error("plunge asymptotic extension exceeded 1000000 phase-resolved steps")
    end
    radii[end] = radius_stop
    return radii
end

function extend_plunge_trajectory_for_abel_tail(
    cfg::EquatorialScatteringConfig,
    kerr::KerrParams,
    constants::GeodesicConstants,
    trajectory::PieceTrajectory,
)
    cfg.orbit_kind == "plunge" || return trajectory
    cfg.asymptotic_tail_correction || return trajectory
    length(trajectory.points) >= 2 ||
        error("plunge trajectory needs at least two points for tail matching")
    outer_point = trajectory.points[1].r >= trajectory.points[end].r ?
                  trajectory.points[1] : trajectory.points[end]
    inner_step = trajectory.points[1].r >= trajectory.points[end].r ?
                 abs(trajectory.points[1].r - trajectory.points[2].r) :
                 abs(trajectory.points[end].r - trajectory.points[end - 1].r)
    radii = plunge_tail_extension_radii(cfg, outer_point, inner_step)
    isempty(radii) && return trajectory
    extension_piece = OrbitPiece(
        label=:plunge_asymptotic_extension,
        r_start=radii[1],
        r_stop=radii[end],
        radial_sign=outer_point.ur_sign,
        theta_sign=outer_point.utheta_sign,
    )
    extension = integrate_orbit_piece(
        kerr,
        constants,
        extension_piece,
        cfg.omega,
        cfg.m;
        t0=outer_point.t,
        theta0=outer_point.theta,
        phi0=outer_point.phi,
        chi0=0.0,
        radial_nodes=radii,
        allow_theta_turns=cfg.allow_theta_turns,
    )
    inward_points = trajectory.points[1].r >= trajectory.points[end].r ?
                    trajectory.points : reverse(trajectory.points)
    extended_points = vcat(reverse(extension.points[2:end]), inward_points)
    extended_piece = OrbitPiece(
        label=trajectory.piece.label,
        r_start=extended_points[1].r,
        r_stop=extended_points[end].r,
        radial_sign=trajectory.piece.radial_sign,
        theta_sign=extended_points[1].utheta_sign,
        r_turn=trajectory.piece.r_turn,
    )
    return PieceTrajectory(extended_piece, extended_points)
end

function point_to_state(point)
    return BLState(
        r=point.r,
        theta=point.theta,
        t=point.t,
        phi=point.phi,
        ur_sign=point.ur_sign,
        utheta_sign=point.utheta_sign,
    )
end

finite_complex(z) = isfinite(real(z)) && isfinite(imag(z))

function oscillatory_segment(x0, x1, a0, a1, phase0, phase1)
    h = x1 - x0
    dphase = phase1 - phase0
    if abs(dphase) < 1e-6
        return 0.5 * h * (a0 * cis(phase0) + a1 * cis(phase1))
    end
    ed = cis(dphase)
    idphase = 1im * dphase
    i0 = (ed - 1) / idphase
    i1 = ed / idphase + (ed - 1) / (dphase^2)
    return h * cis(phase0) * (a0 * i0 + (a1 - a0) * i1)
end

function reverse_oscillatory_integral(r, values)
    n = length(r)
    n == length(values) || error("r and values lengths differ")
    n >= 2 || error("need at least two points for reverse integration")
    result = Vector{ComplexF64}(undef, n)
    return reverse_oscillatory_integral!(result, r, values)
end

@inline scaled_integrand_value(values, ::Nothing, ::Nothing, i) = values[i]
@inline scaled_integrand_value(values, scale, ::Nothing, i) = values[i] * scale[i]
@inline scaled_integrand_value(values, ::Nothing, moment, i) = values[i] * moment[i]
@inline scaled_integrand_value(values, scale, moment, i) =
    moment[i] * values[i] * scale[i]

function reverse_oscillatory_integral!(result, r, values;
                                       scale=nothing, moment=nothing)
    n = length(r)
    n == length(values) == length(result) || error("reverse-integral array lengths differ")
    scale === nothing || n == length(scale) || error("reverse-integral scale length differs")
    moment === nothing || n == length(moment) || error("reverse-integral moment length differs")
    n >= 2 || error("need at least two points for reverse integration")

    initialize_reverse_oscillatory_tail!(
        result, r, values; scale=scale, moment=moment,
    )
    for i in (n - 2):-1:1
        h0 = r[i + 1] - r[i]
        h1 = r[i + 2] - r[i + 1]
        h0 > 0 && h1 > 0 || error("reverse integration grid must increase")
        span = h0 + h1
        w0 = span * (2h0 - h1) / (6h0)
        w1 = span^3 / (6h0 * h1)
        w2 = span * (2h1 - h0) / (6h1)
        result[i] = result[i + 2] +
                    w0 * scaled_integrand_value(values, scale, moment, i) +
                    w1 * scaled_integrand_value(values, scale, moment, i + 1) +
                    w2 * scaled_integrand_value(values, scale, moment, i + 2)
    end
    return result
end

function initialize_reverse_oscillatory_tail!(result, r, values;
                                              scale=nothing, moment=nothing)
    n = length(r)
    result[n] = 0.0 + 0.0im
    value0 = scaled_integrand_value(values, scale, moment, n - 1)
    value1 = scaled_integrand_value(values, scale, moment, n)
    phase0 = angle(value0)
    phase1_raw = angle(value1)
    dphase = phase1_raw - phase0
    dphase -= 2pi * round(dphase / (2pi))
    phase1 = phase0 + dphase
    result[n - 1] = oscillatory_segment(
        r[n - 1], r[n],
        value0 * cis(-phase0), value1 * cis(-phase1),
        phase0, phase1,
    )
    return nothing
end

function reverse_oscillatory_integrals3!(result1, result2, result3,
                                         r, values1, values2;
                                         scale=nothing, moment=nothing)
    n = length(r)
    if values1 === nothing
        fill!(result1, 0.0 + 0.0im)
    else
        initialize_reverse_oscillatory_tail!(
            result1, r, values1; scale=scale,
        )
    end
    initialize_reverse_oscillatory_tail!(
        result2, r, values2; scale=scale,
    )
    initialize_reverse_oscillatory_tail!(
        result3, r, values2; scale=scale, moment=moment,
    )
    @inbounds for i in (n - 2):-1:1
        h0 = r[i + 1] - r[i]
        h1 = r[i + 2] - r[i + 1]
        h0 > 0 && h1 > 0 || error("reverse integration grid must increase")
        span = h0 + h1
        w0 = span * (2h0 - h1) / (6h0)
        w1 = span^3 / (6h0 * h1)
        w2 = span * (2h1 - h0) / (6h1)
        if values1 !== nothing
            result1[i] = result1[i + 2] +
                         w0 * scaled_integrand_value(values1, scale, nothing, i) +
                         w1 * scaled_integrand_value(values1, scale, nothing, i + 1) +
                         w2 * scaled_integrand_value(values1, scale, nothing, i + 2)
        end
        result2[i] = result2[i + 2] +
                     w0 * scaled_integrand_value(values2, scale, nothing, i) +
                     w1 * scaled_integrand_value(values2, scale, nothing, i + 1) +
                     w2 * scaled_integrand_value(values2, scale, nothing, i + 2)
        result3[i] = result3[i + 2] +
                     w0 * scaled_integrand_value(values2, scale, moment, i) +
                     w1 * scaled_integrand_value(values2, scale, moment, i + 1) +
                     w2 * scaled_integrand_value(values2, scale, moment, i + 2)
    end
    return nothing
end

function reverse_phase_aware_integral!(result, x, values, phase_values;
                                       phase_arguments=nothing,
                                       scale=nothing, moment=nothing)
    n = length(x)
    n == length(values) == length(phase_values) == length(result) ||
        error("phase-aware reverse-integral array lengths differ")
    phase_arguments === nothing || n == length(phase_arguments) ||
        error("phase-aware reverse-integral phase lengths differ")
    result[n] = 0.0 + 0.0im
    @inbounds for i in (n - 1):-1:1
        value0 = scaled_integrand_value(values, scale, moment, i)
        value1 = scaled_integrand_value(values, scale, moment, i + 1)
        amplitude0 = value0 * conj(phase_values[i])
        amplitude1 = value1 * conj(phase_values[i + 1])
        if phase_arguments === nothing
            phase_increment = angle(phase_values[i + 1] * conj(phase_values[i]))
        else
            phase_increment = phase_arguments[i + 1] - phase_arguments[i]
        end
        width = x[i + 1] - x[i]
        phase_factor = phase_values[i]
        phase_ratio = phase_values[i + 1] * conj(phase_factor)
        if abs(phase_increment) < 1e-6
            segment = 0.5 * width * (
                phase_factor * amplitude0 + phase_values[i + 1] * amplitude1
            )
        else
            inverse_i_phase = inv(1im * phase_increment)
            zeroth_moment = (phase_ratio - 1) * inverse_i_phase
            first_moment = phase_ratio * inverse_i_phase +
                            (phase_ratio - 1) / phase_increment^2
            segment = width * phase_factor * (
                amplitude0 * zeroth_moment +
                (amplitude1 - amplitude0) * first_moment
            )
        end
        result[i] = result[i + 1] + segment
    end
    return result
end

function interp_complex(xs, ys, x)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    lo = 1
    hi = length(xs)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if xs[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    t = (x - xs[lo]) / (xs[lo + 1] - xs[lo])
    return ys[lo] + t * (ys[lo + 1] - ys[lo])
end

function point_chi_prime(kerr::KerrParams, constants::GeodesicConstants,
                         point, omega, m)
    state = point_to_state(point)
    u = four_velocity(kerr, constants, state)
    proj = source_projections(kerr, state, u)
    return chi_prime_r(kerr, state, u, proj, omega, m)
end

function point_chi_prime_cached(kerr::KerrParams, point, omega, m)
    phi_prime = point.uphi / point.ur + kerr.a / delta(kerr, point.r)
    xi = (kerr.a * omega * sin(point.theta)^2 - m) * phi_prime
    return omega * point.N / point.ur + xi
end

function oscillatory_tail(value_at_outer, chi_prime; min_abs_chi_prime=1e-8)
    abs(chi_prime) < min_abs_chi_prime && return 0.0 + 0.0im
    return -value_at_outer / (1im * chi_prime)
end

function oscillatory_first_moment_tail(value_at_outer, chi_prime;
                                       min_abs_chi_prime=1e-8)
    abs(chi_prime) < min_abs_chi_prime && return 0.0 + 0.0im
    return -value_at_outer / chi_prime^2
end

function polynomial_product(a, b, degree)
    result = zeros(ComplexF64, degree + 1)
    for i in 0:min(degree, length(a) - 1)
        for j in 0:min(degree - i, length(b) - 1)
            result[i + j + 1] += a[i + 1] * b[j + 1]
        end
    end
    return result
end

function reciprocal_series(a, degree)
    abs(a[1]) > sqrt(eps(Float64)) || error("singular tail phase derivative")
    result = zeros(ComplexF64, degree + 1)
    result[1] = inv(a[1])
    for n in 1:degree
        total = 0.0 + 0.0im
        for k in 1:min(n, length(a) - 1)
            total += a[k + 1] * result[n - k + 1]
        end
        result[n + 1] = -total / a[1]
    end
    return result
end


function derivative_series(a, degree)
    result = zeros(ComplexF64, degree + 1)
    for n in 0:min(degree, length(a) - 2)
        result[n + 1] = (n + 1) * a[n + 2]
    end
    return result
end


function fit_outer_taylor(r, values; degree::Int=7, max_points::Int=64)
    n = length(r)
    n == length(values) || error("tail fit arrays have different lengths")
    count = min(n, max(max_points, degree + 1))
    first = n - count + 1
    radius_scale = max(r[end] - r[first], 1.0)
    x = (r[first:end] .- r[end]) ./ radius_scale
    fit_degree = min(degree, count - 1)
    vandermonde = Matrix{Float64}(undef, count, fit_degree + 1)
    for i in 1:count, p in 0:fit_degree
        vandermonde[i, p + 1] = x[i]^p
    end
    scaled_coefficients = vandermonde \ ComplexF64.(values[first:end])
    coefficients = zeros(ComplexF64, degree + 1)
    for p in 0:fit_degree
        coefficients[p + 1] = scaled_coefficients[p + 1] / radius_scale^p
    end
    return coefficients
end


function asymptotic_fit_point_count(r;
                                    minimum_points::Int=384,
                                    radial_window_fraction::Float64=0.025)
    n = length(r)
    n >= 2 || error("need at least two radii for an asymptotic fit window")
    minimum_points >= 2 || error("minimum asymptotic fit points must be at least two")
    0 < radial_window_fraction < 1 ||
        error("asymptotic radial fit-window fraction must lie between zero and one")
    cutoff = r[end] - radial_window_fraction * (r[end] - r[1])
    first = searchsortedfirst(r, cutoff)
    radial_window_points = n - first + 1
    return min(n, max(minimum_points, radial_window_points))
end


function fitted_oscillatory_tail(r, integrand, phase_values, chi_prime_values;
                                 order::Int=3, max_points::Int=64,
                                 radial_power::Int=0)
    n = length(r)
    (n == length(integrand) && n == length(phase_values) &&
     n == length(chi_prime_values)) || error("tail arrays have different lengths")
    abs(chi_prime_values[end]) > 1e-8 || return 0.0 + 0.0im
    degree = max(2 * order + 1, 5)
    count = min(n, max(max_points, degree + 1))
    first = n - count + 1
    r_outer = @view r[first:n]
    amplitude = Vector{ComplexF64}(undef, count)
    @inbounds for (j, i) in enumerate(first:n)
        radial_factor = radial_power == 0 ? 1.0 : r[i]^radial_power
        amplitude[j] = radial_factor * integrand[i] * conj(phase_values[i])
    end
    amplitude_series = fit_outer_taylor(
        r_outer, amplitude; degree=degree, max_points=count,
    )
    phase_derivative_series = fit_outer_taylor(
        r_outer, @view(chi_prime_values[first:n]); degree=degree, max_points=count,
    )
    inverse_i_phase_derivative = -1im .* reciprocal_series(
        phase_derivative_series, degree,
    )
    term = polynomial_product(
        amplitude_series, inverse_i_phase_derivative, degree,
    )
    tail_amplitude = -term[1]
    sign = 1.0
    for _ in 1:order
        term = polynomial_product(
            derivative_series(term, degree), inverse_i_phase_derivative, degree,
        )
        tail_amplitude += sign * term[1]
        sign = -sign
    end
    return phase_values[end] * tail_amplitude
end

function parabolic_fitted_oscillatory_tail(r, integrand, phase_values,
                                            chi_prime_values;
                                            order::Int=3, max_points::Int=64)
    y = sqrt.(r)
    jacobian = 2 .* y
    return fitted_oscillatory_tail(
        y, integrand .* jacobian, phase_values, chi_prime_values .* jacobian;
        order=order, max_points=max_points,
    )
end

function mathcalW_channel_from_values!(W, scratch1, scratch2, scratch3,
                                       r, v0, v1, v2, chi_prime_input;
                                       asymptotic_tail_correction::Bool=true,
                                       integration_x=nothing,
                                       dr_dx=nothing,
                                        phase_values=nothing,
                                        phase_arguments=nothing,
                                        tail_order::Int=3,
                                        parabolic_tail::Bool=false)
    x = integration_x === nothing ? r : integration_x
    jac = dr_dx
    if phase_values === nothing
        reverse_oscillatory_integrals3!(
            scratch1, scratch2, scratch3, x, v1, v2;
            scale=jac, moment=r,
        )
    else
        if v1 === nothing
            fill!(scratch1, 0.0 + 0.0im)
        else
            reverse_phase_aware_integral!(
                scratch1, x, v1, phase_values;
                phase_arguments=phase_arguments, scale=jac,
            )
        end
        reverse_phase_aware_integral!(
            scratch2, x, v2, phase_values;
            phase_arguments=phase_arguments, scale=jac,
        )
        reverse_phase_aware_integral!(
            scratch3, x, v2, phase_values;
            phase_arguments=phase_arguments, scale=jac, moment=r,
        )
    end

    if !asymptotic_tail_correction
        @inbounds @simd for i in eachindex(W)
            value0 = v0 === nothing ? 0.0 + 0.0im : v0[i]
            W[i] = value0 + scratch1[i] + scratch3[i] - r[i] * scratch2[i]
        end
        return (inner_m0=scratch2[1], inner_m1=scratch1[1] + scratch3[1])
    end

    chi_prime_outer = chi_prime_input isa Number ? chi_prime_input : chi_prime_input[end]
    if phase_values === nothing || chi_prime_input isa Number
        outer_tail_1 = v1 === nothing ? 0.0 + 0.0im :
                       oscillatory_tail(v1[end], chi_prime_outer)
        outer_tail_2 = oscillatory_tail(v2[end], chi_prime_outer)
        outer_moment_2 = oscillatory_first_moment_tail(v2[end], chi_prime_outer)
    else
        max_points = asymptotic_fit_point_count(r)
        tail_function = parabolic_tail ?
            parabolic_fitted_oscillatory_tail : fitted_oscillatory_tail
        outer_tail_1 = v1 === nothing ? 0.0 + 0.0im :
            tail_function(
                r, v1, phase_values, chi_prime_input;
                order=tail_order, max_points=max_points,
            )
        outer_tail_2 = tail_function(
            r, v2, phase_values, chi_prime_input;
            order=tail_order, max_points=max_points,
        )
        outer_moment_2 = if parabolic_tail
            tail_function(
                r, r .* v2, phase_values, chi_prime_input;
                order=tail_order, max_points=max_points,
            ) - r[end] * outer_tail_2
        else
            fitted_oscillatory_tail(
                r, v2, phase_values, chi_prime_input;
                order=tail_order, max_points=max_points, radial_power=1,
            ) - r[end] * outer_tail_2
        end
    end
    r_outer = r[end]
    @inbounds @simd for i in eachindex(W)
        value0 = v0 === nothing ? 0.0 + 0.0im : v0[i]
        W[i] = value0 + scratch1[i] + outer_tail_1 +
               scratch3[i] - r[i] * scratch2[i] +
               (r_outer - r[i]) * outer_tail_2 + outer_moment_2
    end
    return (
        inner_m0=scratch2[1] + outer_tail_2,
        inner_m1=scratch1[1] + outer_tail_1 + scratch3[1] +
                 r_outer * outer_tail_2 + outer_moment_2,
    )
end

function channel_integration_grid(r, r_turn)
    if isfinite(r_turn)
        x = sqrt.(max.(r .- r_turn, 0.0))
        return x, 2 .* x
    end
    return Float64.(r), ones(Float64, length(r))
end

function mathcalW_channel_from_values(r, v0, v1, v2, chi_prime_input;
                                      asymptotic_tail_correction::Bool=true,
                                      integration_x=nothing,
                                      dr_dx=nothing,
                                      phase_values=nothing,
                                      phase_arguments=nothing,
                                      tail_order::Int=3,
                                      parabolic_tail::Bool=false)
    W = Vector{ComplexF64}(undef, length(r))
    scratch1 = similar(W)
    scratch2 = similar(W)
    scratch3 = similar(W)
    moments = mathcalW_channel_from_values!(
        W, scratch1, scratch2, scratch3,
        r, v0, v1, v2, chi_prime_input;
        asymptotic_tail_correction=asymptotic_tail_correction,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        tail_order=tail_order,
        parabolic_tail=parabolic_tail,
    )
    return (W=W, inner_m0=moments.inner_m0, inner_m1=moments.inner_m1)
end

function points_ordered_by_radius(points)
    length(points) <= 1 && return points
    if points[1].r <= points[end].r
        ordered = points
    else
        ordered = @view points[end:-1:1]
    end
    for i in 1:(length(ordered) - 1)
        if ordered[i + 1].r < ordered[i].r
            return sort(points; by=p -> p.r)
        end
    end
    return ordered
end

function table_points(traj::PieceTrajectory, r_turn, cfg::EquatorialScatteringConfig)
    points = points_ordered_by_radius(traj.points)
    filtered = if !cfg.regularize_turn || !isfinite(r_turn)
        points
    else
        min_delta = 1e-12 * max(1.0, abs(r_turn))
        first_regular = findfirst(p -> p.r > r_turn + min_delta, points)
        first_regular === nothing &&
            error("no points outside the regularized radial turn")
        @view points[first_regular:end]
    end
    length(filtered) >= 4 ||
        error("not enough points outside the regularized turn; increase nsteps_per_branch")

    return filtered
end

function add_turn_moment_correction(
    table,
    kerr::KerrParams,
    constants::GeodesicConstants,
    traj::PieceTrajectory,
    points,
    mode::AngularMode,
    cfg::EquatorialScatteringConfig,
    t_origin,
    phi_origin,
)
    isfinite(traj.piece.r_turn) || return table
    full_points = points_ordered_by_radius(traj.points)
    turn_point = first(full_points)
    first_regular = first(points)
    u_regular = sqrt(max(first_regular.r - traj.piece.r_turn, 0.0))
    u_regular > 0 || return table

    radial_prime = UG.radial_potential_derivative(
        kerr, constants, traj.piece.r_turn,
    )
    radial_prime > 0 ||
        error("turning point must have positive radial-potential derivative")
    turn_measure =
        2 * UG.sigma(kerr, turn_point.r, turn_point.theta) /
        sqrt(radial_prime)
    regular_measure = 2u_regular / abs(first_regular.ur)
    turn_coefficients = UG.q_distribution_coefficients(
        kerr, constants, mode, turn_point, cfg.omega, cfg.m;
        t_origin=t_origin, phi_origin=phi_origin,
    )
    regular_coefficients = UG.q_distribution_coefficients(
        kerr, constants, mode, first_regular, cfg.omega, cfg.m;
        t_origin=t_origin, phi_origin=phi_origin,
    )
    correction(channel, moment) = begin
        c0 = getproperty(turn_coefficients, channel)
        c1 = getproperty(regular_coefficients, channel)
        value0 = moment == 0 ? c0.q0 :
                 turn_point.r * c0.q0 - c0.q1
        value1 = moment == 0 ? c1.q0 :
                 first_regular.r * c1.q0 - c1.q1
        0.5 * u_regular *
        (turn_measure * value0 + regular_measure * value1)
    end
    return merge(table, (
        inner_m0_nn=table.inner_m0_nn + correction(:nn, 0),
        inner_m1_nn=table.inner_m1_nn + correction(:nn, 1),
        inner_m0_nmb=table.inner_m0_nmb + correction(:nm, 0),
        inner_m1_nmb=table.inner_m1_nmb + correction(:nm, 1),
        inner_m0_mbmb=table.inner_m0_mbmb + correction(:mm, 0),
        inner_m1_mbmb=table.inner_m1_mbmb + correction(:mm, 1),
    ))
end

function build_piece_table_from_direct_distribution(
    kerr::KerrParams,
    constants::GeodesicConstants,
    points,
    mode::AngularMode,
    cfg::EquatorialScatteringConfig,
    traj::PieceTrajectory,
    t_origin,
    phi_origin,
)
    npoints = length(points)
    r = Vector{Float64}(undef, npoints)
    f2 = Vector{ComplexF64}(undef, npoints)
    g1 = similar(f2)
    g2 = similar(f2)
    h0 = similar(f2)
    h1 = similar(f2)
    h2 = similar(f2)
    phases = similar(f2)
    phase_arguments = Vector{Float64}(undef, npoints)
    chi_prime_values = Vector{Float64}(undef, npoints)
    terms = UG.TeukolskyATermsBuffer()
    @inbounds for i in eachindex(points)
        point = points[i]
        r[i] = point.r
        chi = k_over_delta_antiderivative(kerr, point.r, cfg.omega, cfg.m)
        orbit_argument = cfg.omega * (point.t - t_origin) -
                         cfg.m * (point.phi - phi_origin)
        orbit_phase = cis(orbit_argument)
        echi = cis(chi)
        phase_arguments[i] = orbit_argument + chi
        phases[i] = cis(phase_arguments[i])
        UG.fill_q_distribution_integrands!(
            f2, g1, g2, h0, h1, h2, i,
            terms, kerr, mode, point, cfg.omega, cfg.m, orbit_phase, echi,
        )
        chi_prime_values[i] = point_chi_prime_cached(
            kerr, point, cfg.omega, cfg.m,
        )
    end

    table = build_piece_table_from_integrands(
        kerr, constants, traj, cfg, r,
        nothing, nothing, f2, nothing, g1, g2, h0, h1, h2,
        points[end], chi_prime_values, phases, phase_arguments,
    )
    return add_turn_moment_correction(
        table, kerr, constants, traj, points, mode, cfg,
        t_origin, phi_origin,
    )
end

function build_piece_table_from_integrands(kerr::KerrParams, constants::GeodesicConstants,
                                           traj::PieceTrajectory,
                                           cfg::EquatorialScatteringConfig,
                                           r,
                                           f0, f1, f2,
                                           g0, g1, g2,
                                           h0, h1, h2,
                                           outer_point,
                                           chi_prime_values,
                                           phase_values,
                                           phase_arguments)

    integration_x, dr_dx = channel_integration_grid(r, traj.piece.r_turn)
    chi_prime_outer = point_chi_prime_cached(kerr, outer_point, cfg.omega, cfg.m)
    use_tail = effective_asymptotic_tail_correction(cfg)
    Wnn = Vector{ComplexF64}(undef, length(r))
    Wnmb = similar(Wnn)
    Wmbmb = similar(Wnn)
    scratch1 = similar(Wnn)
    scratch2 = similar(Wnn)
    scratch3 = similar(Wnn)
    nn_solution = mathcalW_channel_from_values!(
        Wnn, scratch1, scratch2, scratch3,
        r, f0, f1, f2, chi_prime_values;
        asymptotic_tail_correction=use_tail,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        tail_order=cfg.asymptotic_tail_order,
        parabolic_tail=cfg.energy == 1.0,
    )
    nm_solution = mathcalW_channel_from_values!(
        Wnmb, scratch1, scratch2, scratch3,
        r, g0, g1, g2, chi_prime_values;
        asymptotic_tail_correction=use_tail,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        tail_order=cfg.asymptotic_tail_order,
        parabolic_tail=cfg.energy == 1.0,
    )
    mm_solution = mathcalW_channel_from_values!(
        Wmbmb, scratch1, scratch2, scratch3,
        r, h0, h1, h2, chi_prime_values;
        asymptotic_tail_correction=use_tail,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        tail_order=cfg.asymptotic_tail_order,
        parabolic_tail=cfg.energy == 1.0,
    )

    all(finite_complex, Wnn) || error("non-finite Wnn table for $(traj.piece.label)")
    all(finite_complex, Wnmb) || error("non-finite Wnmb table for $(traj.piece.label)")
    all(finite_complex, Wmbmb) || error("non-finite Wmbmb table for $(traj.piece.label)")

    return (
        label=traj.piece.label,
        r=r,
        Wnn=Wnn,
        Wnmb=Wnmb,
        Wmbmb=Wmbmb,
        inner_m0_nn=nn_solution.inner_m0,
        inner_m1_nn=nn_solution.inner_m1,
        inner_m0_nmb=nm_solution.inner_m0,
        inner_m1_nmb=nm_solution.inner_m1,
        inner_m0_mbmb=mm_solution.inner_m0,
        inner_m1_mbmb=mm_solution.inner_m1,
        integration_x=integration_x,
        dr_dx=dr_dx,
        phase_values=phase_values,
        phase_arguments=phase_arguments,
        chi_prime_values=chi_prime_values,
        chi_prime_outer=chi_prime_outer,
        asymptotic_tail_correction=effective_asymptotic_tail_correction(cfg),
    )
end

function sum_table_values(tables, key, r)
    total = 0.0 + 0.0im
    for table in tables
        total += interp_complex(table.r, getproperty(table, key), r)
    end
    return total
end

function tables_share_grid(tables, r)
    for table in tables
        length(table.r) == length(r) || return false
        for i in eachindex(r)
            table.r[i] == r[i] || return false
        end
    end
    return true
end

function branch_sum_channels(tables, r)
    if tables_share_grid(tables, r)
        Wnn = zeros(ComplexF64, length(r))
        Wnmb = similar(Wnn)
        Wmbmb = similar(Wnn)
        fill!(Wnmb, 0.0 + 0.0im)
        fill!(Wmbmb, 0.0 + 0.0im)
        for table in tables
            @inbounds @simd for i in eachindex(r)
                Wnn[i] += table.Wnn[i]
                Wnmb[i] += table.Wnmb[i]
                Wmbmb[i] += table.Wmbmb[i]
            end
        end
        return Wnn, Wnmb, Wmbmb
    end
    return (
        ComplexF64[sum_table_values(tables, :Wnn, x) for x in r],
        ComplexF64[sum_table_values(tables, :Wnmb, x) for x in r],
        ComplexF64[sum_table_values(tables, :Wmbmb, x) for x in r],
    )
end

function build_branch_summed_table(tables, cfg::EquatorialScatteringConfig)
    isempty(tables) && error("cannot build a branch-summed table from no pieces")
    r = tables[1].r
    integration_x = hasproperty(tables[1], :integration_x) ? tables[1].integration_x : r
    dr_dx = hasproperty(tables[1], :dr_dx) ? tables[1].dr_dx : ones(Float64, length(r))
    Wnn, Wnmb, Wmbmb = branch_sum_channels(tables, r)

    return (
        label=:branch_sum,
        r=r,
        Wnn=Wnn,
        Wnmb=Wnmb,
        Wmbmb=Wmbmb,
        inner_m0_nn=sum(table.inner_m0_nn for table in tables),
        inner_m1_nn=sum(table.inner_m1_nn for table in tables),
        inner_m0_nmb=sum(table.inner_m0_nmb for table in tables),
        inner_m1_nmb=sum(table.inner_m1_nmb for table in tables),
        inner_m0_mbmb=sum(table.inner_m0_mbmb for table in tables),
        inner_m1_mbmb=sum(table.inner_m1_mbmb for table in tables),
        integration_x=integration_x,
        dr_dx=dr_dx,
        chi_prime_outer=NaN,
        asymptotic_tail_correction=effective_asymptotic_tail_correction(cfg),
    )
end

function build_piece_table(kerr::KerrParams, constants::GeodesicConstants,
                           traj::PieceTrajectory, mode::AngularMode,
                           cfg::EquatorialScatteringConfig, r_turn,
                           t_origin, phi_origin)
    points = table_points(traj, r_turn, cfg)
    return build_piece_table_from_direct_distribution(
        kerr, constants, points, mode, cfg, traj, t_origin, phi_origin,
    )
end

function assemble_direct_source(cfg::EquatorialScatteringConfig,
                                kerr::KerrParams,
                                constants::GeodesicConstants,
                                orbit_kind::Symbol,
                                r_turn,
                                pieces::Vector{OrbitPiece},
                                trajectories::Vector{PieceTrajectory},
                                harmonic,
                                mode::AngularMode,
                                t_origin,
                                phi_origin)
    tables = [
        build_piece_table(
            kerr, constants, traj, mode, cfg, r_turn, t_origin, phi_origin,
        )
        for traj in trajectories
    ]
    branch_summed_table = build_branch_summed_table(tables, cfg)
    table_support = (
        r_minimum=maximum(table.r[1] for table in tables),
        r_maximum=minimum(table.r[end] for table in tables),
    )

    function reduced_at_r(r)
        inside_turn = orbit_kind === :scattering &&
                      r < branch_summed_table.r[1]
        Wnn = inside_turn ?
              branch_summed_table.inner_m1_nn -
              r * branch_summed_table.inner_m0_nn :
              interp_complex(
                  branch_summed_table.r, branch_summed_table.Wnn, r,
              )
        Wnmb = inside_turn ?
               branch_summed_table.inner_m1_nmb -
               r * branch_summed_table.inner_m0_nmb :
               interp_complex(
                   branch_summed_table.r, branch_summed_table.Wnmb, r,
               )
        Wmbmb = inside_turn ?
                branch_summed_table.inner_m1_mbmb -
                r * branch_summed_table.inner_m0_mbmb :
                interp_complex(
                    branch_summed_table.r, branch_summed_table.Wmbmb, r,
                )
        total = Wnn + Wnmb + Wmbmb
        prefactor = delta(kerr, r) / (r^2 * (r^2 + kerr.a^2)^(3 / 2))
        outgoing_phase = exp(
            -1im * k_over_delta_antiderivative(kerr, r, cfg.omega, cfg.m),
        )
        return cfg.particle_mass * total * prefactor * outgoing_phase
    end

    return (
        kerr=kerr,
        constants=constants,
        harmonic=harmonic,
        mode=mode,
        particle_mass=cfg.particle_mass,
        t_origin=Float64(t_origin),
        phi_origin=Float64(phi_origin),
        r_turn=r_turn,
        r_minimum=orbit_kind === :scattering ?
                  KerrGeometry.r_from_rstar(kerr.a, -50.0) :
                  table_support.r_minimum,
        r_maximum=table_support.r_maximum,
        table_r_maximum=table_support.r_maximum,
        pieces=pieces,
        trajectories=trajectories,
        tables=tables,
        branch_summed_table=branch_summed_table,
        table_r_minimum=table_support.r_minimum,
        reduced_at_r=reduced_at_r,
    )
end

function build_direct_source_from_trajectories(
    cfg::EquatorialScatteringConfig,
    kerr::KerrParams,
    constants::GeodesicConstants,
    trajectories::Vector{PieceTrajectory};
    orbit_kind::Symbol=:plunge,
    r_turn::Float64=NaN,
)
    validate_source_config(cfg)
    orbit_kind in (:scattering, :plunge) ||
        error("orbit_kind must be scattering or plunge")
    cfg.a == kerr.a || error("trajectory spin does not match source config")
    cfg.energy == constants.energy ||
        error("trajectory energy does not match source config")
    cfg.lz == constants.lz || error("trajectory lz does not match source config")
    cfg.carter_q == constants.carter_q ||
        error("trajectory Carter constant does not match source config")
    theta_potential(kerr, constants, cfg.theta_infinity) < -1e-10 &&
        error("theta_infinity lies outside the allowed theta region")
    isempty(trajectories) && error("trajectory list is empty")
    pieces = OrbitPiece[traj.piece for traj in trajectories]
    t_origin, phi_origin = trajectory_phase_origin(trajectories, cfg)
    harmonic, mode = scattering_mode(kerr, cfg)
    return assemble_direct_source(
        cfg,
        kerr,
        constants,
        orbit_kind,
        r_turn,
        pieces,
        trajectories,
        harmonic,
        mode,
        t_origin,
        phi_origin,
    )
end

function build_scattering_source_from_cache(cfg::EquatorialScatteringConfig,
                                            cache::ScatteringOrbitCache)
    validate_config(cfg)
    validate_orbit_cache(cfg, cache)
    kerr = cache.kerr
    constants = cache.constants
    r_turn = cache.r_turn
    pieces = cache.pieces
    trajectories = clipped_trajectories(cache, cfg)
    if cache.orbit_kind === :plunge &&
       effective_asymptotic_tail_correction(cfg)
        trajectories = PieceTrajectory[
            extend_plunge_trajectory_for_abel_tail(
                cfg, kerr, constants, trajectory,
            )
            for trajectory in trajectories
        ]
    end
    t_origin, phi_origin = trajectory_phase_origin(trajectories, cfg)
    harmonic, mode = scattering_mode(kerr, cfg)
    built = assemble_direct_source(
        cfg,
        kerr,
        constants,
        cache.orbit_kind,
        r_turn,
        pieces,
        trajectories,
        harmonic,
        mode,
        t_origin,
        phi_origin,
    )
    if cache.orbit_kind === :plunge
        return merge(built, (
            r_maximum=scattering_source_r_outer(cfg),
            table_r_maximum=built.r_maximum,
        ))
    end
    return built
end

function build_scattering_source(cfg::EquatorialScatteringConfig)
    cache = build_scattering_orbit_cache(cfg)
    return build_scattering_source_from_cache(cfg, cache)
end

function write_scattering_source_csv(cfg::EquatorialScatteringConfig; built=nothing)
    built === nothing && (built = build_scattering_source(cfg))
    lambda = getproperty(built.harmonic, :lambda)
    rs_min = KerrGeometry.rstar_from_r(cfg.a, built.r_minimum)
    rs_max = KerrGeometry.rstar_from_r(cfg.a, built.r_maximum)
    rs_grid = collect(range(rs_min, rs_max; length=cfg.npoints))
    mkpath(dirname(cfg.out))
    open(cfg.out, "w") do io
        println(io, "rstar,re_source,im_source")
        for rs in rs_grid
            r = KerrGeometry.r_from_rstar(cfg.a, rs)
            reduced = built.reduced_at_r(r)
            value = if cfg.source_kind == "reduced"
                reduced
            else
                eta = NativeSN.eta(NativeSN.SNMode(cfg.a, cfg.m, cfg.omega, lambda), r)
                eta * reduced
            end
            println(io, join((csv_value(rs), csv_value(real(value)), csv_value(imag(value))), ","))
        end
    end
    return (
        path=cfg.out,
        source_kind=cfg.source_kind,
        r_turn=built.r_turn,
        r_minimum=built.r_minimum,
        r_maximum=built.r_maximum,
        rstar_min=rs_min,
        rstar_max=rs_max,
        npoints=cfg.npoints,
        lambda=real(lambda),
    )
end

function write_scattering_diagnostics_csv(path, cfg::EquatorialScatteringConfig; built=nothing)
    isempty(path) && return ""
    built === nothing && (built = build_scattering_source(cfg))
    lambda = getproperty(built.harmonic, :lambda)
    rs_min = KerrGeometry.rstar_from_r(cfg.a, built.r_minimum)
    rs_max = KerrGeometry.rstar_from_r(cfg.a, built.r_maximum)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "rstar,r,re_reduced_source,im_reduced_source,contributors,r_turn,lambda")
        for rs in range(rs_min, rs_max; length=min(cfg.npoints, 101))
            r = KerrGeometry.r_from_rstar(cfg.a, rs)
            value = built.reduced_at_r(r)
            contributors = count(traj -> supports_radius(traj, r), built.trajectories)
            println(io, join((
                csv_value(rs),
                csv_value(r),
                csv_value(real(value)),
                csv_value(imag(value)),
                csv_value(contributors),
                csv_value(built.r_turn),
                csv_value(real(lambda)),
            ), ","))
        end
    end
    return path
end

function write_scattering_outputs(cfg::EquatorialScatteringConfig)
    built = build_scattering_source(cfg)
    source_result = write_scattering_source_csv(cfg; built=built)
    write_scattering_diagnostics_csv(cfg.diagnostics_out, cfg; built=built)
    return source_result
end

end
