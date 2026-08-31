module UnboundOrbitSpectrum

include(joinpath(@__DIR__, "FastReducedSourceWaveform.jl"))
include(joinpath(@__DIR__, "SpheroidalSphericalMixing.jl"))
include(joinpath(@__DIR__, "AnalyticDisplacementMemory.jl"))

using .FastReducedSourceWaveform
using .SpheroidalSphericalMixing
using .AnalyticDisplacementMemory
using Printf

const FR = FastReducedSourceWaveform
const SSM = SpheroidalSphericalMixing
const ADM = AnalyticDisplacementMemory

const SN_MEMORY_SIGN = -1.0

export UnboundSpectrumConfig,
       automatic_frequency_window,
       logarithmic_frequency_grid,
       hybrid_frequency_grid,
       fft_aligned_frequency_grid,
       fft_sparse_frequency_grid,
       pchip_slopes,
       pchip_values,
       pchip_integral,
       asymptotic_velocity_data,
       soft_frequency_term,
       soft_time_term,
       run_unbound_spectrum,
       write_spectrum_outputs

Base.@kwdef mutable struct UnboundSpectrumConfig
    a::Float64 = 0.9
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 4.0
    theta_infinity::Float64 = pi / 2
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    orbit_kind::String = "scattering"
    plunge_anchor_radius::Float64 = NaN

    frequency_count::Int = 200
    omega_min::Float64 = NaN
    omega_max::Float64 = NaN
    low_frequency_ratio::Float64 = 0.01
    high_frequency_ratio_scattering::Float64 = 6.0
    high_frequency_ratio_plunge::Float64 = 5.0
    frequency_grid::String = "logarithmic"
    logarithmic_fraction::Float64 = 0.42
    interpolation_coordinate::String = "logarithmic"
    fft_soft_point_count::Int = 16
    fft_dense_low_bin_count::Int = 8
    fft_bin_stride::Int = 1
    soft_deviation_tolerance::Float64 = 0.05
    high_endpoint_fraction_tolerance::Float64 = 0.01

    mode_policy::String = "dominant"
    explicit_modes::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[]
    dominant_l::Int = 2
    dominant_m::Int = 2
    auto_l_max::Int = 8
    angular_energy_tolerance::Float64 = 1e-3
    angular_consecutive_shells::Int = 2
    spheroidal_buffer::Int = 2
    spheroidal_l_max::Int = 10
    mixing_tolerance::Float64 = 2e-4

    r_outer_floor::Float64 = 400.0
    r_outer_cap::Float64 = 2.0e6
    orbit_reference_r_outer::Float64 = NaN
    impact_outer_factor::Float64 = 2.5
    asymptotic_match_phase::Float64 = 200.0
    nsteps_per_branch::Int = 32000
    scale_nsteps_with_outer_radius::Bool = false
    radial_resolution_reference_phase::Float64 = 0.0
    asymptotic_tail_correction::Bool = true
    source_tail_order::Int = 3
    green_tail_correction::Bool = true
    scattering_green_tail_slow_order::Int = 3
    scattering_green_tail_fast_order::Int = 2
    source_grid::String = "table"
    source_npoints::Int = 12001
    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_tolerance::Float64 = 1e-12
    integration_rule::String = "trapezoid"

    interpolation_count::Int = 601
    integrate_soft_segment::Bool = true
    inverse_fft::Bool = false
    fft_points::Int = 1024
    time_domain_memory_baseline::String = "symmetric"
    low_frequency_regular_completion::String = "hold"
    low_frequency_completion_tolerance::Float64 = 0.02
    progress_callback::Union{Nothing, Function} = nothing
    output_dir::String = joinpath(@__DIR__, "..", "output", "unbound_spectrum")
end

function validate_config(cfg::UnboundSpectrumConfig)
    cfg.energy > 1 || error("energy must be greater than one")
    cfg.orbit_kind in ("scattering", "plunge") ||
        error("orbit_kind must be scattering or plunge")
    if !isnan(cfg.plunge_anchor_radius)
        cfg.orbit_kind == "plunge" ||
            error("plunge_anchor_radius is available only for plunge orbits")
        cfg.plunge_anchor_radius > 0 ||
            error("plunge_anchor_radius must be positive")
    end
    cfg.frequency_grid in ("logarithmic", "hybrid", "fft_aligned", "fft_sparse") ||
        error("frequency_grid must be logarithmic, hybrid, fft_aligned, or fft_sparse")
    cfg.interpolation_coordinate in ("linear", "logarithmic") ||
        error("interpolation_coordinate must be linear or logarithmic")
    cfg.frequency_grid in ("fft_aligned", "fft_sparse") && !cfg.inverse_fft &&
        error("FFT-structured frequency grids require inverse_fft=true")
    cfg.frequency_grid == "hybrid" && cfg.frequency_count < 8 &&
        error("frequency_count must be at least 8")
    cfg.fft_soft_point_count >= 4 ||
        error("fft_soft_point_count must be at least four")
    cfg.fft_dense_low_bin_count >= 1 ||
        error("fft_dense_low_bin_count must be positive")
    cfg.fft_bin_stride >= 1 || error("fft_bin_stride must be positive")
    cfg.mode_policy in ("dominant", "explicit", "auto") ||
        error("mode_policy must be dominant, explicit, or auto")
    cfg.mode_policy == "explicit" && isempty(cfg.explicit_modes) &&
        error("explicit mode policy requires at least one mode")
    0 < cfg.soft_deviation_tolerance < 1 ||
        error("soft_deviation_tolerance must lie between zero and one")
    0 < cfg.high_endpoint_fraction_tolerance < 1 ||
        error("high_endpoint_fraction_tolerance must lie between zero and one")
    cfg.angular_energy_tolerance > 0 ||
        error("angular_energy_tolerance must be positive")
    cfg.angular_consecutive_shells >= 1 ||
        error("angular_consecutive_shells must be positive")
    cfg.spheroidal_buffer >= 1 || error("spheroidal_buffer must be positive")
    cfg.spheroidal_l_max >= 2 || error("spheroidal_l_max must be at least 2")
    cfg.inverse_fft && !ispow2(cfg.fft_points) &&
        error("fft_points must be a power of two")
    cfg.inverse_fft && cfg.fft_points < 64 &&
        error("fft_points must be at least 64")
    cfg.fft_dense_low_bin_count <= div(cfg.fft_points, 2) ||
        error("fft_dense_low_bin_count exceeds the positive FFT-bin count")
    minimum_interpolation_count = if cfg.frequency_grid == "fft_aligned"
        div(cfg.fft_points, 2) + cfg.fft_soft_point_count
    elseif cfg.frequency_grid == "fft_sparse"
        fft_sparse_frequency_node_count(
            cfg.fft_points, cfg.fft_dense_low_bin_count, cfg.fft_bin_stride,
        ) + cfg.fft_soft_point_count
    else
        cfg.frequency_count
    end
    cfg.interpolation_count >= minimum_interpolation_count ||
        error("interpolation_count is smaller than the number of solved frequency nodes")
    cfg.low_frequency_regular_completion in
        ("hold", "taper_to_zero", "linear_extrapolate") ||
        error("low_frequency_regular_completion must be hold, taper_to_zero, or linear_extrapolate")
    cfg.time_domain_memory_baseline in
        ("symmetric", "incoming_zero", "outgoing_zero") ||
        error("time_domain_memory_baseline must be symmetric, incoming_zero, or outgoing_zero")
    cfg.low_frequency_completion_tolerance > 0 ||
        error("low_frequency_completion_tolerance must be positive")
    cfg.r_outer_floor > 0 || error("r_outer_floor must be positive")
    cfg.r_outer_cap >= cfg.r_outer_floor ||
        error("r_outer_cap must not be smaller than r_outer_floor")
    if !isnan(cfg.orbit_reference_r_outer)
        isfinite(cfg.orbit_reference_r_outer) &&
            cfg.orbit_reference_r_outer >= cfg.r_outer_floor ||
            error("orbit_reference_r_outer must be finite and not smaller than r_outer_floor")
    end
    cfg.asymptotic_match_phase > 0 ||
        error("asymptotic_match_phase must be positive")
    cfg.nsteps_per_branch >= 10 ||
        error("nsteps_per_branch must be at least ten")
    cfg.radial_resolution_reference_phase >= 0 ||
        error("radial_resolution_reference_phase must be nonnegative")
    cfg.source_npoints >= 5 ||
        error("source_npoints must be at least five")
    return cfg
end

function base_waveform_config(cfg::UnboundSpectrumConfig;
                              ell::Int=cfg.dominant_l,
                              m::Int=cfg.dominant_m,
                              omega::Float64=0.1,
                              r_outer::Float64=cfg.r_outer_floor,
                              nsteps::Int=cfg.nsteps_per_branch)
    return FR.FastReducedWaveformConfig(
        a=cfg.a,
        ell=ell,
        m=m,
        omega=omega,
        energy=cfg.energy,
        lz=cfg.lz,
        carter_q=cfg.carter_q,
        theta_infinity=cfg.theta_infinity,
        phi_infinity=cfg.phi_infinity,
        theta_sign=cfg.theta_sign,
        orbit_kind=cfg.orbit_kind,
        r_outer_min=r_outer,
        r_outer_floor=cfg.r_outer_floor,
        nsteps_per_branch=nsteps,
        asymptotic_tail_correction=cfg.asymptotic_tail_correction,
        asymptotic_match_phase=cfg.asymptotic_match_phase,
        source_tail_order=cfg.source_tail_order,
        green_tail_correction=cfg.green_tail_correction,
        scattering_green_tail_slow_order=cfg.scattering_green_tail_slow_order,
        scattering_green_tail_fast_order=cfg.scattering_green_tail_fast_order,
        allow_theta_turns=true,
        source_grid=cfg.source_grid,
        npoints=cfg.source_npoints,
        homogeneous_rsin=cfg.homogeneous_rsin,
        homogeneous_rsout_min=cfg.homogeneous_rsout_min,
        homogeneous_tolerance=cfg.homogeneous_tolerance,
        integration_rule=cfg.integration_rule,
    )
end

function asymptotic_impact_scale(cfg::UnboundSpectrumConfig)
    momentum = sqrt(cfg.energy^2 - 1)
    return sqrt(cfg.lz^2 + max(cfg.carter_q, 0.0)) / momentum
end

function probe_outer_radius(cfg::UnboundSpectrumConfig)
    return max(cfg.r_outer_floor,
               cfg.impact_outer_factor * asymptotic_impact_scale(cfg))
end

function periastron_point(cache)
    points = Iterators.flatten(trajectory.points for trajectory in cache.trajectories)
    return reduce((left, right) -> left.r <= right.r ? left : right, points)
end

function characteristic_frequency(cfg::UnboundSpectrumConfig, cache)
    point = periastron_point(cache)
    omega_phi = abs(point.uphi / point.ut)
    omega_theta = abs(point.utheta / point.ut)
    orbital = max(
        max(2, abs(cfg.dominant_m)) * omega_phi,
        2omega_theta,
        2 / max(point.r, 1.0)^1.5,
    )
    if cfg.orbit_kind == "plunge"
        orbital = max(orbital, 0.2)
    end
    return max(orbital, 1e-5)
end

function automatic_frequency_window(cfg::UnboundSpectrumConfig, cache)
    scale = characteristic_frequency(cfg, cache)
    automatic_low = max(
        cfg.low_frequency_ratio * scale,
        cfg.asymptotic_match_phase / cfg.r_outer_cap,
    )
    omega_min = isfinite(cfg.omega_min) ? cfg.omega_min : automatic_low
    high_ratio = cfg.orbit_kind == "scattering" ?
                 cfg.high_frequency_ratio_scattering :
                 cfg.high_frequency_ratio_plunge
    omega_max = isfinite(cfg.omega_max) ? cfg.omega_max : high_ratio * scale
    omega_min > 0 || error("omega_min must be positive")
    omega_max > omega_min || error(
        "automatic frequency window is empty; increase r_outer_cap, " *
        "reduce asymptotic_match_phase, or specify omega_min and omega_max",
    )
    return (omega_min=omega_min, omega_max=omega_max, scale=scale)
end

function hybrid_frequency_grid(omega_min::Real,
                               omega_max::Real,
                               count::Int;
                               scale::Real=sqrt(omega_min * omega_max),
                               logarithmic_fraction::Real=0.42)
    count >= 8 || error("frequency grid needs at least eight points")
    0 < omega_min < omega_max || error("invalid positive frequency interval")
    nlog = clamp(round(Int, logarithmic_fraction * count), 4, count - 4)
    split = clamp(0.55 * Float64(scale), 8Float64(omega_min),
                  0.55Float64(omega_max))
    if !(omega_min < split < omega_max)
        return collect(exp.(range(log(omega_min), log(omega_max); length=count)))
    end
    low = collect(exp.(range(log(omega_min), log(split); length=nlog)))
    high = collect(range(split, omega_max; length=count - nlog + 1))[2:end]
    return vcat(low, high)
end

function logarithmic_frequency_grid(omega_min::Real,
                                    omega_max::Real,
                                    count::Int)
    count >= 2 || error("logarithmic frequency grid needs at least two points")
    0 < omega_min < omega_max || error("invalid positive frequency interval")
    return collect(exp.(range(log(Float64(omega_min)), log(Float64(omega_max));
                              length=count)))
end

function fft_aligned_frequency_grid(omega_min::Real,
                                    omega_max::Real,
                                    fft_points::Int,
                                    soft_point_count::Int)
    ispow2(fft_points) || error("fft_points must be a power of two")
    fft_points >= 64 || error("fft_points must be at least 64")
    soft_point_count >= 4 || error("soft_point_count must be at least four")
    0 < omega_min < omega_max || error("invalid positive frequency interval")
    domega = 2Float64(omega_max) / fft_points
    soft_max = 0.9domega
    omega_min < soft_max || error(
        "omega_min must lie below 0.9 times the first positive IFFT bin",
    )
    soft = collect(exp.(range(log(Float64(omega_min)), log(soft_max);
                               length=soft_point_count)))
    fft_nodes = domega .* collect(1:div(fft_points, 2))
    return vcat(soft, fft_nodes)
end

function fft_sparse_frequency_node_count(fft_points::Int,
                                         dense_low_bin_count::Int,
                                         bin_stride::Int)
    maximum_bin = div(fft_points, 2)
    1 <= dense_low_bin_count <= maximum_bin ||
        error("invalid dense low-bin count")
    bin_stride >= 1 || error("bin stride must be positive")
    bins = collect(1:dense_low_bin_count)
    append!(bins, collect((dense_low_bin_count + 1):bin_stride:maximum_bin))
    last(bins) == maximum_bin || push!(bins, maximum_bin)
    return length(unique(bins))
end

function fft_sparse_frequency_grid(omega_min::Real,
                                   omega_max::Real,
                                   fft_points::Int,
                                   soft_point_count::Int,
                                   dense_low_bin_count::Int,
                                   bin_stride::Int)
    direct = fft_aligned_frequency_grid(
        omega_min, omega_max, fft_points, soft_point_count,
    )
    soft = direct[1:soft_point_count]
    domega = 2Float64(omega_max) / fft_points
    maximum_bin = div(fft_points, 2)
    1 <= dense_low_bin_count <= maximum_bin ||
        error("invalid dense low-bin count")
    bin_stride >= 1 || error("bin stride must be positive")
    bins = collect(1:dense_low_bin_count)
    append!(bins, collect((dense_low_bin_count + 1):bin_stride:maximum_bin))
    last(bins) == maximum_bin || push!(bins, maximum_bin)
    return vcat(soft, domega .* unique(bins))
end

unit_radial(theta, phi) =
    (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))

function rotate_xy(vector, angle)
    cosine = cos(angle)
    sine = sin(angle)
    return (
        cosine * vector[1] - sine * vector[2],
        sine * vector[1] + cosine * vector[2],
        vector[3],
    )
end

function asymptotic_velocity_data(cfg::UnboundSpectrumConfig, cache)
    incoming_point = first(cache.trajectories).points[1]
    incoming_direction = (
        theta=cfg.theta_infinity,
        phi=cfg.phi_infinity,
        direction=unit_radial(cfg.theta_infinity, cfg.phi_infinity),
    )
    beta = sqrt(cfg.energy^2 - 1) / cfg.energy
    incoming_raw = Tuple(-beta * value for value in incoming_direction.direction)
    if cfg.orbit_kind == "plunge"
        return (v_in=incoming_raw, v_out=(0.0, 0.0, 0.0), beta=beta,
                phi_origin=0.0, incoming=incoming_direction, outgoing=nothing)
    end
    outgoing_point = last(cache.trajectories).points[end]
    outgoing_tail = FR.EquatorialScatteringSource.UG.asymptote_from_finite_outer_state(
        cache.kerr,
        cache.constants,
        outgoing_point.r,
        outgoing_point.theta,
        outgoing_point.phi,
        outgoing_point.ur_sign,
        outgoing_point.utheta_sign,
    )
    outgoing_direction = (
        theta=outgoing_tail.theta,
        phi=outgoing_tail.phi,
        direction=unit_radial(outgoing_tail.theta, outgoing_tail.phi),
    )
    outgoing_raw = Tuple(beta * value for value in outgoing_direction.direction)
    phi_origin = periastron_point(cache).phi
    incoming = rotate_xy(incoming_raw, -phi_origin)
    outgoing = rotate_xy(outgoing_raw, -phi_origin)
    return (v_in=incoming, v_out=outgoing, beta=beta, phi_origin=phi_origin,
            incoming=incoming_direction, outgoing=outgoing_direction)
end

function plunge_anchor_event(cfg::UnboundSpectrumConfig, cache)
    if isnan(cfg.plunge_anchor_radius)
        return (radius=NaN, time=0.0, phi=0.0)
    end
    cfg.orbit_kind == "plunge" ||
        error("a fixed plunge anchor was requested for a non-plunge orbit")
    length(cache.trajectories) == 1 ||
        error("a plunge anchor requires exactly one incoming trajectory")
    trajectory = only(cache.trajectories)
    FR.UG.supports_radius(trajectory, cfg.plunge_anchor_radius) ||
        error("plunge_anchor_radius=$(cfg.plunge_anchor_radius) is outside the cached worldline")
    point = FR.UG.interpolated_point(trajectory, cfg.plunge_anchor_radius)
    return (radius=cfg.plunge_anchor_radius, time=point.t, phi=point.phi)
end

function anchored_velocity_data(cfg::UnboundSpectrumConfig, cache, anchor)
    velocities = asymptotic_velocity_data(cfg, cache)
    isnan(anchor.radius) && return velocities
    incoming = rotate_xy(velocities.v_in, -anchor.phi)
    outgoing = rotate_xy(velocities.v_out, -anchor.phi)
    return merge(velocities, (v_in=incoming, v_out=outgoing,
                              phi_origin=velocities.phi_origin + anchor.phi))
end

function anchor_spherical_mode(mode, signed_m::Int, signed_omega::Float64,
                               anchor)
    isnan(anchor.radius) && return mode
    phase = cis(-signed_omega * anchor.time + signed_m * anchor.phi)
    return merge(mode, (value=ComplexF64(mode.value * phase),))
end

function memory_pair(cfg::UnboundSpectrumConfig, velocities, ell, m)
    positive = ADM.displacement_memory_mode(
        ell, m, velocities.v_in, velocities.v_out;
        particle_energy=cfg.energy,
    )
    conjugate = ADM.displacement_memory_mode(
        ell, -m, velocities.v_in, velocities.v_out;
        particle_energy=cfg.energy,
    )
    pair_power = abs2(positive) + abs2(conjugate)
    zfl = pair_power / (32pi^2 * cfg.energy^2)
    return (positive=positive, conjugate=conjugate,
            pair_power=pair_power, zfl=zfl)
end

function soft_frequency_term(delta_h::Complex, omega::Real)
    omega == 0 && error("the displacement-memory frequency pole is undefined at omega=0")
    return ComplexF64(SN_MEMORY_SIGN * 1im * delta_h / (2pi * omega))
end

function soft_time_term(delta_h::Complex, time::Real)
    time == 0 && return 0.0 + 0.0im
    return ComplexF64(SN_MEMORY_SIGN * sign(time) * delta_h / 2)
end

function memory_baseline_offset(delta_h::Complex, baseline::String)
    baseline == "symmetric" && return 0.0 + 0.0im
    half_jump = ComplexF64(SN_MEMORY_SIGN * delta_h / 2)
    baseline == "incoming_zero" && return half_jump
    baseline == "outgoing_zero" && return -half_jump
    error("unknown time-domain memory baseline: $baseline")
end

const ScatteringOrbitCache =
    FR.EquatorialScatteringSource.ScatteringOrbitCache

mutable struct ModeComputationCache
    orbit::ScatteringOrbitCache
    frequency_orbits::Dict{Float64, ScatteringOrbitCache}
    values::Dict{Tuple{Int, Int, Float64}, ComplexF64}
    metadata::Dict{Tuple{Int, Int, Float64}, Any}
    mixing_expansions::Dict{Tuple{Int, Int, Float64}, Any}
end

ModeComputationCache(orbit) = ModeComputationCache(
    orbit,
    Dict{Float64, ScatteringOrbitCache}(Float64(orbit.r_outer) => orbit),
    Dict{Tuple{Int, Int, Float64}, ComplexF64}(),
    Dict{Tuple{Int, Int, Float64}, Any}(),
    Dict{Tuple{Int, Int, Float64}, Any}(),
)

function frequency_outer_radius(cfg::UnboundSpectrumConfig,
                                 maximum_outer::Float64,
                                 omega::Float64)
    waveform_cfg = base_waveform_config(
        cfg; omega=omega, r_outer=maximum_outer,
    )
    return FR.effective_source_r_outer(waveform_cfg)
end

function frequency_radial_steps(cfg::UnboundSpectrumConfig,
                                r_outer::Float64,
                                omega::Float64)
    if cfg.radial_resolution_reference_phase > 0
        return ceil(Int, cfg.nsteps_per_branch * max(
            1.0,
            r_outer * abs(omega) / cfg.radial_resolution_reference_phase,
        ))
    end
    cfg.scale_nsteps_with_outer_radius || return cfg.nsteps_per_branch
    return ceil(Int, cfg.nsteps_per_branch * max(1.0, r_outer / cfg.r_outer_floor))
end

function frequency_orbit!(cache::ModeComputationCache,
                          cfg::UnboundSpectrumConfig,
                          omega::Float64)
    cfg.orbit_kind == "plunge" && return cache.orbit
    target_outer = frequency_outer_radius(
        cfg, Float64(cache.orbit.r_outer), omega,
    )
    target_outer >= cache.orbit.r_outer * (1 - 1e-12) && return cache.orbit
    return get!(cache.frequency_orbits, target_outer) do
        build_frequency_orbit(cache, cfg, omega, target_outer)
    end
end

function build_frequency_orbit(cache::ModeComputationCache,
                               cfg::UnboundSpectrumConfig,
                               omega::Float64,
                               target_outer::Float64)
    waveform_cfg = base_waveform_config(
        cfg;
        omega=omega,
        r_outer=target_outer,
        nsteps=frequency_radial_steps(cfg, target_outer, omega),
    )
    return FR.build_fast_reduced_orbit_cache(waveform_cfg)
end

function spheroidal_mode!(cache::ModeComputationCache,
                          cfg::UnboundSpectrumConfig,
                          ell::Int,
                          m::Int,
                          omega::Float64)
    key = (ell, m, omega)
    return get!(cache.values, key) do
        orbit = frequency_orbit!(cache, cfg, omega)
        waveform_cfg = base_waveform_config(
            cfg;
            ell=ell,
            m=m,
            omega=omega,
            r_outer=orbit.r_outer,
            nsteps=frequency_radial_steps(cfg, orbit.r_outer, omega),
        )
        result = FR.compute_fast_reduced_waveform(
            waveform_cfg; orbit_cache=orbit,
        )
        cache.metadata[key] = FR.fast_reduced_waveform_row(result)
        ComplexF64(8 * result.green.xinf_over_c0)
    end
end

function spherical_mode!(cache::ModeComputationCache,
                         cfg::UnboundSpectrumConfig,
                         spherical_l::Int,
                         m::Int,
                         omega::Float64)
    ell_min = max(2, abs(m))
    if cfg.a == 0.0
        value = spheroidal_mode!(cache, cfg, spherical_l, m, omega)
        return (value=value, spheroidal_l_max=spherical_l,
                mixing_converged=true)
    end
    start_max = max(spherical_l + cfg.spheroidal_buffer, ell_min)
    cfg.spheroidal_l_max >= start_max ||
        error("spheroidal_l_max=$(cfg.spheroidal_l_max) is below the required minimum $start_max")
    hard_max = cfg.spheroidal_l_max
    value = 0.0 + 0.0im
    converged = false
    used_max = ell_min
    for ell in ell_min:hard_max
        amplitude = spheroidal_mode!(cache, cfg, ell, m, omega)
        expansion = get!(cache.mixing_expansions, (ell, m, omega)) do
            SSM.spheroidal_expansion_coefficients(
                -2, ell, m, cfg.a * omega; coefficient_cutoff=0.0,
            )
        end
        coefficient = 0.0 + 0.0im
        for row in expansion
            row.spherical_l == spherical_l || continue
            coefficient += row.coefficient
        end
        contribution = coefficient * amplitude
        value += contribution
        used_max = ell
        ell < start_max && continue
        relative = abs(contribution) / max(abs(value), eps(Float64))
        if relative <= cfg.mixing_tolerance
            converged = true
            break
        end
    end
    return (value=ComplexF64(value), spheroidal_l_max=used_max,
            mixing_converged=converged)
end

function pchip_slopes(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || error("x and y lengths differ")
    n >= 2 || error("PCHIP requires at least two points")
    h = diff(Float64.(x))
    all(>(0), h) || error("PCHIP x values must be strictly increasing")
    delta = diff(Float64.(y)) ./ h
    n == 2 && return [delta[1], delta[1]]
    slopes = zeros(Float64, n)
    for i in 2:(n - 1)
        if delta[i - 1] * delta[i] <= 0
            slopes[i] = 0.0
        else
            w1 = 2h[i] + h[i - 1]
            w2 = h[i] + 2h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        end
    end
    slopes[1] = ((2h[1] + h[2]) * delta[1] - h[1] * delta[2]) /
                (h[1] + h[2])
    sign(slopes[1]) == sign(delta[1]) || (slopes[1] = 0.0)
    abs(slopes[1]) <= 3abs(delta[1]) || (slopes[1] = 3delta[1])
    slopes[end] = ((2h[end] + h[end - 1]) * delta[end] -
                   h[end] * delta[end - 1]) / (h[end] + h[end - 1])
    sign(slopes[end]) == sign(delta[end]) || (slopes[end] = 0.0)
    abs(slopes[end]) <= 3abs(delta[end]) || (slopes[end] = 3delta[end])
    return slopes
end

function pchip_value(x, y, slopes, query)
    query <= x[1] && return Float64(y[1])
    query >= x[end] && return Float64(y[end])
    index = searchsortedlast(x, query)
    index = min(index, length(x) - 1)
    width = x[index + 1] - x[index]
    t = (query - x[index]) / width
    h00 = 2t^3 - 3t^2 + 1
    h10 = t^3 - 2t^2 + t
    h01 = -2t^3 + 3t^2
    h11 = t^3 - t^2
    return h00 * y[index] + h10 * width * slopes[index] +
           h01 * y[index + 1] + h11 * width * slopes[index + 1]
end

function pchip_values(x, y, queries)
    slopes = pchip_slopes(x, y)
    return Float64[pchip_value(x, y, slopes, query) for query in queries]
end

function pchip_integral(x, y)
    slopes = pchip_slopes(x, y)
    total = 0.0
    for i in 1:(length(x) - 1)
        width = x[i + 1] - x[i]
        total += width * ((y[i] + y[i + 1]) / 2 +
                          width * (slopes[i] - slopes[i + 1]) / 12)
    end
    return total
end

function unwrap_phase(values)
    phases = angle.(values)
    for i in 2:length(phases)
        jump = phases[i] - phases[i - 1]
        phases[i] -= 2pi * round(jump / (2pi))
    end
    return phases
end

function interpolation_abscissa(values, coordinate::String)
    coordinate == "linear" && return Float64.(values)
    coordinate == "logarithmic" ||
        error("interpolation coordinate must be linear or logarithmic")
    all(value -> value > 0, values) ||
        error("logarithmic interpolation requires positive frequencies")
    return log.(Float64.(values))
end

function interpolate_complex(x, values, queries; coordinate::String="linear")
    amplitudes = abs.(values)
    floor_amplitude = max(maximum(amplitudes) * 1e-300, floatmin(Float64))
    log_amplitudes = log.(max.(amplitudes, floor_amplitude))
    phases = unwrap_phase(values)
    interpolation_x = interpolation_abscissa(x, coordinate)
    interpolation_queries = interpolation_abscissa(queries, coordinate)
    new_amplitudes = exp.(pchip_values(interpolation_x, log_amplitudes,
                                        interpolation_queries))
    new_phases = pchip_values(interpolation_x, phases, interpolation_queries)
    return ComplexF64[new_amplitudes[i] * cis(new_phases[i])
                      for i in eachindex(queries)]
end

function mode_spectrum!(cache::ModeComputationCache,
                        cfg::UnboundSpectrumConfig,
                        frequencies,
                        velocities,
                        ell,
                        m)
    memory = memory_pair(cfg, velocities, ell, m)
    rows = NamedTuple[]
    for omega in frequencies
        push!(rows, mode_frequency_row!(
            cache, cfg, ell, m, Float64(omega), memory,
        ))
    end
    return rows, memory
end

function mode_frequency_row!(cache::ModeComputationCache,
                             cfg::UnboundSpectrumConfig,
                             ell::Int,
                             m::Int,
                             omega::Float64,
                             memory)
    positive = spherical_mode!(cache, cfg, ell, m, omega)
    conjugate = spherical_mode!(cache, cfg, ell, -m, -omega)
    return mode_frequency_row_from_modes(
        cfg, ell, m, omega, memory, positive, conjugate,
    )
end

function mode_frequency_row_from_modes(cfg::UnboundSpectrumConfig,
                                       ell::Int,
                                       m::Int,
                                       omega::Float64,
                                       memory,
                                       positive,
                                       conjugate)
    energy = omega^2 * (abs2(positive.value) + abs2(conjugate.value)) /
             (8cfg.energy^2)
    pos_soft = abs(memory.positive) == 0 ? NaN + NaN * im :
               2pi * omega * positive.value /
               (SN_MEMORY_SIGN * 1im * memory.positive)
    neg_soft = abs(memory.conjugate) == 0 ? NaN + NaN * im :
               2pi * (-omega) * conjugate.value /
               (SN_MEMORY_SIGN * 1im * memory.conjugate)
    return (
        spherical_l=ell,
        m=m,
        omega=omega,
        h_positive=positive.value,
        h_conjugate=conjugate.value,
        pair_dE_domega_over_mu2E2=Float64(energy),
        zfl_dE_domega_over_mu2E2=memory.zfl,
        energy_relative_deviation=memory.zfl > 0 ? energy / memory.zfl - 1 : NaN,
        positive_soft_ratio=ComplexF64(pos_soft),
        conjugate_soft_ratio=ComplexF64(neg_soft),
        positive_spheroidal_l_max=positive.spheroidal_l_max,
        conjugate_spheroidal_l_max=conjugate.spheroidal_l_max,
        mixing_converged=positive.mixing_converged && conjugate.mixing_converged,
    )
end

function integrated_mode_energy(cfg, rows)
    x = getproperty.(rows, :omega)
    y = getproperty.(rows, :pair_dE_domega_over_mu2E2)
    resolved = pchip_integral(x, y)
    soft = cfg.integrate_soft_segment ?
           rows[1].zfl_dE_domega_over_mu2E2 * x[1] : 0.0
    return (resolved=resolved, soft_segment=soft, total=resolved + soft)
end

function mode_summary(cfg, ell, m, frequencies, rows, energy)
    peak = max(maximum(getproperty.(rows, :pair_dE_domega_over_mu2E2)),
               eps(Float64))
    high_fraction = rows[end].pair_dE_domega_over_mu2E2 / peak
    return (
        spherical_l=ell,
        m=m,
        resolved_energy_over_mu2E2=energy.resolved,
        soft_segment_energy_over_mu2E2=energy.soft_segment,
        total_energy_over_mu2E2=energy.total,
        lowest_frequency=frequencies[1],
        lowest_energy_relative_deviation=rows[1].energy_relative_deviation,
        soft_limit_pass=abs(rows[1].energy_relative_deviation) <=
                        cfg.soft_deviation_tolerance,
        highest_frequency=frequencies[end],
        high_endpoint_over_peak=high_fraction,
        high_endpoint_pass=high_fraction <=
                           cfg.high_endpoint_fraction_tolerance,
        mixing_converged=all(getproperty.(rows, :mixing_converged)),
    )
end

function prewarm_frequency_orbits!(cache::ModeComputationCache,
                                   cfg::UnboundSpectrumConfig,
                                   frequencies)
    cfg.orbit_kind == "plunge" && return cache
    representative_omega = Dict{Float64, Float64}()
    for omega in frequencies, signed_omega in (Float64(omega), -Float64(omega))
        target = frequency_outer_radius(
            cfg, Float64(cache.orbit.r_outer), signed_omega,
        )
        target >= cache.orbit.r_outer * (1 - 1e-12) && continue
        get!(representative_omega, target, signed_omega)
    end
    targets = sort(collect(keys(representative_omega)); rev=true)
    built = Vector{Any}(undef, length(targets))
    if Threads.nthreads() > 1 && length(targets) > 1
        Threads.@threads :dynamic for index in eachindex(targets)
            target = targets[index]
            built[index] = build_frequency_orbit(
                cache, cfg, representative_omega[target], target,
            )
        end
    else
        for index in eachindex(targets)
            target = targets[index]
            built[index] = build_frequency_orbit(
                cache, cfg, representative_omega[target], target,
            )
        end
    end
    for index in eachindex(targets)
        cache.frequency_orbits[targets[index]] = built[index]
    end
    return cache
end

function mode_worker_cache(cache::ModeComputationCache)
    return ModeComputationCache(
        cache.orbit,
        copy(cache.frequency_orbits),
        Dict{Tuple{Int, Int, Float64}, ComplexF64}(),
        Dict{Tuple{Int, Int, Float64}, Any}(),
        Dict{Tuple{Int, Int, Float64}, Any}(),
    )
end

function merge_mode_cache!(target::ModeComputationCache,
                           source::ModeComputationCache)
    merge!(target.frequency_orbits, source.frequency_orbits)
    merge!(target.values, source.values)
    merge!(target.metadata, source.metadata)
    merge!(target.mixing_expansions, source.mixing_expansions)
    return target
end

function group_modes_by_m(modes)
    grouped = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (ell, m) in modes
        push!(get!(grouped, m, Tuple{Int, Int}[]), (ell, m))
    end
    return [(m=m, modes=grouped[m]) for m in sort(collect(keys(grouped)))]
end

function compute_mode_batch!(worker_caches::Dict{Tuple{Int, Int}, ModeComputationCache},
                              base_cache::ModeComputationCache,
                              cfg::UnboundSpectrumConfig,
                              frequencies,
                              velocities,
                              anchor,
                              modes)
    groups = group_modes_by_m(modes)
    for group in groups, frequency_index in eachindex(frequencies), branch in 1:2
        key = branch == 1 ? (group.m, frequency_index) :
                            (-group.m, -frequency_index)
        haskey(worker_caches, key) ||
            (worker_caches[key] = mode_worker_cache(base_cache))
    end
    memories = [
        [memory_pair(cfg, velocities, ell, m) for (ell, m) in group.modes]
        for group in groups
    ]
    positive_by_group = [
        [Vector{Any}(undef, length(frequencies)) for _ in group.modes]
        for group in groups
    ]
    conjugate_by_group = [
        [Vector{Any}(undef, length(frequencies)) for _ in group.modes]
        for group in groups
    ]
    jobs = [(group_index, frequency_index, branch)
            for group_index in eachindex(groups)
            for frequency_index in eachindex(frequencies)
            for branch in 1:2]
    parallel = Threads.nthreads() > 1 && length(jobs) > 1
    completed_jobs = Ref(0)
    progress_lock = ReentrantLock()

    function report_progress!(signed_m, signed_omega, mode_count, ell_min, ell_max)
        callback = cfg.progress_callback
        isnothing(callback) && return nothing
        lock(progress_lock) do
            completed_jobs[] += 1
            callback(completed_jobs[], length(jobs), (
                m=signed_m,
                omega=signed_omega,
                mode_count=mode_count,
                ell_min=ell_min,
                ell_max=ell_max,
            ))
        end
        return nothing
    end

    function compute_job!(job_index)
        group_index, frequency_index, branch = jobs[job_index]
        group = groups[group_index]
        omega = Float64(frequencies[frequency_index])
        signed_m = branch == 1 ? group.m : -group.m
        signed_omega = branch == 1 ? omega : -omega
        cache_key = branch == 1 ? (group.m, frequency_index) :
                                  (-group.m, -frequency_index)
        local_cache = worker_caches[cache_key]
        output = branch == 1 ? positive_by_group : conjugate_by_group
        for mode_index in eachindex(group.modes)
            ell, _ = group.modes[mode_index]
            raw = spherical_mode!(
                local_cache, cfg, ell, signed_m, signed_omega,
            )
            output[group_index][mode_index][frequency_index] =
                anchor_spherical_mode(raw, signed_m, signed_omega, anchor)
        end
        report_progress!(signed_m, signed_omega, length(group.modes),
                         minimum(first.(group.modes)),
                         maximum(first.(group.modes)))
    end

    if parallel
        Threads.@threads :dynamic for job_index in eachindex(jobs)
            compute_job!(job_index)
        end
    else
        for job_index in eachindex(jobs)
            compute_job!(job_index)
        end
    end

    records = Any[]
    for group_index in eachindex(groups)
        group = groups[group_index]
        for mode_index in eachindex(group.modes)
            ell, m = group.modes[mode_index]
            memory = memories[group_index][mode_index]
            rows = NamedTuple[
                mode_frequency_row_from_modes(
                    cfg, ell, m, Float64(frequencies[frequency_index]), memory,
                    positive_by_group[group_index][mode_index][frequency_index],
                    conjugate_by_group[group_index][mode_index][frequency_index],
                )
                for frequency_index in eachindex(frequencies)
            ]
            energy = integrated_mode_energy(cfg, rows)
            push!(records, (
                key=(ell, m),
                spectrum=(rows=rows, memory=memory, energy=energy),
                summary=mode_summary(
                    cfg, ell, m, frequencies, rows, energy,
                ),
            ))
        end
    end
    return records, parallel
end

function requested_fixed_modes(cfg::UnboundSpectrumConfig)
    modes = if cfg.mode_policy == "dominant"
        [(cfg.dominant_l, cfg.dominant_m)]
    elseif cfg.mode_policy == "explicit"
        copy(cfg.explicit_modes)
    else
        Tuple{Int, Int}[]
    end
    for (ell, m) in modes
        ell >= 2 || error("requested ell must be at least two")
        abs(m) <= ell || error("requested mode must satisfy |m| <= ell")
    end
    return unique(modes)
end

function choose_production_outer(cfg, window)
    phase_outer = cfg.asymptotic_match_phase / window.omega_min
    orbit_outer = probe_outer_radius(cfg)
    phase_limited = min(phase_outer, cfg.r_outer_cap)
    return max(orbit_outer, phase_limited), phase_outer > cfg.r_outer_cap
end

function run_unbound_spectrum(cfg::UnboundSpectrumConfig; write_output::Bool=true)
    run_started = time()
    validate_config(cfg)
    probe_outer = probe_outer_radius(cfg)
    probe_cfg = base_waveform_config(
        cfg;
        omega=0.1,
        r_outer=probe_outer,
        nsteps=min(cfg.nsteps_per_branch, 8000),
    )
    probe_cache = FR.build_fast_reduced_orbit_cache(probe_cfg)
    window = automatic_frequency_window(cfg, probe_cache)
    frequencies = if cfg.frequency_grid == "fft_aligned"
        fft_aligned_frequency_grid(
            window.omega_min, window.omega_max, cfg.fft_points,
            cfg.fft_soft_point_count,
        )
    elseif cfg.frequency_grid == "fft_sparse"
        fft_sparse_frequency_grid(
            window.omega_min, window.omega_max, cfg.fft_points,
            cfg.fft_soft_point_count, cfg.fft_dense_low_bin_count,
            cfg.fft_bin_stride,
        )
    elseif cfg.frequency_grid == "hybrid"
        hybrid_frequency_grid(
            window.omega_min,
            window.omega_max,
            cfg.frequency_count;
            scale=window.scale,
            logarithmic_fraction=cfg.logarithmic_fraction,
        )
    else
        logarithmic_frequency_grid(
            window.omega_min, window.omega_max, cfg.frequency_count,
        )
    end
    production_outer, outer_capped = choose_production_outer(cfg, window)
    orbit_reference_outer = isnan(cfg.orbit_reference_r_outer) ?
        production_outer : max(production_outer, cfg.orbit_reference_r_outer)
    production_cfg = base_waveform_config(
        cfg;
        omega=frequencies[1],
        r_outer=orbit_reference_outer,
        nsteps=frequency_radial_steps(
            cfg, orbit_reference_outer, frequencies[1],
        ),
    )
    orbit = if abs(orbit_reference_outer - probe_outer) <=
               1e-12 * max(orbit_reference_outer, probe_outer) &&
               cfg.nsteps_per_branch <= 8000
        probe_cache
    else
        FR.build_fast_reduced_orbit_cache(production_cfg)
    end
    orbit_ready = time()
    cache = ModeComputationCache(orbit)
    prewarm_frequency_orbits!(cache, cfg, frequencies)
    frequency_orbits_ready = time()
    anchor = plunge_anchor_event(cfg, orbit)
    velocities = anchored_velocity_data(cfg, orbit, anchor)
    spectra = Dict{Tuple{Int, Int}, Any}()
    summaries = NamedTuple[]
    angular_rows = NamedTuple[]
    worker_caches = Dict{Tuple{Int, Int}, ModeComputationCache}()
    parallel_mode_frequency_jobs = false

    function store_records!(records)
        for record in records
            haskey(spectra, record.key) && continue
            spectra[record.key] = record.spectrum
            push!(summaries, record.summary)
        end
    end

    if cfg.mode_policy == "auto"
        cumulative = 0.0
        small_shells = 0
        for ell in 2:cfg.auto_l_max
            shell_modes = [(ell, m) for m in -ell:ell]
            records, did_parallel = compute_mode_batch!(
                worker_caches, cache, cfg, frequencies, velocities, anchor, shell_modes,
            )
            parallel_mode_frequency_jobs |= did_parallel
            store_records!(records)
            shell_energy = sum(record.spectrum.energy.total for record in records)
            cumulative += shell_energy
            fraction = shell_energy / max(cumulative, eps(Float64))
            small_shells = fraction < cfg.angular_energy_tolerance ?
                           small_shells + 1 : 0
            push!(angular_rows, (
                spherical_l=ell,
                shell_energy_over_mu2E2=shell_energy,
                cumulative_energy_over_mu2E2=cumulative,
                shell_fraction=fraction,
                below_tolerance=fraction < cfg.angular_energy_tolerance,
                consecutive_below=small_shells,
            ))
            small_shells >= cfg.angular_consecutive_shells && break
        end
    else
        records, did_parallel = compute_mode_batch!(
            worker_caches, cache, cfg, frequencies, velocities, anchor,
            requested_fixed_modes(cfg),
        )
        parallel_mode_frequency_jobs |= did_parallel
        store_records!(records)
    end

    for local_cache in values(worker_caches)
        merge_mode_cache!(cache, local_cache)
    end
    modes_ready = time()

    interpolated = Dict{Tuple{Int, Int}, Any}()
    dense_frequencies = collect(range(
        frequencies[1], frequencies[end]; length=cfg.interpolation_count,
    ))
    for (key, spectrum) in spectra
        rows = spectrum.rows
        positive = interpolate_complex(
            frequencies, getproperty.(rows, :h_positive), dense_frequencies,
            coordinate=cfg.interpolation_coordinate,
        )
        conjugate = interpolate_complex(
            frequencies, getproperty.(rows, :h_conjugate), dense_frequencies,
            coordinate=cfg.interpolation_coordinate,
        )
        energy = pchip_values(
            interpolation_abscissa(frequencies, cfg.interpolation_coordinate),
            getproperty.(rows, :pair_dE_domega_over_mu2E2),
            interpolation_abscissa(dense_frequencies, cfg.interpolation_coordinate),
        )
        interpolated[key] = (
            frequencies=dense_frequencies,
            positive=positive,
            conjugate=conjugate,
            energy=max.(energy, 0.0),
        )
    end
    interpolation_ready = time()

    total_energy = sum(row.total_energy_over_mu2E2 for row in summaries)
    result = (
        config=cfg,
        window=window,
        frequencies=frequencies,
        production_outer=production_outer,
        orbit_reference_outer=orbit_reference_outer,
        plunge_anchor=anchor,
        outer_capped=outer_capped,
        velocities=velocities,
        spectra=spectra,
        interpolated=interpolated,
        mode_summaries=summaries,
        angular_convergence=angular_rows,
        spheroidal_cache=cache,
        total_energy_over_mu2E2=total_energy,
        julia_threads=Threads.nthreads(),
        parallel_mode_frequency_jobs=parallel_mode_frequency_jobs,
        timings=(
            orbit_setup_seconds=orbit_ready - run_started,
            frequency_orbit_prewarm_seconds=frequency_orbits_ready - orbit_ready,
            mode_solve_seconds=modes_ready - frequency_orbits_ready,
            interpolation_seconds=interpolation_ready - modes_ready,
        ),
    )
    if write_output
        write_spectrum_outputs(cfg.output_dir, result)
        cfg.inverse_fft && write_time_domain_outputs(cfg.output_dir, result)
    end
    return result
end

csv_value(x::AbstractFloat) = isnan(x) ? "NaN" : repr(Float64(x))
csv_value(x::Integer) = string(x)
csv_value(x::Bool) = string(x)
csv_value(x::AbstractString) = x

function write_rows(path, columns, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(String.(columns), ","))
        for row in rows
            println(io, join((csv_value(getproperty(row, column))
                              for column in columns), ","))
        end
    end
    return path
end

function write_spectrum_outputs(output_dir, result)
    mkpath(output_dir)
    mode_rows = NamedTuple[]
    soft_rows = NamedTuple[]
    for ((ell, m), spectrum) in sort(collect(result.spectra); by=first)
        for row in spectrum.rows
            push!(mode_rows, (
                spherical_l=ell, m=m, omega=row.omega,
                re_h_positive=real(row.h_positive),
                im_h_positive=imag(row.h_positive),
                re_h_conjugate=real(row.h_conjugate),
                im_h_conjugate=imag(row.h_conjugate),
                pair_dE_domega_over_mu2E2=row.pair_dE_domega_over_mu2E2,
                zfl_dE_domega_over_mu2E2=row.zfl_dE_domega_over_mu2E2,
                energy_relative_deviation=row.energy_relative_deviation,
                positive_spheroidal_l_max=row.positive_spheroidal_l_max,
                conjugate_spheroidal_l_max=row.conjugate_spheroidal_l_max,
                mixing_converged=row.mixing_converged,
            ))
        end
        first_row = spectrum.rows[1]
        memory = spectrum.memory
        push!(soft_rows, (
            spherical_l=ell, m=m,
            re_delta_h_lm=real(memory.positive),
            im_delta_h_lm=imag(memory.positive),
            re_delta_h_l_minus_m=real(memory.conjugate),
            im_delta_h_l_minus_m=imag(memory.conjugate),
            zfl_dE_domega_over_mu2E2=memory.zfl,
            lowest_frequency=first_row.omega,
            numerical_dE_domega_over_mu2E2=first_row.pair_dE_domega_over_mu2E2,
            energy_relative_deviation=first_row.energy_relative_deviation,
            re_positive_soft_ratio=real(first_row.positive_soft_ratio),
            im_positive_soft_ratio=imag(first_row.positive_soft_ratio),
            re_conjugate_soft_ratio=real(first_row.conjugate_soft_ratio),
            im_conjugate_soft_ratio=imag(first_row.conjugate_soft_ratio),
        ))
    end
    write_rows(joinpath(output_dir, "spherical_modes.csv"),
        (:spherical_l, :m, :omega, :re_h_positive, :im_h_positive,
         :re_h_conjugate, :im_h_conjugate, :pair_dE_domega_over_mu2E2,
         :zfl_dE_domega_over_mu2E2, :energy_relative_deviation,
         :positive_spheroidal_l_max, :conjugate_spheroidal_l_max,
         :mixing_converged), mode_rows)
    write_rows(joinpath(output_dir, "soft_limit.csv"),
        (:spherical_l, :m, :re_delta_h_lm, :im_delta_h_lm,
         :re_delta_h_l_minus_m, :im_delta_h_l_minus_m,
         :zfl_dE_domega_over_mu2E2, :lowest_frequency,
         :numerical_dE_domega_over_mu2E2, :energy_relative_deviation,
         :re_positive_soft_ratio, :im_positive_soft_ratio,
         :re_conjugate_soft_ratio, :im_conjugate_soft_ratio), soft_rows)
    write_rows(joinpath(output_dir, "mode_energy_summary.csv"),
        (:spherical_l, :m, :resolved_energy_over_mu2E2,
         :soft_segment_energy_over_mu2E2, :total_energy_over_mu2E2,
         :lowest_frequency, :lowest_energy_relative_deviation, :soft_limit_pass,
         :highest_frequency, :high_endpoint_over_peak, :high_endpoint_pass,
         :mixing_converged),
        result.mode_summaries)
    if !isempty(result.angular_convergence)
        write_rows(joinpath(output_dir, "angular_convergence.csv"),
            (:spherical_l, :shell_energy_over_mu2E2,
             :cumulative_energy_over_mu2E2, :shell_fraction,
             :below_tolerance, :consecutive_below),
            result.angular_convergence)
    end
    raw_keys = sort(collect(keys(result.spheroidal_cache.metadata)))
    raw_modes = Any[result.spheroidal_cache.metadata[key] for key in raw_keys]
    FR.write_fast_reduced_waveform_rows_csv(
        joinpath(output_dir, "spheroidal_modes.csv"), raw_modes,
    )
    open(joinpath(output_dir, "run_summary.txt"), "w") do io
        @printf(io, "orbit_kind=%s\n", result.config.orbit_kind)
        @printf(io, "frequency_count=%d\n", length(result.frequencies))
        @printf(io, "frequency_grid=%s\n", result.config.frequency_grid)
        @printf(io, "interpolation_coordinate=%s\n",
                result.config.interpolation_coordinate)
        @printf(io, "omega_min=%.17g\n", first(result.frequencies))
        @printf(io, "omega_max=%.17g\n", last(result.frequencies))
        @printf(io, "characteristic_omega=%.17g\n", result.window.scale)
        @printf(io, "production_r_outer=%.17g\n", result.production_outer)
        @printf(io, "orbit_reference_r_outer=%.17g\n",
                result.orbit_reference_outer)
        @printf(io, "plunge_anchor_radius=%.17g\n",
                result.plunge_anchor.radius)
        @printf(io, "plunge_anchor_time=%.17g\n",
                result.plunge_anchor.time)
        @printf(io, "plunge_anchor_phi=%.17g\n",
                result.plunge_anchor.phi)
        @printf(io, "outer_radius_capped=%s\n", result.outer_capped)
        @printf(io, "soft_deviation_tolerance=%.17g\n",
                result.config.soft_deviation_tolerance)
        @printf(io, "high_endpoint_fraction_tolerance=%.17g\n",
                result.config.high_endpoint_fraction_tolerance)
        @printf(io, "low_frequency_regular_completion=%s\n",
                result.config.low_frequency_regular_completion)
        @printf(io, "low_frequency_completion_cutoff=%.17g\n",
                first(result.frequencies))
        @printf(io, "low_frequency_completion_tolerance=%.17g\n",
                result.config.low_frequency_completion_tolerance)
        @printf(io, "total_energy_over_mu2E2=%.17g\n",
                result.total_energy_over_mu2E2)
        @printf(io, "compute_backend=CPU\n")
        @printf(io, "julia_threads=%d\n", result.julia_threads)
        @printf(io, "parallel_mode_frequency_jobs=%s\n",
                result.parallel_mode_frequency_jobs)
        @printf(io, "parallel_axis=signed_m_x_signed_frequency\n")
        @printf(io, "orbit_setup_seconds=%.9g\n",
                result.timings.orbit_setup_seconds)
        @printf(io, "frequency_orbit_prewarm_seconds=%.9g\n",
                result.timings.frequency_orbit_prewarm_seconds)
        @printf(io, "mode_solve_seconds=%.9g\n",
                result.timings.mode_solve_seconds)
        @printf(io, "interpolation_seconds=%.9g\n",
                result.timings.interpolation_seconds)
    end
    return output_dir
end

function radix2_negative_dft(values::Vector{ComplexF64})
    n = length(values)
    ispow2(n) || error("radix-2 transform requires a power-of-two length")
    output = copy(values)
    j = 0
    for i in 0:(n - 1)
        i < j && ((output[i + 1], output[j + 1]) =
                  (output[j + 1], output[i + 1]))
        bit = n >> 1
        while bit > 0 && (j & bit) != 0
            j &= ~bit
            bit >>= 1
        end
        j |= bit
    end
    span = 2
    while span <= n
        root = cis(-2pi / span)
        for start in 1:span:n
            phase = 1.0 + 0.0im
            half = span >> 1
            for offset in 0:(half - 1)
                even = output[start + offset]
                odd = phase * output[start + offset + half]
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                phase *= root
            end
        end
        span <<= 1
    end
    return output
end

function cpu_negative_dft_rows(values::Matrix{ComplexF64})
    output = similar(values)
    if Threads.nthreads() > 1 && size(values, 1) > 1
        Threads.@threads :static for row in axes(values, 1)
            output[row, :] .= radix2_negative_dft(vec(values[row, :]))
        end
    else
        for row in axes(values, 1)
            output[row, :] .= radix2_negative_dft(vec(values[row, :]))
        end
    end
    return output
end

function soft_subtracted_frequency_bins(frequencies,
                                        positive_values,
                                        negative_values,
                                        delta_h::Complex,
                                        n::Int;
                                        low_frequency_regular_completion::String="hold",
                                        interpolation_coordinate::String="linear")
    length(frequencies) == length(positive_values) == length(negative_values) ||
        error("signed-frequency sample arrays have different lengths")
    omega_max = frequencies[end]
    domega = 2omega_max / n
    taper_start = 0.9omega_max
    high_taper(omega) = omega <= taper_start ? 1.0 :
        0.5 * (1 + cos(pi * (omega - taper_start) /
                       (omega_max - taper_start)))
    positive_regular = ComplexF64[
        positive_values[i] - soft_frequency_term(delta_h, frequencies[i])
        for i in eachindex(frequencies)
    ]
    negative_regular = ComplexF64[
        negative_values[i] - soft_frequency_term(delta_h, -frequencies[i])
        for i in eachindex(frequencies)
    ]
    low_frequency_regular_completion in
        ("hold", "taper_to_zero", "linear_extrapolate") ||
        error("low_frequency_regular_completion must be hold, taper_to_zero, or linear_extrapolate")
    endpoint_taper(x) = x <= 0 ? 0.0 : x >= 1 ? 1.0 : x^2 * (3 - 2x)
    function regular_value(values, omega)
        if omega <= frequencies[1]
            if low_frequency_regular_completion == "taper_to_zero"
                return values[1] * endpoint_taper(omega / frequencies[1])
            elseif low_frequency_regular_completion == "linear_extrapolate"
                slope = (values[2] - values[1]) /
                        (frequencies[2] - frequencies[1])
                return values[1] + (omega - frequencies[1]) * slope
            end
            return values[1]
        end
        return interpolate_complex(
            frequencies, values, (omega,);
            coordinate=interpolation_coordinate,
        )[1]
    end

    bins = zeros(ComplexF64, n)
    bins[1] = 0.5 * (regular_value(positive_regular, 0.0) +
                     regular_value(negative_regular, 0.0))
    for k in 1:(n >> 1)
        omega = k * domega
        omega <= omega_max || continue
        bins[k + 1] = regular_value(positive_regular, omega) * high_taper(omega)
    end
    for k in 1:((n >> 1) - 1)
        omega = k * domega
        omega <= omega_max || continue
        bins[n - k + 1] = regular_value(negative_regular, omega) * high_taper(omega)
    end
    return bins
end

function soft_subtracted_inverse_mode(frequencies,
                                      positive_values,
                                      negative_values,
                                      delta_h::Complex,
                                      n::Int;
                                      low_frequency_regular_completion::String="hold",
                                      interpolation_coordinate::String="linear")
    bins = soft_subtracted_frequency_bins(
        frequencies, positive_values, negative_values, delta_h, n;
        low_frequency_regular_completion=low_frequency_regular_completion,
        interpolation_coordinate=interpolation_coordinate,
    )
    omega_max = frequencies[end]
    domega = 2omega_max / n
    dt = 2pi / (n * domega)
    regular_unshifted = domega .* radix2_negative_dft(bins)
    regular = vcat(
        regular_unshifted[(n >> 1) + 1:end],
        regular_unshifted[1:(n >> 1)],
    )
    times = Float64[(index - 1 - (n >> 1)) * dt for index in 1:n]
    memory = ComplexF64[soft_time_term(delta_h, time) for time in times]
    return (times=times, regular=regular, memory=memory,
            total=regular + memory)
end

function write_time_domain_outputs(output_dir, result)
    cfg = result.config
    n = cfg.fft_points
    omega_max = result.frequencies[end]
    domega = 2omega_max / n
    dt = 2pi / (n * domega)
    modes = sort(collect(result.spectra); by=first)
    bins = zeros(ComplexF64, length(modes), n)
    memories = Vector{ComplexF64}(undef, length(modes))
    completion_inputs = Vector{Any}(undef, length(modes))
    for (mode_index, ((ell, m), spectrum)) in enumerate(modes)
        positive_rows = spectrum.rows
        companion = get(result.spectra, (ell, -m), nothing)
        negative_same_m = if isnothing(companion)
            values = ComplexF64[]
            for omega in result.frequencies
                negative_mode = spherical_mode!(
                    result.spheroidal_cache, cfg, ell, m, -omega,
                )
                push!(values, anchor_spherical_mode(
                    negative_mode, m, -omega, result.plunge_anchor,
                ).value)
            end
            values
        else
            ComplexF64[getproperty(row, :h_conjugate) for row in companion.rows]
        end
        positive_values = getproperty.(positive_rows, :h_positive)
        memories[mode_index] = spectrum.memory.positive
        completion_inputs[mode_index] = (
            positive_values=positive_values,
            negative_values=negative_same_m,
            delta_h=memories[mode_index],
        )
        bins[mode_index, :] .= soft_subtracted_frequency_bins(
            result.frequencies,
            positive_values,
            negative_same_m,
            memories[mode_index],
            n,
            low_frequency_regular_completion=cfg.low_frequency_regular_completion,
            interpolation_coordinate=cfg.interpolation_coordinate,
        )
    end
    transformed = cpu_negative_dft_rows(bins)
    rows = NamedTuple[]
    completion_rows = NamedTuple[]
    for (mode_index, ((ell, m), _)) in enumerate(modes)
        regular_unshifted = domega .* vec(transformed[mode_index, :])
        regular = vcat(
            regular_unshifted[(n >> 1) + 1:end],
            regular_unshifted[1:(n >> 1)],
        )
        delta_h = memories[mode_index]
        memory_offset = memory_baseline_offset(
            delta_h, cfg.time_domain_memory_baseline,
        )
        completion_input = completion_inputs[mode_index]
        completion_models = ("hold", "taper_to_zero", "linear_extrapolate")
        completion_differences = Dict{String, Float64}()
        for model in completion_models
            if model == cfg.low_frequency_regular_completion
                completion_differences[model] = 0.0
                continue
            end
            alternate = soft_subtracted_inverse_mode(
                result.frequencies,
                completion_input.positive_values,
                completion_input.negative_values,
                completion_input.delta_h,
                n;
                low_frequency_regular_completion=model,
                interpolation_coordinate=cfg.interpolation_coordinate,
            ).regular
            completion_differences[model] = maximum(abs.(alternate .- regular))
        end
        completion_spread = maximum(values(completion_differences))
        completion_relative_spread = completion_spread / max(abs(delta_h), eps(Float64))
        push!(completion_rows, (
            spherical_l=ell,
            m=m,
            selected_model=cfg.low_frequency_regular_completion,
            cutoff_frequency=first(result.frequencies),
            max_difference_to_hold=completion_differences["hold"],
            max_difference_to_taper_to_zero=completion_differences["taper_to_zero"],
            max_difference_to_linear_extrapolate=completion_differences["linear_extrapolate"],
            max_relative_difference_to_memory=completion_relative_spread,
            tolerance=cfg.low_frequency_completion_tolerance,
            completion_stable=completion_relative_spread <= cfg.low_frequency_completion_tolerance,
        ))
        for index in eachindex(regular)
            time = (index - 1 - (n >> 1)) * dt
            memory = soft_time_term(delta_h, time) + memory_offset
            total = regular[index] + memory
            push!(rows, (
                spherical_l=ell, m=m, time=time,
                re_h_regular=real(regular[index]),
                im_h_regular=imag(regular[index]),
                re_h_memory=real(memory),
                im_h_memory=imag(memory),
                re_h=real(total), im_h=imag(total),
            ))
        end
    end
    write_rows(joinpath(output_dir, "spherical_time_modes.csv"),
        (:spherical_l, :m, :time,
         :re_h_regular, :im_h_regular,
         :re_h_memory, :im_h_memory,
         :re_h, :im_h), rows)
    write_rows(joinpath(output_dir, "time_domain_completion_diagnostics.csv"),
        (:spherical_l, :m, :selected_model, :cutoff_frequency,
         :max_difference_to_hold, :max_difference_to_taper_to_zero,
         :max_difference_to_linear_extrapolate,
         :max_relative_difference_to_memory, :tolerance,
         :completion_stable), completion_rows)
    raw_keys = sort(collect(keys(result.spheroidal_cache.metadata)))
    raw_modes = Any[result.spheroidal_cache.metadata[key] for key in raw_keys]
    FR.write_fast_reduced_waveform_rows_csv(
        joinpath(output_dir, "spheroidal_modes.csv"), raw_modes,
    )
    return rows
end

end
