module FastReducedSourceWaveform

include(joinpath(@__DIR__, "EquatorialScatteringSource.jl"))
include(joinpath(@__DIR__, "SNGreenAmplitude.jl"))

using .EquatorialScatteringSource
using .SNGreenAmplitude
import ...KerrGeometry

const UG = EquatorialScatteringSource.UG

const SN_INFINITY_MINIMUM_PHASE = 100.0
const DEFAULT_ASYMPTOTIC_MATCH_PHASE = 200.0
const DEFAULT_SOURCE_STEPS_PER_BRANCH = 32_000
const SOURCE_PHASE_INCREMENT_LIMIT = 0.05
const SOURCE_PHASE_INCREMENT_RELATIVE_TOLERANCE = 1.0e-7
const MAX_AUTOMATIC_SOURCE_REFINEMENTS = 8
const MAX_AUTOMATIC_SOURCE_STEPS = 1_000_000

export FastReducedWaveformConfig,
       FastReducedWaveformResult,
       effective_source_grid,
       effective_source_r_outer,
       scattering_green_tail_required_outer,
       scattering_green_tail_minimum_frequency,
       build_fast_reduced_orbit_cache,
       compute_fast_reduced_waveform,
       fast_reduced_waveform_row,
       write_fast_reduced_waveform_rows_csv

Base.@kwdef mutable struct FastReducedWaveformConfig
    a::Float64 = 0.9
    ell::Int = 2
    m::Int = 2
    omega::Float64 = 0.1
    energy::Float64 = 1.2
    lz::Float64 = 8.0
    carter_q::Float64 = 4.0
    theta_infinity::Float64 = pi / 2
    phi_infinity::Float64 = 0.0
    theta_sign::Float64 = 1.0
    orbit_kind::String = "scattering"
    r_outer_min::Float64 = 80.0
    r_outer_floor::Float64 = 0.0
    nsteps_per_branch::Int = DEFAULT_SOURCE_STEPS_PER_BRANCH
    plunge_inner_rstar_spacing::Float64 = 0.1
    asymptotic_tail_correction::Bool = true
    asymptotic_match_phase::Float64 = DEFAULT_ASYMPTOTIC_MATCH_PHASE
    source_tail_order::Int = 3
    green_tail_correction::Bool = true
    scattering_green_tail_slow_order::Int = 3
    scattering_green_tail_fast_order::Int = 2
    allow_theta_turns::Bool = true
    source_grid::String = "table"
    npoints::Int = 12001
    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_method::String = "linear"
    homogeneous_tolerance::Float64 = 1e-12
    integration_rule::String = "trapezoid"
    radial_backend::AbstractRadialBackend = NATIVE_SN_BACKEND
end

struct FastReducedWaveformResult
    cfg::FastReducedWaveformConfig
    built_source
    rstar::Vector{Float64}
    reduced_source::Vector{ComplexF64}
    green::GreenAmplitudeResult
    elapsed_seconds::Float64
end

effective_source_grid(cfg::FastReducedWaveformConfig) =
    cfg.orbit_kind == "plunge" && cfg.source_grid == "table" ?
    "plunge-composite-rstar" : cfg.source_grid

effective_asymptotic_tail_correction(cfg::FastReducedWaveformConfig) =
    cfg.asymptotic_tail_correction

function scattering_green_tail_phase_rate(cfg::FastReducedWaveformConfig)
    momentum = sqrt(cfg.energy^2 - 1)
    momentum > 0 || error("the finite-velocity scattering phase rate is undefined at E=1")
    abs(cfg.omega) > 0 || error("a scattering Green tail requires nonzero frequency")
    return abs(cfg.omega) / (momentum * (cfg.energy + momentum))
end

function scattering_green_tail_required_outer(cfg::FastReducedWaveformConfig)
    cfg.orbit_kind == "scattering" ||
        error("a scattering Green-tail outer radius was requested for a non-scattering orbit")
    abs(cfg.omega) > 0 || error("a scattering Green tail requires nonzero frequency")
    cfg.asymptotic_match_phase > 0 ||
        error("asymptotic_match_phase must be positive")
    worldline_outer = if cfg.energy == 1.0
        (3cfg.asymptotic_match_phase / (sqrt(2.0) * abs(cfg.omega)))^(2 / 3)
    else
        cfg.asymptotic_match_phase / scattering_green_tail_phase_rate(cfg)
    end
    radial_outer = SN_INFINITY_MINIMUM_PHASE / abs(cfg.omega)
    return max(
        cfg.r_outer_floor,
        worldline_outer,
        radial_outer,
    )
end

function scattering_green_tail_minimum_frequency(cfg::FastReducedWaveformConfig,
                                                  r_outer::Real)
    cfg.orbit_kind == "scattering" ||
        error("a scattering Green-tail frequency bound was requested for a non-scattering orbit")
    r_outer > 0 || error("outer radius must be positive")
    worldline_minimum = if cfg.energy == 1.0
        3cfg.asymptotic_match_phase /
        (sqrt(2.0) * Float64(r_outer)^(3 / 2))
    else
        momentum = sqrt(cfg.energy^2 - 1)
        cfg.asymptotic_match_phase * momentum * (cfg.energy + momentum) /
        Float64(r_outer)
    end
    radial_minimum = SN_INFINITY_MINIMUM_PHASE / Float64(r_outer)
    return max(worldline_minimum, radial_minimum)
end

function effective_source_r_outer(cfg::FastReducedWaveformConfig)
    cfg.r_outer_floor >= 0 || error("r_outer_floor must be nonnegative")
    cfg.r_outer_min > 0 || error("r_outer_min must be positive")
    if cfg.orbit_kind == "scattering" &&
       effective_asymptotic_tail_correction(cfg)
        required_outer = scattering_green_tail_required_outer(cfg)
        return max(cfg.r_outer_min, required_outer)
    elseif cfg.orbit_kind == "plunge" &&
           effective_asymptotic_tail_correction(cfg)
        cfg.asymptotic_match_phase > 0 ||
            error("asymptotic_match_phase must be positive")
        abs(cfg.omega) > 0 || error("a plunge asymptotic tail requires nonzero frequency")
        return max(
            cfg.r_outer_min,
            cfg.r_outer_floor,
            SN_INFINITY_MINIMUM_PHASE / abs(cfg.omega),
        )
    end
    return cfg.r_outer_min
end

function source_config(cfg::FastReducedWaveformConfig; orbit_cache::Bool=false)
    r_outer = effective_source_r_outer(cfg)
    return EquatorialScatteringConfig(
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
        orbit_kind=cfg.orbit_kind,
        source_kind="reduced",
        r_outer_min=r_outer,
        npoints=max(cfg.npoints, 2),
        nsteps_per_branch=cfg.nsteps_per_branch,
        plunge_inner_rstar_spacing=cfg.plunge_inner_rstar_spacing,
        allow_theta_turns=cfg.allow_theta_turns,
        asymptotic_tail_correction=effective_asymptotic_tail_correction(cfg),
        asymptotic_match_phase=cfg.asymptotic_match_phase,
        asymptotic_tail_order=cfg.source_tail_order,
    )
end

function build_fast_reduced_orbit_cache(cfg::FastReducedWaveformConfig)
    return build_scattering_orbit_cache(source_config(cfg; orbit_cache=true))
end

function source_phase_increment_details(built)
    maximum_increment = -Inf
    maximum_table = 0
    maximum_index = 0
    for (table_index, table) in enumerate(built.tables)
        length(table.phase_arguments) >= 2 || continue
        increment, segment_index = findmax(abs.(diff(table.phase_arguments)))
        if increment > maximum_increment
            maximum_increment = increment
            maximum_table = table_index
            maximum_index = segment_index
        end
    end
    maximum_table != 0 || error("source table has no phase increments")
    return maximum_increment, maximum_table, maximum_index
end

source_phase_increment(built) = first(source_phase_increment_details(built))

function locally_refine_plunge_inner_grid!(active, cache, built,
                                            table_index::Int,
                                            segment_index::Int,
                                            increment::Float64)
    active.orbit_kind == "plunge" || return false
    table = built.tables[table_index]
    r_right = table.r[segment_index + 1]
    r_right <= 200.0 * (1 + 1.0e-12) || return false
    any(piece -> piece.r_start > 200.0 && piece.r_stop < 200.0, cache.pieces) ||
        return false
    factor = ceil(Int, increment / SOURCE_PHASE_INCREMENT_LIMIT)
    factor > 1 || return false
    active.plunge_inner_rstar_spacing /= factor
    active.nsteps_per_branch += sum(
        max(2, ceil(Int, (
            KerrGeometry.rstar_from_r(cache.kerr.a, min(200.0, piece.r_start)) -
            KerrGeometry.rstar_from_r(cache.kerr.a, piece.r_stop)
        ) / active.plunge_inner_rstar_spacing)) -
        max(2, ceil(Int, (
            KerrGeometry.rstar_from_r(cache.kerr.a, min(200.0, piece.r_start)) -
            KerrGeometry.rstar_from_r(cache.kerr.a, piece.r_stop)
        ) / (active.plunge_inner_rstar_spacing * factor)))
        for piece in cache.pieces if piece.r_start > 200.0 && piece.r_stop < 200.0
    )
    active.nsteps_per_branch <= MAX_AUTOMATIC_SOURCE_STEPS || error(
        "automatic local plunge source refinement requires " *
        "$(active.nsteps_per_branch) steps per branch, above the safe limit " *
        "$MAX_AUTOMATIC_SOURCE_STEPS",
    )
    return true
end

function resolved_fast_reduced_source(cfg::FastReducedWaveformConfig;
                                      orbit_cache=nothing)
    active = deepcopy(cfg)
    candidate_cache = orbit_cache
    phase_history = Float64[]
    for _ in 1:MAX_AUTOMATIC_SOURCE_REFINEMENTS
        scfg = source_config(active)
        cache = candidate_cache === nothing ?
                build_scattering_orbit_cache(source_config(active; orbit_cache=true)) :
                candidate_cache
        built = build_scattering_source_from_cache(scfg, cache)
        increment, table_index, segment_index = source_phase_increment_details(built)
        push!(phase_history, increment)
        increment <= SOURCE_PHASE_INCREMENT_LIMIT *
                     (1 + SOURCE_PHASE_INCREMENT_RELATIVE_TOLERANCE) &&
            return active, cache, built

        if locally_refine_plunge_inner_grid!(
            active, cache, built, table_index, segment_index, increment,
        )
            candidate_cache = nothing
            continue
        end

        required_steps = ceil(Int, active.nsteps_per_branch *
                              increment / SOURCE_PHASE_INCREMENT_LIMIT)
        required_steps = max(required_steps, active.nsteps_per_branch + 1)
        required_steps <= MAX_AUTOMATIC_SOURCE_STEPS || error(
            "automatic source-phase resolution requires $required_steps steps per " *
            "branch, above the safe limit $MAX_AUTOMATIC_SOURCE_STEPS",
        )
        active.nsteps_per_branch = required_steps
        candidate_cache = nothing
    end
    error(
        "automatic source-phase resolution did not converge for " *
        "(ell,m,omega)=($(cfg.ell),$(cfg.m),$(cfg.omega)); " *
        "phase increments=$(phase_history)",
    )
end

function table_rstar_and_source(cfg::FastReducedWaveformConfig, built, table)
    outer_r = table.r
    issorted(outer_r) || error("branch-summed source table is not radius ordered")
    outer_rstar = Vector{Float64}(undef, length(outer_r))
    outer_source = Vector{ComplexF64}(undef, length(outer_r))
    @inbounds for i in eachindex(outer_r)
        r = outer_r[i]
        outer_rstar[i] = KerrGeometry.rstar_from_r(cfg.a, r)
        total = table.Wnn[i] + table.Wnmb[i] + table.Wmbmb[i]
        prefactor = UG.delta(built.kerr, r) / (r^2 * (r^2 + cfg.a^2)^(3 / 2))
        outgoing_phase = exp(-1im * UG.k_over_delta_antiderivative(
            built.kerr, r, cfg.omega, cfg.m,
        ))
        outer_source[i] = built.particle_mass * total * prefactor * outgoing_phase
    end
    return outer_rstar, outer_source
end

function table_grid_source(cfg::FastReducedWaveformConfig, built)
    table = built.branch_summed_table
    outer_rstar, outer_source = table_rstar_and_source(cfg, built, table)
    if cfg.orbit_kind == "scattering"
        rs_min = KerrGeometry.rstar_from_r(cfg.a, built.r_minimum)
        rs_turn = KerrGeometry.rstar_from_r(cfg.a, built.r_turn)
        inner_count = max(401, ceil(Int, (rs_turn - rs_min) / 0.05) + 1)
        inner_rstar = collect(range(
            rs_min, rs_turn - 1e-10; length=inner_count,
        ))
        rstar = vcat(inner_rstar, outer_rstar)
        source = vcat(
            ComplexF64[
                built.reduced_at_r(KerrGeometry.r_from_rstar(cfg.a, rs))
                for rs in inner_rstar
            ],
            outer_source,
        )
    else
        rstar = outer_rstar
        source = outer_source
    end
    return rstar, source
end

function uniform_rstar_source(cfg::FastReducedWaveformConfig, built)
    rs_min = KerrGeometry.rstar_from_r(cfg.a, built.r_minimum)
    rs_max = KerrGeometry.rstar_from_r(cfg.a, built.r_maximum)
    rstar = collect(range(rs_min, rs_max; length=cfg.npoints))
    source = ComplexF64[
        built.reduced_at_r(KerrGeometry.r_from_rstar(cfg.a, rs))
        for rs in rstar
    ]
    return rstar, source
end

function plunge_composite_rstar_source(cfg::FastReducedWaveformConfig, built)
    length(built.tables) == 1 ||
        error("a plunge source requires exactly one source table")
    table = only(built.tables)
    outer_rstar, outer_source = table_rstar_and_source(cfg, built, table)
    rs_min = KerrGeometry.rstar_from_r(cfg.a, built.r_minimum)
    physical_outer = built.r_maximum
    source_end = searchsortedlast(
        table.r,
        physical_outer + 1e-10 * max(1.0, physical_outer),
    )
    source_end >= 2 ||
        error("plunge source table does not reach the physical outer boundary")
    rs_max = KerrGeometry.rstar_from_r(cfg.a, physical_outer)
    r_cut = min(200.0, physical_outer)
    rs_cut = KerrGeometry.rstar_from_r(cfg.a, r_cut)
    if rs_cut <= rs_min || rs_cut >= rs_max
        return @view(outer_rstar[1:source_end]),
               @view(outer_source[1:source_end])
    end

    inner_count = max(3, ceil(Int, (rs_cut - rs_min) / 0.1) + 1)
    inner = collect(range(rs_min, rs_cut; length=inner_count))
    inner_source = ComplexF64[
        built.reduced_at_r(KerrGeometry.r_from_rstar(cfg.a, rs))
        for rs in inner
    ]
    outer_start = searchsortedfirst(table.r, r_cut)
    outer_start <= source_end ||
        error("plunge source table does not reach the composite outer region")
    if isapprox(outer_rstar[outer_start], rs_cut; rtol=1e-12, atol=1e-12)
        outer_start += 1
    end
    outer_start <= source_end ||
        error("plunge source table has no points beyond the composite join")
    return vcat(inner, @view(outer_rstar[outer_start:source_end])),
           vcat(inner_source, @view(outer_source[outer_start:source_end]))
end

function source_grid_values(cfg::FastReducedWaveformConfig, built)
    grid = effective_source_grid(cfg)
    grid == "table" && return table_grid_source(cfg, built)
    grid == "uniform" && return uniform_rstar_source(cfg, built)
    grid == "plunge-composite-rstar" &&
        return plunge_composite_rstar_source(cfg, built)
    error("source_grid must be table or uniform")
end

function green_config(cfg::FastReducedWaveformConfig, rstar, lambda::Real=NaN)
    return GreenAmplitudeConfig(
        a=cfg.a,
        spin=-2,
        ell=cfg.ell,
        m=cfg.m,
        omega=cfg.omega,
        lambda=Float64(lambda),
        energy_norm=cfg.energy,
        homogeneous_rsin=cfg.homogeneous_rsin,
        homogeneous_rsout_min=max(cfg.homogeneous_rsout_min, maximum(rstar)),
        homogeneous_method=cfg.homogeneous_method,
        homogeneous_tolerance=cfg.homogeneous_tolerance,
        integration_rule=cfg.integration_rule,
        radial_backend=cfg.radial_backend,
    )
end

function explicit_phase_integral(r, amplitude, phase, first_index::Int)
    first_index >= length(r) && return 0.0 + 0.0im
    total = 0.0 + 0.0im
    @inbounds for i in first_index:(length(r) - 1)
        total += EquatorialScatteringSource.oscillatory_segment(
            r[i], r[i + 1], amplitude[i], amplitude[i + 1],
            phase[i], phase[i + 1],
        )
    end
    return ComplexF64(total)
end

function plunge_green_tail_integral(kernel::GreenKernel, built,
                                    cfg::FastReducedWaveformConfig)
    length(built.tables) == 1 ||
        error("a plunge Green tail requires exactly one source table")
    table = only(built.tables)
    physical_outer = built.r_maximum
    table.r[end] + 1e-12 * max(1.0, physical_outer) >= physical_outer ||
        error("plunge source table does not reach the physical outer radius")
    insertion = searchsortedfirst(table.r, physical_outer)
    candidates = filter(i -> 1 <= i <= length(table.r), (insertion - 1):insertion)
    isempty(candidates) &&
        error("plunge source table does not reach the physical outer radius")
    numeric_start_global = argmin(i -> abs(table.r[i] - physical_outer), candidates)
    abs(table.r[numeric_start_global] - physical_outer) <=
        1e-8 * max(1.0, physical_outer) ||
        error("plunge source table misses the physical outer boundary")

    fit_count = EquatorialScatteringSource.asymptotic_fit_point_count(table.r)
    fit_start_global = length(table.r) - fit_count + 1
    work_start = min(numeric_start_global, fit_start_global)
    r = @view table.r[work_start:end]
    phase = @view table.phase_arguments[work_start:end]
    source_rate = @view table.chi_prime_values[work_start:end]
    W = @view(table.Wnn[work_start:end]) .+
        @view(table.Wnmb[work_start:end]) .+
        @view(table.Wmbmb[work_start:end])
    smooth_W = W .* cis.(-phase)
    chi = Float64[
        UG.k_over_delta_antiderivative(
            built.kerr, radius, cfg.omega, cfg.m,
        )
        for radius in r
    ]
    radial_prefactor = inv.(r .^ 2 .* sqrt.(r .^ 2 .+ cfg.a^2))
    slow_amplitude = built.particle_mass * kernel.bref .* radial_prefactor .* smooth_W
    fast_amplitude = built.particle_mass * kernel.binc .* radial_prefactor .* smooth_W
    fast_phase = phase .- 2 .* chi
    fast_rate = source_rate .-
                2 .* UG.phase_integrand.(
                    Ref(built.kerr), r, cfg.omega, cfg.m,
                )

    numeric_start = numeric_start_global - work_start + 1
    tail = explicit_phase_integral(r, slow_amplitude, phase, numeric_start) +
           explicit_phase_integral(r, fast_amplitude, fast_phase, numeric_start)

    fit_start = length(r) - fit_count + 1
    fit_r = @view r[fit_start:end]
    fit_phase = @view phase[fit_start:end]
    fit_fast_phase = @view fast_phase[fit_start:end]
    fit_source_rate = @view source_rate[fit_start:end]
    fit_fast_rate = @view fast_rate[fit_start:end]
    slow_values = @view(slow_amplitude[fit_start:end]) .* cis.(fit_phase)
    fast_values = @view(fast_amplitude[fit_start:end]) .* cis.(fit_fast_phase)
    tail_function = cfg.energy == 1.0 ?
                    EquatorialScatteringSource.parabolic_fitted_oscillatory_tail :
                    EquatorialScatteringSource.fitted_oscillatory_tail
    tail += tail_function(
        fit_r,
        slow_values,
        cis.(fit_phase),
        fit_source_rate;
        order=3,
        max_points=fit_count,
    )
    tail += tail_function(
        fit_r,
        fast_values,
        cis.(fit_fast_phase),
        fit_fast_rate;
        order=2,
        max_points=fit_count,
    )
    return ComplexF64(tail)
end

function scattering_fitted_green_tail(cfg::FastReducedWaveformConfig,
                                      r, integrand, phase, phase_rate;
                                      order::Int, max_points::Int)
    if cfg.energy == 1.0
        return EquatorialScatteringSource.parabolic_fitted_oscillatory_tail(
            r, integrand, phase, phase_rate;
            order=order, max_points=max_points,
        )
    end
    return EquatorialScatteringSource.fitted_oscillatory_tail(
        r, integrand, phase, phase_rate;
        order=order, max_points=max_points,
    )
end

function scattering_green_tail_integral(kernel::GreenKernel, built,
                                        cfg::FastReducedWaveformConfig)
    length(built.tables) == 2 ||
        error("a scattering Green tail requires incoming and outgoing source tables")
    coefficient_order = kernel.cfg.infinity_expansion_order
    outgoing_coefficients = SNGreenAmplitude.infinity_coefficients(
        kernel.cfg.radial_backend, kernel.cfg, kernel.lambda,
        :outgoing, coefficient_order,
    )
    incoming_coefficients = SNGreenAmplitude.infinity_coefficients(
        kernel.cfg.radial_backend, kernel.cfg, kernel.lambda,
        :ingoing, coefficient_order,
    )
    asymptotic_factor(coefficients, radius) = sum(
        coefficients[order + 1] / (cfg.omega * radius)^order
        for order in 0:coefficient_order;
        init=0.0 + 0.0im,
    )
    tail = 0.0 + 0.0im
    for table in built.tables
        fit_count = EquatorialScatteringSource.asymptotic_fit_point_count(table.r)
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
            )
            for radius in r
        ]
        orbit_phase = phase .- radial_phase
        rstar = KerrGeometry.rstar_from_r.(Ref(cfg.a), r)
        outgoing_phase = orbit_phase .+ cfg.omega .* rstar
        incoming_phase = orbit_phase .- cfg.omega .* rstar
        radial_phase_rate = UG.phase_integrand.(
            Ref(built.kerr), r, cfg.omega, cfg.m,
        )
        orbit_rate = source_rate .- radial_phase_rate
        rstar_rate = (r .^ 2 .+ cfg.a^2) ./
                     UG.delta.(Ref(built.kerr), r)
        outgoing_rate = orbit_rate .+ cfg.omega .* rstar_rate
        incoming_rate = orbit_rate .- cfg.omega .* rstar_rate
        outgoing_factor = ComplexF64[
            asymptotic_factor(outgoing_coefficients, radius) for radius in r
        ]
        incoming_factor = ComplexF64[
            asymptotic_factor(incoming_coefficients, radius) for radius in r
        ]
        reconstructed_outer =
            kernel.bref * outgoing_factor[end] * cis(cfg.omega * rstar[end]) +
            kernel.binc * incoming_factor[end] * cis(-cfg.omega * rstar[end])
        reconstruction_error = abs(reconstructed_outer - kernel.xin[end]) /
                               max(abs(kernel.xin[end]), eps(Float64))
        reconstruction_error < 5e-8 || error(
            "scattering Green-tail asymptotic basis does not reconstruct Xin",
        )
        radial_prefactor = inv.(r .^ 2 .* sqrt.(r .^ 2 .+ cfg.a^2))
        outgoing_amplitude = built.particle_mass * kernel.bref .*
                             radial_prefactor .* smooth_W .* outgoing_factor
        incoming_amplitude = built.particle_mass * kernel.binc .*
                             radial_prefactor .* smooth_W .* incoming_factor
        tail += scattering_fitted_green_tail(
            cfg,
            r,
            outgoing_amplitude .* cis.(outgoing_phase),
            cis.(outgoing_phase),
            outgoing_rate;
            order=cfg.scattering_green_tail_slow_order,
            max_points=fit_count,
        )
        tail += scattering_fitted_green_tail(
            cfg,
            r,
            incoming_amplitude .* cis.(incoming_phase),
            cis.(incoming_phase),
            incoming_rate;
            order=cfg.scattering_green_tail_fast_order,
            max_points=fit_count,
        )
    end
    return ComplexF64(tail)
end

function compute_fast_reduced_waveform(cfg::FastReducedWaveformConfig;
                                       orbit_cache=nothing)
    t0 = time()
    cfg, cache, built = resolved_fast_reduced_source(cfg; orbit_cache=orbit_cache)
    scfg = source_config(cfg)
    rstar, source = source_grid_values(cfg, built)
    gcfg = green_config(cfg, rstar, real(getproperty(built.harmonic, :lambda)))
    green = if cfg.orbit_kind == "plunge" &&
               effective_asymptotic_tail_correction(cfg)
        kernel = build_green_kernel(rstar, gcfg)
        inner = apply_green_kernel(kernel, source)
        tail_integral = plunge_green_tail_integral(kernel, built, cfg)
        SNGreenAmplitude.green_result_from_integral(
            kernel,
            inner.reduced_source,
            inner.integrand,
            inner.cumulative_integral,
            inner.integral + tail_integral,
            "$(inner.integration_rule)+abel-plunge-tail",
        )
    elseif cfg.orbit_kind == "scattering" &&
               effective_source_grid(cfg) == "table"
        kernel = build_green_kernel(rstar, gcfg)
        table = built.branch_summed_table
        inner_count = length(rstar) - length(table.r)
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
                built.kerr,
                built.constants,
                built.mode,
                point,
                cfg.omega,
                cfg.m,
                t_origin=built.t_origin,
                phi_origin=built.phi_origin,
            ).total.q2
            for point in turn_points
        )
        radial_prime = UG.radial_potential_derivative(
            built.kerr, built.constants, built.r_turn,
        )
        sigma_turn = UG.sigma(
            built.kerr, built.r_turn, turn_points[1].theta,
        )
        source_phase_turn = cis(-UG.k_over_delta_antiderivative(
            built.kerr, built.r_turn, cfg.omega, cfg.m,
        ))
        endpoint_value =
            kernel.xin[inner_count] * scfg.particle_mass *
            source_phase_turn *
            2 * sigma_turn * q2_turn /
            (
                built.r_turn^2 *
                sqrt(built.r_turn^2 + cfg.a^2) *
                sqrt(radial_prime)
            )
        inner = apply_green_kernel_partitioned(
            kernel,
            source,
            inner_count,
            table.integration_x,
            drstar_du,
            endpoint_value,
        )
        tail_integral = effective_asymptotic_tail_correction(cfg) &&
                        cfg.green_tail_correction ?
                        scattering_green_tail_integral(kernel, built, cfg) :
                        0.0 + 0.0im
        SNGreenAmplitude.green_result_from_integral(
            kernel,
            inner.reduced_source,
            inner.integrand,
            inner.cumulative_integral,
            inner.integral + tail_integral,
            "$(inner.integration_rule)+abel-scattering-tail",
        )
    else
        compute_green_amplitude(rstar, source, gcfg)
    end
    return FastReducedWaveformResult(
        cfg,
        built,
        rstar,
        source,
        green,
        time() - t0,
    )
end

csv_value(x::AbstractFloat) = isnan(x) ? "NaN" : repr(Float64(x))
csv_value(x::Integer) = string(x)
csv_value(x::Bool) = string(x)
csv_value(x::AbstractString) = x

const FAST_REDUCED_COLUMNS = [
    :ell,
    :m,
    :omega,
    :orbit_kind,
    :energy,
    :lz,
    :carter_q,
    :theta_infinity,
    :phi_infinity,
    :theta_sign,
    :r_outer_min,
    :nsteps_per_branch,
    :asymptotic_match_phase,
    :effective_source_r_outer,
    :source_grid,
    :source_points,
    :asymptotic_tail_correction,
    :homogeneous_rsin,
    :homogeneous_rsout,
    :integration_rule,
    :re_Xinf_over_c0,
    :im_Xinf_over_c0,
    :abs_Xinf_over_c0,
    :re_Zinf,
    :im_Zinf,
    :abs_Zinf,
    :one_sided_dE_domega_over_mu2,
    :one_sided_dE_domega_over_mu2E2,
    :elapsed_seconds,
]

function fast_reduced_waveform_row(result::FastReducedWaveformResult)
    cfg = result.cfg
    green = result.green
    return (
        ell=cfg.ell,
        m=cfg.m,
        omega=cfg.omega,
        orbit_kind=cfg.orbit_kind,
        energy=cfg.energy,
        lz=cfg.lz,
        carter_q=cfg.carter_q,
        theta_infinity=cfg.theta_infinity,
        phi_infinity=cfg.phi_infinity,
        theta_sign=cfg.theta_sign,
        r_outer_min=cfg.r_outer_min,
        nsteps_per_branch=cfg.nsteps_per_branch,
        asymptotic_match_phase=cfg.asymptotic_match_phase,
        effective_source_r_outer=effective_source_r_outer(cfg),
        source_grid=effective_source_grid(cfg),
        source_points=length(result.rstar),
        asymptotic_tail_correction=effective_asymptotic_tail_correction(cfg),
        homogeneous_rsin=green.homogeneous_rsin,
        homogeneous_rsout=green.homogeneous_rsout,
        integration_rule=green.integration_rule,
        re_Xinf_over_c0=real(green.xinf_over_c0),
        im_Xinf_over_c0=imag(green.xinf_over_c0),
        abs_Xinf_over_c0=abs(green.xinf_over_c0),
        re_Zinf=real(green.zinf),
        im_Zinf=imag(green.zinf),
        abs_Zinf=abs(green.zinf),
        one_sided_dE_domega_over_mu2=green.one_sided_dE_domega_over_mu2,
        one_sided_dE_domega_over_mu2E2=green.one_sided_dE_domega_over_mu2Enorm2,
        elapsed_seconds=result.elapsed_seconds,
    )
end

function write_fast_reduced_waveform_rows_csv(path::AbstractString, results)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(String.(FAST_REDUCED_COLUMNS), ","))
        for result in results
            row = result isa FastReducedWaveformResult ?
                  fast_reduced_waveform_row(result) : result
            println(io, join((csv_value(getproperty(row, col))
                              for col in FAST_REDUCED_COLUMNS), ","))
        end
    end
    return path
end

end
