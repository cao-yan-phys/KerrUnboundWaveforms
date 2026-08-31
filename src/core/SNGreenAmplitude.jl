module SNGreenAmplitude

import ....NativeSN
import ....NativeSNLinear
using SpinWeightedSpheroidalHarmonics

export GreenAmplitudeConfig,
       GreenKernel,
       GreenAmplitudeResult,
       build_green_kernel,
       apply_green_kernel,
       apply_green_kernel_partitioned,
       integrate_grid,
       compute_green_amplitude,
       compute_green_amplitude_from_function,
       read_reduced_source_csv,
       write_green_summary_csv,
       relative_difference

Base.@kwdef mutable struct GreenAmplitudeConfig
    a::Float64 = 0.9
    spin::Int = -2
    ell::Int = 2
    m::Int = 2
    omega::Float64 = 0.1
    lambda::Float64 = NaN
    mu::Float64 = 1.0
    energy_norm::Float64 = 1.0
    horizon_expansion_order::Int = 8
    infinity_expansion_order::Int = 10
    homogeneous_rsin::Float64 = -50.0
    homogeneous_rsout_min::Float64 = 500.0
    homogeneous_method::String = "linear"
    homogeneous_tolerance::Float64 = 1e-12
    integration_rule::String = "trapezoid"
end

struct GreenAmplitudeResult
    cfg::GreenAmplitudeConfig
    rstar::Vector{Float64}
    homogeneous_rsin::Float64
    homogeneous_rsout::Float64
    reduced_source::Vector{ComplexF64}
    xin::Vector{ComplexF64}
    integrand::Vector{ComplexF64}
    cumulative_integral::Vector{ComplexF64}
    integral::ComplexF64
    integration_rule::String
    lambda::ComplexF64
    c0::ComplexF64
    bref::ComplexF64
    binc::ComplexF64
    xinf_over_c0::ComplexF64
    zinf::ComplexF64
    one_sided_dE_domega_over_mu2::Float64
    one_sided_dE_domega_over_mu2Enorm2::Float64
end

struct GreenKernel
    cfg::GreenAmplitudeConfig
    rstar::Vector{Float64}
    homogeneous_rsin::Float64
    homogeneous_rsout::Float64
    xin::Vector{ComplexF64}
    lambda::ComplexF64
    c0::ComplexF64
    bref::ComplexF64
    binc::ComplexF64
end

function validate_config(cfg::GreenAmplitudeConfig)
    cfg.spin == -2 || error("only the native spin -2 SN radial equation is supported")
    cfg.omega == 0 && error("omega=0 is not supported")
    cfg.homogeneous_method == "linear" ||
        error("homogeneous_method must be linear")
    cfg.integration_rule in ("trapezoid", "simpson", "boole", "oscillatory") ||
        error("integration_rule must be trapezoid, simpson, boole, or oscillatory")
    return cfg
end

function cumulative_trapezoid(xs, ys)
    length(xs) == length(ys) || error("xs and ys lengths differ")
    length(xs) >= 2 || error("need at least two grid points")
    out = Vector{ComplexF64}(undef, length(xs))
    out[1] = 0.0 + 0.0im
    for i in 2:length(xs)
        h = xs[i] - xs[i - 1]
        out[i] = out[i - 1] + 0.5 * h * (ys[i - 1] + ys[i])
    end
    return out
end

function trapezoid_integral(xs, ys)
    length(xs) == length(ys) || error("xs and ys lengths differ")
    length(xs) >= 2 || error("need at least two grid points")
    total = 0.0 + 0.0im
    @inbounds for i in 2:length(xs)
        total += 0.5 * (xs[i] - xs[i - 1]) * (ys[i - 1] + ys[i])
    end
    return ComplexF64(total)
end

function sorted_rstar_grid(rstar)
    rs = Float64.(collect(rstar))
    order = sortperm(rs)
    rs = rs[order]
    strictly_increasing(rs) ||
        error("rstar grid must be strictly increasing after sorting")
    return rs, order
end

function strictly_increasing(values)
    @inbounds for i in 2:length(values)
        values[i] > values[i - 1] || return false
    end
    return true
end

function uniform_grid_step(xs; rtol::Float64=1e-9, atol::Float64=1e-12)
    length(xs) >= 2 || error("need at least two grid points")
    h = xs[2] - xs[1]
    h > 0 || error("grid must be strictly increasing")
    for i in 3:length(xs)
        hi = xs[i] - xs[i - 1]
        abs(hi - h) <= atol + rtol * max(abs(h), abs(hi)) ||
            error("high-order integration requires a uniform rstar grid")
    end
    return h
end

function simpson_integral(xs, ys)
    n = length(xs)
    n == length(ys) || error("xs and ys lengths differ")
    isodd(n) || error("Simpson rule requires an odd number of points")
    h = uniform_grid_step(xs)
    total = ys[1] + ys[end]
    for i in 2:2:(n - 1)
        total += 4 * ys[i]
    end
    for i in 3:2:(n - 2)
        total += 2 * ys[i]
    end
    return h * total / 3
end

function boole_integral(xs, ys)
    n = length(xs)
    n == length(ys) || error("xs and ys lengths differ")
    (n - 1) % 4 == 0 || error("Boole rule requires number of intervals divisible by 4")
    h = uniform_grid_step(xs)
    total = 0.0 + 0.0im
    for i in 1:4:(n - 4)
        total += 7 * ys[i] + 32 * ys[i + 1] + 12 * ys[i + 2] +
                 32 * ys[i + 3] + 7 * ys[i + 4]
    end
    return 2 * h * total / 45
end

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

function unwrap_phase_values(values)
    n = length(values)
    phase = Vector{Float64}(undef, n)
    phase[1] = angle(values[1])
    for i in 2:n
        raw = angle(values[i])
        delta = raw - angle(values[i - 1])
        delta -= 2pi * round(delta / (2pi))
        phase[i] = phase[i - 1] + delta
    end
    return phase
end

function oscillatory_integral_values(xs, ys)
    length(xs) == length(ys) || error("xs and ys lengths differ")
    length(xs) >= 2 || error("need at least two grid points")
    phase = unwrap_phase_values(ys)
    amplitude = ys .* cis.(-phase)
    total = 0.0 + 0.0im
    for i in 1:(length(xs) - 1)
        total += oscillatory_segment(
            xs[i], xs[i + 1],
            amplitude[i], amplitude[i + 1],
            phase[i], phase[i + 1],
        )
    end
    return ComplexF64(total)
end

function integrate_grid(xs, ys, rule::AbstractString)
    rule == "trapezoid" && return trapezoid_integral(xs, ys)
    rule == "simpson" && return ComplexF64(simpson_integral(xs, ys))
    rule == "boole" && return ComplexF64(boole_integral(xs, ys))
    rule == "oscillatory" && return oscillatory_integral_values(xs, ys)
    error("unknown integration rule: $rule")
end

function build_green_kernel(rstar, cfg::GreenAmplitudeConfig)
    validate_config(cfg)
    length(rstar) >= 2 || error("need at least two source points")
    rs_input = rstar isa Vector{Float64} ? rstar : Float64.(collect(rstar))
    rs = if strictly_increasing(rs_input)
        rs_input
    else
        sorted_rstar_grid(rs_input)[1]
    end
    rsin = min(rs[1], cfg.homogeneous_rsin)
    rsout = max(rs[end], cfg.homogeneous_rsout_min)
    lambda = isfinite(cfg.lambda) ? cfg.lambda :
        spin_weighted_spheroidal_harmonic(
            cfg.spin, cfg.ell, cfg.m, cfg.a * cfg.omega; method="auto",
        ).lambda
    mode = NativeSN.SNMode(cfg.a, cfg.m, cfg.omega, lambda)
    solution = NativeSNLinear.solve_in(
        mode;
        rsin=rsin,
        rsout=rsout,
        tolerance=cfg.homogeneous_tolerance,
        horizon_order=cfg.horizon_expansion_order,
    )
    amplitudes = NativeSNLinear.match_infinity(
        solution; order=cfg.infinity_expansion_order,
    )
    bref = amplitudes.bref
    binc = amplitudes.binc
    c0 = NativeSN.eta_coefficient(mode, 0)
    xin_values = Vector{ComplexF64}(undef, length(rs))
    solution.numerical_solution(xin_values, rs; idxs=1)
    return GreenKernel(
        cfg,
        rs,
        rsin,
        rsout,
        xin_values,
        ComplexF64(lambda),
        ComplexF64(c0),
        ComplexF64(bref),
        ComplexF64(binc),
    )
end

function green_result_from_integral(
    kernel::GreenKernel,
    src,
    integrand,
    cumulative,
    integral,
    integration_rule,
)
    cfg = kernel.cfg
    xinf_over_c0 = integral / (2im * cfg.omega * kernel.binc)
    zinf = -4 * cfg.omega^2 * xinf_over_c0
    one_sided = 8 * cfg.mu^2 * cfg.omega^2 * abs2(xinf_over_c0)
    return GreenAmplitudeResult(
        cfg,
        kernel.rstar,
        kernel.homogeneous_rsin,
        kernel.homogeneous_rsout,
        src,
        kernel.xin,
        integrand,
        cumulative,
        ComplexF64(integral),
        integration_rule,
        kernel.lambda,
        kernel.c0,
        kernel.bref,
        kernel.binc,
        ComplexF64(xinf_over_c0),
        ComplexF64(zinf),
        Float64(one_sided / cfg.mu^2),
        Float64(one_sided / (cfg.mu^2 * cfg.energy_norm^2)),
    )
end

function apply_green_kernel(kernel::GreenKernel, reduced_source)
    length(reduced_source) == length(kernel.rstar) ||
        error("source length does not match SN Green kernel grid")
    src = reduced_source isa Vector{ComplexF64} ? reduced_source :
          ComplexF64.(collect(reduced_source))
    cfg = kernel.cfg
    integrand = kernel.xin .* src
    cumulative = cumulative_trapezoid(kernel.rstar, integrand)
    integral = integrate_grid(kernel.rstar, integrand, cfg.integration_rule)
    return green_result_from_integral(
        kernel, src, integrand, cumulative, integral, cfg.integration_rule,
    )
end

function apply_green_kernel_partitioned(
    kernel::GreenKernel,
    reduced_source,
    inner_count::Int,
    outer_parameter,
    drstar_dparameter,
    outer_endpoint_value,
)
    length(reduced_source) == length(kernel.rstar) ||
        error("source length does not match SN Green kernel grid")
    2 <= inner_count < length(kernel.rstar) ||
        error("invalid inner partition size")
    outer_length = length(kernel.rstar) - inner_count
    length(outer_parameter) == outer_length ||
        error("outer parameter length does not match source partition")
    length(drstar_dparameter) == outer_length ||
        error("outer Jacobian length does not match source partition")
    src = reduced_source isa Vector{ComplexF64} ? reduced_source :
          ComplexF64.(collect(reduced_source))
    integrand = kernel.xin .* src
    inner_integral = trapezoid_integral(
        @view(kernel.rstar[1:inner_count]),
        @view(integrand[1:inner_count]),
    )
    outer_integral = 0.0 + 0.0im
    previous_x = 0.0
    previous_value = ComplexF64(outer_endpoint_value)
    @inbounds for j in eachindex(outer_parameter)
        x = Float64(outer_parameter[j])
        value = integrand[inner_count + j] * drstar_dparameter[j]
        outer_integral += 0.5 * (x - previous_x) * (previous_value + value)
        previous_x = x
        previous_value = value
    end
    integral = inner_integral + outer_integral
    cumulative = cumulative_trapezoid(kernel.rstar, integrand)
    return green_result_from_integral(
        kernel,
        src,
        integrand,
        cumulative,
        integral,
        "partitioned-rstar-u",
    )
end

function compute_green_amplitude(rstar, reduced_source, cfg::GreenAmplitudeConfig)
    length(rstar) == length(reduced_source) ||
        error("rstar and reduced_source lengths differ")
    rs, order = sorted_rstar_grid(rstar)
    src = ComplexF64.(collect(reduced_source))[order]
    kernel = build_green_kernel(rs, cfg)
    return apply_green_kernel(kernel, src)
end

function compute_green_amplitude_from_function(source, rstar, cfg::GreenAmplitudeConfig)
    rs, _ = sorted_rstar_grid(rstar)
    kernel = build_green_kernel(rs, cfg)
    src = ComplexF64[source(x) for x in rs]
    return apply_green_kernel(kernel, src)
end

function split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if inquote && i < lastindex(line) && line[nextind(line, i)] == '"'
                print(buf, '"')
                i = nextind(line, i)
            else
                inquote = !inquote
            end
        elseif c == ',' && !inquote
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function read_reduced_source_csv(path::AbstractString;
                                 rstar_col::String="rstar",
                                 re_col::String="re_source",
                                 im_col::String="im_source")
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && error("empty CSV: $path")
    header = strip.(split_csv_line(lines[1]))
    idx(name) = begin
        found = findfirst(==(name), header)
        found === nothing && error("missing column '$name' in $path")
        found
    end
    irs = idx(rstar_col)
    ire = idx(re_col)
    iim = idx(im_col)
    rstar = Float64[]
    source = ComplexF64[]
    for line in lines[2:end]
        fields = strip.(split_csv_line(line))
        length(fields) >= maximum((irs, ire, iim)) || continue
        rs = parse(Float64, fields[irs])
        re = parse(Float64, fields[ire])
        im = parse(Float64, fields[iim])
        if isfinite(rs) && isfinite(re) && isfinite(im)
            push!(rstar, rs)
            push!(source, ComplexF64(re, im))
        end
    end
    length(rstar) >= 2 || error("need at least two finite CSV rows")
    return rstar, source
end

csv_value(x::AbstractFloat) = isnan(x) ? "NaN" : repr(Float64(x))
csv_value(x::Integer) = string(x)
csv_value(x::AbstractString) = x

function write_green_summary_csv(path::AbstractString, result::GreenAmplitudeResult)
    mkpath(dirname(path))
    row = (
        spin=result.cfg.spin,
        ell=result.cfg.ell,
        m=result.cfg.m,
        a=result.cfg.a,
        omega=result.cfg.omega,
        n_source_points=length(result.rstar),
        source_rstar_min=result.rstar[1],
        source_rstar_max=result.rstar[end],
        homogeneous_rsin=result.homogeneous_rsin,
        homogeneous_rsout=result.homogeneous_rsout,
        integration_rule=result.integration_rule,
        re_integral=real(result.integral),
        im_integral=imag(result.integral),
        re_lambda=real(result.lambda),
        im_lambda=imag(result.lambda),
        re_c0=real(result.c0),
        im_c0=imag(result.c0),
        re_Binc=real(result.binc),
        im_Binc=imag(result.binc),
        re_Xinf_over_c0=real(result.xinf_over_c0),
        im_Xinf_over_c0=imag(result.xinf_over_c0),
        abs_Xinf_over_c0=abs(result.xinf_over_c0),
        re_Zinf=real(result.zinf),
        im_Zinf=imag(result.zinf),
        abs_Zinf=abs(result.zinf),
        one_sided_dE_domega_over_mu2=result.one_sided_dE_domega_over_mu2,
        one_sided_dE_domega_over_mu2Enorm2=result.one_sided_dE_domega_over_mu2Enorm2,
    )
    names = collect(keys(row))
    open(path, "w") do io
        println(io, join(String.(names), ","))
        println(io, join((csv_value(getproperty(row, name)) for name in names), ","))
    end
    return path
end

function relative_difference(a, b)
    scale = max(abs(a), abs(b), eps(Float64))
    return abs(a - b) / scale
end

end
