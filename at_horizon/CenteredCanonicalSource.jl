function centered_mathcalW_channel(
    r, x, dr_dx, v0, v1, v2,
    phase_values, phase_arguments, chi_prime_values;
    tail_order::Integer=3,
    tail_correction::Bool=true,
)
    n = length(r)
    n >= 4 || error("need at least four source points")
    v2 === nothing && error("v2 is required")
    tail_order >= 0 || error("tail_order must be nonnegative")

    _ccs_check_length(n, x, "x")
    _ccs_check_length(n, dr_dx, "dr_dx")
    _ccs_check_length(n, v2, "v2")
    _ccs_check_length(n, phase_values, "phase_values")
    _ccs_check_length(n, phase_arguments, "phase_arguments")
    _ccs_check_length(n, chi_prime_values, "chi_prime_values")
    v0 === nothing || _ccs_check_length(n, v0, "v0")
    v1 === nothing || _ccs_check_length(n, v1, "v1")

    rr = Float64.(r)
    xx = Float64.(x)
    jac = Float64.(dr_dx)
    phases = ComplexF64.(phase_values)
    phi = Float64.(phase_arguments)
    chip = ComplexF64.(chi_prime_values)
    vv2 = ComplexF64.(v2)
    vv0 = v0 === nothing ? nothing : ComplexF64.(v0)
    vv1 = v1 === nothing ? nothing : ComplexF64.(v1)

    _ccs_allfinite(rr) || error("r contains non-finite values")
    _ccs_allfinite(xx) || error("x contains non-finite values")
    _ccs_allfinite(jac) || error("dr_dx contains non-finite values")
    _ccs_allfinite(phi) || error("phase_arguments contains non-finite values")
    _ccs_allfinite(phases) || error("phase_values contains non-finite values")
    _ccs_allfinite(chip) || error("chi_prime_values contains non-finite values")
    _ccs_allfinite(vv2) || error("v2 contains non-finite values")
    vv0 === nothing || _ccs_allfinite(vv0) || error("v0 contains non-finite values")
    vv1 === nothing || _ccs_allfinite(vv1) || error("v1 contains non-finite values")

    @inbounds for i in 1:n-1
        rr[i+1] > rr[i] || error("r must be strictly increasing")
        xx[i+1] > xx[i] || error("x must be strictly increasing")
    end

    tail_correction && abs(chip[end]) <= 1.0e-12 &&
        error("outer phase derivative is too small for Abel tail")

    A = zeros(ComplexF64, n)
    B = zeros(ComplexF64, n)
    C = zeros(ComplexF64, n)

    if tail_correction
        if vv1 !== nothing
            A[end] = _ccs_fitted_tail(
                rr, vv1, phases, chip; order=Int(tail_order), centered=false,
            )
        end
        B[end] = _ccs_fitted_tail(
            rr, vv2, phases, chip; order=Int(tail_order), centered=false,
        )
        C[end] = _ccs_fitted_tail(
            rr, vv2, phases, chip; order=Int(tail_order), centered=true,
        )
    end

    @inbounds for i in (n-1):-1:1
        if vv1 !== nothing
            A[i] = A[i+1] + _ccs_segment_integral(
                xx[i], xx[i+1], jac[i], jac[i+1],
                vv1[i], vv1[i+1], phases[i], phases[i+1],
                phi[i], phi[i+1],
            )
        end

        segB = _ccs_segment_integral(
            xx[i], xx[i+1], jac[i], jac[i+1],
            vv2[i], vv2[i+1], phases[i], phases[i+1],
            phi[i], phi[i+1],
        )
        B[i] = B[i+1] + segB

        local_moment = _ccs_segment_first_moment(
            rr[i], rr[i+1], xx[i], xx[i+1], jac[i], jac[i+1],
            vv2[i], vv2[i+1], phases[i], phases[i+1],
            phi[i], phi[i+1],
        )
        C[i] = C[i+1] + (rr[i+1] - rr[i]) * B[i+1] + local_moment
    end

    W = Vector{ComplexF64}(undef, n)
    @inbounds for i in eachindex(W)
        value0 = vv0 === nothing ? 0.0 + 0.0im : vv0[i]
        W[i] = value0 + A[i] + C[i]
    end

    _ccs_allfinite(W) || error("centered source contains non-finite values")
    m0 = ComplexF64(B[1])
    m1 = ComplexF64(A[1] + C[1] + rr[1] * B[1])
    _ccs_allfinite((m0, m1)) || error("inner moments contain non-finite values")
    return (W=W, inner_m0=m0, inner_m1=m1)
end

@inline function _ccs_check_length(n, a, name)
    length(a) == n || error("$name length differs from r")
    return nothing
end

@inline _ccs_finite(z::Real) = isfinite(z)
@inline _ccs_finite(z::Complex) = isfinite(real(z)) && isfinite(imag(z))
_ccs_allfinite(a) = all(_ccs_finite, a)

function _ccs_phase_moments(delta::Float64, phase_ratio::ComplexF64)
    if abs(delta) <= 0.25
        q = 1im * delta
        f0 = 0.0 + 0.0im
        f1 = 0.0 + 0.0im
        f2 = 0.0 + 0.0im
        term = 1.0 + 0.0im  # q^k/k!
        for k in 0:28
            f0 += term / (k + 1)
            f1 += term / (k + 2)
            f2 += term / (k + 3)
            term *= q / (k + 1)
        end
        return ComplexF64(f0), ComplexF64(f1), ComplexF64(f2)
    end

    q = 1im * delta
    f0 = (phase_ratio - 1) / q
    f1 = phase_ratio / q - f0 / q
    f2 = phase_ratio / q - 2f1 / q
    return ComplexF64(f0), ComplexF64(f1), ComplexF64(f2)
end

function _ccs_segment_integral(
    x0::Float64, x1::Float64, jac0::Float64, jac1::Float64,
    value0::ComplexF64, value1::ComplexF64,
    phase0::ComplexF64, phase1::ComplexF64,
    phi0::Float64, phi1::Float64,
)
    width = x1 - x0
    width > 0 || error("source integration grid must increase")
    delta = phi1 - phi0
    ratio = phase1 * conj(phase0)
    f0, f1, _ = _ccs_phase_moments(delta, ratio)
    amp0 = value0 * jac0 * conj(phase0)
    amp1 = value1 * jac1 * conj(phase1)
    return width * phase0 * (amp0 * f0 + (amp1 - amp0) * f1)
end

function _ccs_segment_first_moment(
    r0::Float64, r1::Float64,
    x0::Float64, x1::Float64, jac0::Float64, jac1::Float64,
    value0::ComplexF64, value1::ComplexF64,
    phase0::ComplexF64, phase1::ComplexF64,
    phi0::Float64, phi1::Float64,
)
    width = x1 - x0
    dr = r1 - r0
    width > 0 && dr > 0 || error("source grids must increase")
    delta = phi1 - phi0
    ratio = phase1 * conj(phase0)
    _, f1, f2 = _ccs_phase_moments(delta, ratio)
    amp0 = value0 * jac0 * conj(phase0)
    amp1 = value1 * jac1 * conj(phase1)
    return width * dr * phase0 * (amp0 * f1 + (amp1 - amp0) * f2)
end

function _ccs_fit_point_count(r::Vector{Float64};
                              minimum_points::Int=384,
                              radial_window_fraction::Float64=0.025)
    n = length(r)
    cutoff = r[end] - radial_window_fraction * (r[end] - r[1])
    first = searchsortedfirst(r, cutoff)
    return min(n, max(minimum_points, n - first + 1))
end

function _ccs_fit_outer_taylor(r::Vector{Float64}, values::Vector{ComplexF64};
                               degree::Int, max_points::Int,
                               zero_constant::Bool=false)
    n = length(r)
    n == length(values) || error("tail-fit arrays have different lengths")
    count = min(n, max(max_points, degree + 1))
    first = n - count + 1
    scale = r[end] - r[first]
    scale > 0 || error("degenerate asymptotic fit window")
    fit_degree = min(degree, count - 1)
    first_power = zero_constant ? 1 : 0
    ncols = fit_degree - first_power + 1
    ncols >= 1 || return zeros(ComplexF64, degree + 1)
    u = (r[first:end] .- r[end]) ./ scale
    V = Matrix{Float64}(undef, count, ncols)
    @inbounds for i in 1:count
        power_value = u[i]^first_power
        for col in 1:ncols
            V[i, col] = power_value
            power_value *= u[i]
        end
    end
    coeff_scaled = _ccs_least_squares(V, values[first:end])
    coeff = zeros(ComplexF64, degree + 1)
    @inbounds for (col, p) in enumerate(first_power:fit_degree)
        coeff[p+1] = coeff_scaled[col] / scale^p
    end
    return coeff
end


function _ccs_least_squares(Ain::Matrix{Float64}, bin::Vector{ComplexF64})
    m, n = size(Ain)
    m >= n || error("tail polynomial fit is underdetermined")
    length(bin) == m || error("least-squares right-hand side length differs")
    A = copy(Ain)
    b = copy(bin)

    @inbounds for k in 1:n
        sigma2 = 0.0
        for i in k:m
            sigma2 += A[i,k]^2
        end
        sigma = sqrt(sigma2)
        sigma > 0 || error("singular tail polynomial fit")
        alpha = A[k,k] >= 0 ? -sigma : sigma

        v = Vector{Float64}(undef, m-k+1)
        for (j, i) in enumerate(k:m)
            v[j] = A[i,k]
        end
        v[1] -= alpha
        vnorm2 = sum(abs2, v)
        vnorm2 > 0 || error("degenerate Householder reflector")
        beta = 2.0 / vnorm2

        for j in k:n
            dotv = 0.0
            for (q, i) in enumerate(k:m)
                dotv += v[q] * A[i,j]
            end
            scale = beta * dotv
            for (q, i) in enumerate(k:m)
                A[i,j] -= scale * v[q]
            end
        end
        dotb = 0.0 + 0.0im
        for (q, i) in enumerate(k:m)
            dotb += v[q] * b[i]
        end
        scaleb = beta * dotb
        for (q, i) in enumerate(k:m)
            b[i] -= scaleb * v[q]
        end

        A[k,k] = alpha
        for i in k+1:m
            A[i,k] = 0.0
        end
    end

    x = zeros(ComplexF64, n)
    @inbounds for i in n:-1:1
        rhs = b[i]
        for j in i+1:n
            rhs -= A[i,j] * x[j]
        end
        abs(A[i,i]) > 1.0e-14 || error("singular triangular tail fit")
        x[i] = rhs / A[i,i]
    end
    return x
end

function _ccs_series_product(a::Vector{ComplexF64}, b::Vector{ComplexF64}, degree::Int)
    c = zeros(ComplexF64, degree + 1)
    @inbounds for i in 0:min(degree, length(a)-1)
        for j in 0:min(degree-i, length(b)-1)
            c[i+j+1] += a[i+1] * b[j+1]
        end
    end
    return c
end

function _ccs_reciprocal_series(a::Vector{ComplexF64}, degree::Int)
    abs(a[1]) > 1.0e-14 || error("singular outer phase derivative")
    b = zeros(ComplexF64, degree + 1)
    b[1] = inv(a[1])
    @inbounds for n in 1:degree
        s = 0.0 + 0.0im
        for k in 1:min(n, length(a)-1)
            s += a[k+1] * b[n-k+1]
        end
        b[n+1] = -s / a[1]
    end
    return b
end

function _ccs_derivative_series(a::Vector{ComplexF64}, degree::Int)
    b = zeros(ComplexF64, degree + 1)
    @inbounds for n in 0:min(degree, length(a)-2)
        b[n+1] = (n+1) * a[n+2]
    end
    return b
end

function _ccs_fitted_tail(
    r::Vector{Float64}, values::Vector{ComplexF64},
    phase_values::Vector{ComplexF64}, chi_prime_values::Vector{ComplexF64};
    order::Int, centered::Bool,
)
    n = length(r)
    degree = max(2 * order + 1, 5)
    count = _ccs_fit_point_count(r)
    first = n - count + 1
    R = r[end]

    amplitude = Vector{ComplexF64}(undef, count)
    @inbounds for (j, i) in enumerate(first:n)
        factor = centered ? (r[i] - R) : 1.0
        amplitude[j] = factor * values[i] * conj(phase_values[i])
    end
    amp_series = _ccs_fit_outer_taylor(
        r[first:end], amplitude; degree=degree, max_points=count,
        zero_constant=centered,
    )
    chi_series = _ccs_fit_outer_taylor(
        r[first:end], chi_prime_values[first:end]; degree=degree, max_points=count,
    )

    inv_i_chi = -1im .* _ccs_reciprocal_series(chi_series, degree)
    term = _ccs_series_product(amp_series, inv_i_chi, degree)
    tail_amp = -term[1]
    sign = 1.0
    for _ in 1:order
        term = _ccs_series_product(
            _ccs_derivative_series(term, degree), inv_i_chi, degree,
        )
        tail_amp += sign * term[1]
        sign = -sign
    end
    return ComplexF64(phase_values[end] * tail_amp)
end
