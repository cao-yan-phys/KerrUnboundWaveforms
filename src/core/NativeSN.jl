module NativeSN

import ..KerrGeometry

export SNMode, delta, rplus, rminus, rstar, r_from_rstar, eta_coefficient, eta, potentials,
       infinity_coefficients, infinity_factor, horizon_coefficients, horizon_factor

struct SNMode
    a::Float64
    m::Int
    omega::Float64
    lambda::Float64
    eta_coefficients::NTuple{5, ComplexF64}
    numerator_coefficients::NTuple{7, ComplexF64}
    bracket_coefficients::NTuple{4, ComplexF64}
end

function SNMode(a::Real, m::Integer, omega::Real, lambda::Real)
    a = Float64(a)
    m = Int(m)
    omega = Float64(omega)
    lambda = Float64(lambda)
    q = a * omega - m
    eta_coefficients = ComplexF64[
        -12im * omega + lambda * (2 + lambda) - 12a * omega * q,
        8im * a * m * lambda + 8im * a^2 * omega * (3 - lambda),
        -24im * a * q + 12a^2 * (1 - 2q^2),
        24im * a^3 * q - 24a^2,
        12a^4,
    ]
    numerator_coefficients = ComplexF64[
        6a^4,
        -4im * a^3 * m + 4im * a^4 * omega - 24a^2,
        4a^3 * m * omega + 6im * a * m - 2a^4 * omega^2 +
            a^2 * (12 - 2m^2 + lambda - 6im * omega) + 24,
        -2im * a * m + 12im * a^2 * omega - 2 * (12 + lambda),
        4a * m * omega - 4a^2 * omega^2 + 6 + lambda - 18im * omega,
        8im * omega,
        -2omega^2,
    ]
    bracket_coefficients = ComplexF64[
        -2a^2,
        im * a * m - im * a^2 * omega + 3,
        -1,
        -im * omega,
    ]
    SNMode(
        a,
        m,
        omega,
        lambda,
        Tuple(eta_coefficients),
        Tuple(numerator_coefficients),
        Tuple(bracket_coefficients),
    )
end

function rplus(mode::SNMode)
    return KerrGeometry.rplus(mode.a)
end

rminus(mode::SNMode) = KerrGeometry.rminus(mode.a)

delta(mode::SNMode, r) = KerrGeometry.delta(mode.a, r)

rstar(mode::SNMode, r::Real) = KerrGeometry.rstar_from_r(mode.a, r)

r_from_rstar(mode::SNMode, target::Real) = KerrGeometry.r_from_rstar(mode.a, target)

function eta_coefficient(mode::SNMode, n::Integer)
    0 <= n <= 4 || return 0.0 + 0.0im
    return mode.eta_coefficients[n + 1]
end

function eta(mode::SNMode, r)
    c0, c1, c2, c3, c4 = mode.eta_coefficients
    c0 + c1 / r + c2 / r^2 + c3 / r^3 + c4 / r^4
end

function eta_prime(mode::SNMode, r)
    _, c1, c2, c3, c4 = mode.eta_coefficients
    -c1 / r^2 - 2c2 / r^3 - 3c3 / r^4 - 4c4 / r^5
end

function alpha_beta_data(mode::SNMode, r, Delta=delta(mode, r))
    a = mode.a
    n0, n1, n2, n3, n4, n5, n6 = mode.numerator_coefficients
    numerator = ((((((n6 * r + n5) * r + n4) * r + n3) * r + n2) * r + n1) * r + n0)
    numerator_prime = n1 + 2n2 * r + 3n3 * r^2 + 4n4 * r^3 + 5n5 * r^4 + 6n6 * r^5
    numerator_second = 2n2 + 6n3 * r + 12n4 * r^2 + 20n5 * r^3 + 30n6 * r^4
    b0, b1, b2, b3 = mode.bracket_coefficients
    bracket = ((b3 * r + b2) * r + b1) * r + b0
    bracket_prime = b1 + 2b2 * r + 3b3 * r^2
    bracket_second = 2b2 + 6b3 * r
    Delta_prime = 2r - 2
    denominator = r^2 * Delta
    denominator_prime = 2r * Delta + r^2 * Delta_prime
    denominator_second = 2Delta + 4r * Delta_prime + 2r^2
    alpha = numerator / denominator
    alpha_prime = numerator_prime / denominator - numerator * denominator_prime / denominator^2
    alpha_second = numerator_second / denominator -
        (2numerator_prime * denominator_prime + numerator * denominator_second) / denominator^2 +
        2numerator * denominator_prime^2 / denominator^3
    prefactor = 2Delta / r
    prefactor_prime = 2 * (1 - a^2 / r^2)
    prefactor_second = 4a^2 / r^3
    beta = prefactor * bracket
    beta_prime = prefactor_prime * bracket + prefactor * bracket_prime
    beta_second = prefactor_second * bracket + 2prefactor_prime * bracket_prime +
        prefactor * bracket_second
    return alpha, alpha_prime, alpha_second, beta, beta_prime, beta_second
end

function teukolsky_potential(mode::SNMode, r, Delta=delta(mode, r))
    a, m, omega, lambda = mode.a, mode.m, mode.omega, mode.lambda
    K = (r^2 + a^2) * omega - m * a
    lambda + 8im * omega * r - (K^2 + 4im * (r - 1) * K) / Delta
end

function potentials(mode::SNMode, r)
    Delta = delta(mode, r)
    Delta_prime = 2r - 2
    radial_factor = r^2 + mode.a^2
    radial_factor_prime = 2r
    alpha, alpha_prime, _, beta, beta_prime, beta_second = alpha_beta_data(mode, r)
    determinant = eta(mode, r)
    determinant_prime = eta_prime(mode, r)
    F1 = determinant_prime / determinant
    F = Delta * F1 / radial_factor
    transformed = alpha + beta_prime / Delta
    transformed_prime = 2alpha_prime + beta_second / Delta -
        beta_prime * Delta_prime / Delta^2
    U1 = teukolsky_potential(mode, r) +
        Delta^2 / beta * (transformed_prime - F1 * transformed)
    G = r * Delta / radial_factor^2 - 2 * (r - 1) / radial_factor
    G_prime = (Delta + r * Delta_prime) / radial_factor^2 -
        2r * radial_factor_prime * Delta / radial_factor^3 -
        2 / radial_factor + 2 * (r - 1) * radial_factor_prime / radial_factor^2
    U = Delta * U1 / radial_factor^2 + G^2 + Delta * G_prime / radial_factor -
        Delta * G * F1 / radial_factor
    return F, U
end

function explicit_infinity_coefficients(mode::SNMode, direction::Symbol)
    a, m, omega, lambda = mode.a, mode.m, mode.omega, mode.lambda
    c0 = eta_coefficient(mode, 0)
    if direction === :ingoing
        c1 = -im * (2 + lambda + 2a * m * omega) / 2
        c2 = (-lambda^2 - 2lambda * (1 + 2a * m * omega) -
            4omega * (-3im + a^2 * m^2 * omega + a * (m + 2im * m * omega))) / 8
        c3 = im * (
            lambda^3 + lambda^2 * (-2 + 6a * m * omega) +
            4lambda * (-2 - (3im + 2a * m) * omega +
                a * (2a + 6im * m + 3a * m^2) * omega^2) +
            8omega * (6im + a^3 * m * (2 + m^2) * omega^2 +
                3a^2 * omega * (-1 + 2im * m^2 * omega) -
                a * m * (6 + 3im * omega + 8omega^2))
        ) / 48
        return (1.0 + 0.0im, c1, c2, c3)
    elseif direction === :outgoing
        c1_numerator =
            -lambda^3 - 2lambda^2 * (2 + a * m * omega) +
            4lambda * (-1 + (3im - 8a * m) * omega + 7a^2 * omega^2) +
            24omega * (im - a^2 * (1 + m^2) * omega + a^3 * m * omega^2 +
                im * a * m * (im + omega))
        c1 = -im * c1_numerator / (2c0)
        c2_numerator =
            lambda^4 + 4lambda^3 * (1 + a * m * omega) +
            4lambda^2 * (1 + 2a * m * (7 - im * omega) * omega +
                a^2 * (-11 + m^2) * omega^2) -
            8a * lambda * omega * (-5a * omega - 15a * m^2 * omega +
                2m * (-4 + 4im * omega + 7a^2 * omega^2)) -
            48omega^2 * (-3 - a^3 * m * (-5 + m^2 + 2im * omega) * omega +
                a^4 * (-4 + m^2) * omega^2 + 2a * m * (im + omega) +
                im * a^2 * (-omega + 5im * m^2 + 3m^2 * omega))
        c2 = -c2_numerator / (8c0)
        c3_numerator =
            -lambda^5 - 6a * m * lambda^4 * omega -
            4lambda^3 * (-3 + 2a * m * (8 - 3im * omega) * omega +
                a^2 * (-13 + 3m^2) * omega^2) -
            8lambda^2 * (-2 + 2a^2 * (10 + 3m^2 * (6 - im * omega)) * omega^2 +
                a^3 * m * (-31 + m^2) * omega^3 -
                a * m * omega * (11 + 12im * omega + 8omega^2)) +
            16lambda * omega * (-9omega +
                3a^2 * (5 + m^2 * (-10 + 19im * omega) - 3im * omega) * omega -
                2a^3 * m * (-11 + 11m^2 + 21im * omega) * omega^2 +
                3a^4 * (-10 + 7m^2) * omega^3 +
                2a * m * (6 + 3im * omega + 13omega^2)) +
            96omega^2 * (6 + a * m * (-3 - 8im * omega) * omega +
                a^4 * (9 - m^4 + 2m^2 * (8 - 3im * omega)) * omega^2 +
                a^5 * m * (-10 + m^2) * omega^3 +
                a^3 * m * omega * (-9 + m^2 * (-12 + 7im * omega) + 5im * omega -
                    8omega^2) +
                a^2 * (-3im * omega + m^2 * (6 + 9im * omega + 14omega^2)))
        c3 = im * c3_numerator / (48c0)
        return (1.0 + 0.0im, c1, c2, c3)
    end
    throw(ArgumentError("direction must be :ingoing or :outgoing"))
end

function infinity_factor(mode::SNMode, r::Real, direction::Symbol; order::Int=3)
    order >= 0 || throw(ArgumentError("order must be nonnegative"))
    mode.omega != 0 || throw(DomainError(mode.omega, "static modes are not covered"))
    coefficients = infinity_coefficients(mode, direction; order=order)
    factor = zero(ComplexF64)
    derivative = zero(ComplexF64)
    for j in 0:order
        coefficient = coefficients[j + 1]
        factor += coefficient / (mode.omega * r)^j
        j == 0 || (derivative -= j * coefficient / (mode.omega^j * r^(j + 1)))
    end
    return factor, derivative
end

struct LaurentSeries
    lower::Int
    coefficients::Vector{ComplexF64}
end

series_upper(series::LaurentSeries) = series.lower + length(series.coefficients) - 1

function series_constant(value::Number, lower::Int, upper::Int)
    lower <= 0 <= upper || throw(ArgumentError("series range must contain zero"))
    coefficients = zeros(ComplexF64, upper - lower + 1)
    coefficients[1 - lower] = value
    LaurentSeries(lower, coefficients)
end

function series_power(power::Int, value::Number, lower::Int, upper::Int)
    lower <= power <= upper || throw(ArgumentError("power lies outside series range"))
    coefficients = zeros(ComplexF64, upper - lower + 1)
    coefficients[power - lower + 1] = value
    LaurentSeries(lower, coefficients)
end

function series_coefficient(series::LaurentSeries, power::Int)
    series.lower <= power <= series_upper(series) || return 0.0 + 0.0im
    return series.coefficients[power - series.lower + 1]
end

function matching_series_ranges(left::LaurentSeries, right::LaurentSeries)
    left.lower == right.lower && length(left.coefficients) == length(right.coefficients) ||
        throw(ArgumentError("Laurent-series ranges differ"))
end

function Base.:+(left::LaurentSeries, right::LaurentSeries)
    matching_series_ranges(left, right)
    LaurentSeries(left.lower, left.coefficients + right.coefficients)
end

function Base.:-(left::LaurentSeries, right::LaurentSeries)
    matching_series_ranges(left, right)
    LaurentSeries(left.lower, left.coefficients - right.coefficients)
end

Base.:-(series::LaurentSeries) = LaurentSeries(series.lower, -series.coefficients)

function Base.:+(series::LaurentSeries, value::Number)
    out = copy(series.coefficients)
    out[1 - series.lower] += value
    LaurentSeries(series.lower, out)
end

Base.:+(value::Number, series::LaurentSeries) = series + value
Base.:-(series::LaurentSeries, value::Number) = series + (-value)
Base.:-(value::Number, series::LaurentSeries) = (-series) + value
Base.:*(series::LaurentSeries, value::Number) = LaurentSeries(series.lower, series.coefficients .* value)
Base.:*(value::Number, series::LaurentSeries) = series * value
Base.:/(series::LaurentSeries, value::Number) = series * inv(value)

function Base.:*(left::LaurentSeries, right::LaurentSeries)
    matching_series_ranges(left, right)
    lower = left.lower
    upper = series_upper(left)
    out = zeros(ComplexF64, length(left.coefficients))
    for i in eachindex(left.coefficients)
        left_value = left.coefficients[i]
        iszero(left_value) && continue
        left_power = lower + i - 1
        for j in eachindex(right.coefficients)
            power = left_power + lower + j - 1
            lower <= power <= upper || continue
            out[power - lower + 1] += left_value * right.coefficients[j]
        end
    end
    LaurentSeries(lower, out)
end

function Base.inv(series::LaurentSeries)
    upper = series_upper(series)
    first_index = findfirst(value -> !iszero(value), series.coefficients)
    isnothing(first_index) && throw(DomainError(series, "cannot invert the zero series"))
    valuation = series.lower + first_index - 1
    maximum_order = upper + valuation
    maximum_order >= 0 || throw(DomainError(series, "series range is too short for inversion"))
    normalized_inverse = zeros(ComplexF64, maximum_order + 1)
    leading = series.coefficients[first_index]
    normalized_inverse[1] = inv(leading)
    for n in 1:maximum_order
        total = 0.0 + 0.0im
        for j in 1:n
            total += series_coefficient(series, valuation + j) * normalized_inverse[n - j + 1]
        end
        normalized_inverse[n + 1] = -total / leading
    end
    out = zeros(ComplexF64, length(series.coefficients))
    for n in 0:maximum_order
        power = -valuation + n
        series.lower <= power <= upper || continue
        out[power - series.lower + 1] = normalized_inverse[n + 1]
    end
    LaurentSeries(series.lower, out)
end

Base.:/(left::LaurentSeries, right::LaurentSeries) = left * inv(right)
Base.:/(value::Number, series::LaurentSeries) = series_constant(value, series.lower, series_upper(series)) / series

function Base.:^(series::LaurentSeries, power::Integer)
    power < 0 && return inv(series) ^ (-power)
    result = series_constant(1.0, series.lower, series_upper(series))
    factor = series
    exponent = power
    while exponent > 0
        isodd(exponent) && (result = result * factor)
        exponent = exponent >> 1
        exponent > 0 && (factor = factor * factor)
    end
    return result
end

function derivative_at_infinity(series::LaurentSeries)
    out = zeros(ComplexF64, length(series.coefficients))
    for i in eachindex(series.coefficients)
        power = series.lower + i - 1
        output_power = power + 1
        series.lower <= output_power <= series_upper(series) || continue
        out[output_power - series.lower + 1] -= power * series.coefficients[i]
    end
    LaurentSeries(series.lower, out)
end

function derivative_at_horizon(series::LaurentSeries)
    out = zeros(ComplexF64, length(series.coefficients))
    for i in eachindex(series.coefficients)
        power = series.lower + i - 1
        output_power = power - 1
        series.lower <= output_power <= series_upper(series) || continue
        out[output_power - series.lower + 1] += power * series.coefficients[i]
    end
    LaurentSeries(series.lower, out)
end

function series_potentials(mode::SNMode, r::LaurentSeries, Delta::LaurentSeries, derivative)
    Delta_prime = 2r - 2
    radial_factor = r^2 + mode.a^2
    radial_factor_prime = 2r
    alpha, alpha_prime, _, beta, beta_prime, beta_second = alpha_beta_data(mode, r, Delta)
    determinant = eta(mode, r)
    determinant_prime = derivative(determinant)
    F1 = determinant_prime / determinant
    F = Delta * F1 / radial_factor
    transformed = alpha + beta_prime / Delta
    transformed_prime = 2alpha_prime + beta_second / Delta - beta_prime * Delta_prime / Delta^2
    K = radial_factor * mode.omega - mode.m * mode.a
    teukolsky = mode.lambda + 8im * mode.omega * r -
        (K^2 + 4im * (r - 1) * K) / Delta
    U1 = teukolsky + Delta^2 / beta * (transformed_prime - F1 * transformed)
    G = r * Delta / radial_factor^2 - 2 * (r - 1) / radial_factor
    G_prime = (Delta + r * Delta_prime) / radial_factor^2 -
        2r * radial_factor_prime * Delta / radial_factor^3 -
        2 / radial_factor + 2 * (r - 1) * radial_factor_prime / radial_factor^2
    U = Delta * U1 / radial_factor^2 + G^2 + Delta * G_prime / radial_factor -
        Delta * G * F1 / radial_factor
    return F, U
end

function infinity_series_pq(mode::SNMode, direction::Symbol, order::Int)
    lower = -12
    upper = order + 24
    r = series_power(-1, 1.0, lower, upper)
    Delta = delta(mode, r)
    radial_factor = r^2 + mode.a^2
    D = Delta / radial_factor
    F, U = series_potentials(mode, r, Delta, derivative_at_infinity)
    sign = direction === :outgoing ? 1 : direction === :ingoing ? -1 :
        throw(ArgumentError("direction must be :ingoing or :outgoing"))
    P = (derivative_at_infinity(D) + sign * 2im * mode.omega - F) / D
    Q = (-mode.omega^2 - sign * im * mode.omega * F - U) / D^2
    return P, Q
end

function horizon_series_pq(mode::SNMode, order::Int)
    lower = -12
    upper = order + 24
    rp = rplus(mode)
    rm = rminus(mode)
    x = series_power(1, 1.0, lower, upper)
    r = x + rp
    Delta = x * (x + (rp - rm))
    radial_factor = r^2 + mode.a^2
    D = Delta / radial_factor
    F, U = series_potentials(mode, r, Delta, derivative_at_horizon)
    p = mode.omega - mode.m * mode.a / (2rp)
    P = (derivative_at_horizon(D) - 2im * p - F) / D
    Q = (-p^2 + im * p * F - U) / D^2
    return P, Q
end

function recursive_infinity_coefficients(mode::SNMode, direction::Symbol, order::Int)
    P, Q = infinity_series_pq(mode, direction, order)
    sign = direction === :outgoing ? 1 : direction === :ingoing ? -1 :
        throw(ArgumentError("direction must be :ingoing or :outgoing"))
    coefficients = zeros(ComplexF64, order + 1)
    coefficients[1] = 1.0 + 0.0im
    P0 = sign * 2im * mode.omega
    for j in 1:order
        numerator = j * (j - 1) * coefficients[j]
        for k in 1:j
            numerator += (series_coefficient(Q, k + 1) -
                (j - k) * series_coefficient(P, k)) * coefficients[j - k + 1]
        end
        coefficients[j + 1] = numerator / (P0 * j)
    end
    for j in 1:order
        coefficients[j + 1] *= mode.omega^j
    end
    return coefficients
end

function horizon_coefficients(mode::SNMode; order::Int=0)
    mode.omega != 0 || throw(DomainError(mode.omega, "static modes are not covered"))
    P, Q = horizon_series_pq(mode, order)
    coefficients = zeros(ComplexF64, order + 1)
    coefficients[1] = 1.0 + 0.0im
    P0 = series_coefficient(P, -1)
    Q0 = 0.0 + 0.0im
    for j in 1:order
        indicial = j * (j - 1) + P0 * j + Q0
        numerator = 0.0 + 0.0im
        for k in 0:(j - 1)
            numerator += (k * series_coefficient(P, j - k - 1) +
                series_coefficient(Q, j - k - 2)) * coefficients[k + 1]
        end
        coefficients[j + 1] = -numerator / indicial
    end
    for j in 1:order
        coefficients[j + 1] /= mode.omega^j
    end
    return coefficients
end

function infinity_coefficients(mode::SNMode, direction::Symbol; order::Int=3)
    order >= 0 || throw(ArgumentError("order must be nonnegative"))
    mode.omega != 0 || throw(DomainError(mode.omega, "static modes are not covered"))
    if order <= 3
        return collect(explicit_infinity_coefficients(mode, direction))[1:(order + 1)]
    end
    return recursive_infinity_coefficients(mode, direction, order)
end

function horizon_factor(mode::SNMode, r::Real; order::Int=0)
    order == 0 && return 1.0 + 0.0im, 0.0 + 0.0im
    coefficients = horizon_coefficients(mode; order=order)
    offset = r - rplus(mode)
    factor = 0.0 + 0.0im
    derivative = 0.0 + 0.0im
    for j in 0:order
        coefficient = coefficients[j + 1]
        factor += coefficient * (mode.omega * offset)^j
        j == 0 || (derivative += j * coefficient * mode.omega^j * offset^(j - 1))
    end
    return factor, derivative
end

end
