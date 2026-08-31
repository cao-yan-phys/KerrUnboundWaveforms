module AnalyticDisplacementMemory

using LinearAlgebra
using SpinWeightedSpheroidalHarmonics

export AnalyticFlKernel,
       analytic_fl_kernel,
       clear_fl_kernel_cache!,
       HARD_CODED_FL_MAX,
       F_l,
       single_particle_memory_mode,
       displacement_memory_mode

struct AnalyticFlKernel
    ell::Int
    normalization::Float64
    legendre_second::Vector{Float64}
    numerator::Vector{Float64}
    series_moments::Vector{Float64}
end

const FL_KERNEL_CACHE = Dict{Int, AnalyticFlKernel}()
const FL_VALUE_CACHE = Dict{Tuple{Int, UInt64}, Float64}()

const HARD_CODED_FL_MAX = 12

const HARD_CODED_A = (
    (-6.0, 10.0),
    (-30.0, 50.0, -16.0),
    (-105.0, 190.0, -81.0),
    (-315.0, 630.0, -343.0, 32.0),
    (-3465 / 4, 7665 / 4, -5103 / 4, 919 / 4),
    (-9009 / 4, 21945 / 4, -86499 / 20, 22923 / 20, -256 / 5),
    (-45045 / 8, 15015.0, -54747 / 4, 4779.0, -3781 / 8),
    (-109395 / 8, 79365 / 2, -164307 / 4, 248985 / 14,
     -155969 / 56, 512 / 7),
    (-2078505 / 64, 6527235 / 64, -3782493 / 32, 13676949 / 224,
     -5907275 / 448, 368961 / 448),
    (-4849845 / 64, 16397095 / 64, -10518937 / 32, 6311591 / 32,
     -73221005 / 1344, 7573735 / 1344, -2048 / 21),
    (-22309287 / 128, 40415375 / 64, -568725157 / 640, 97169501 / 160,
     -392983591 / 1920, 5779709 / 192, -497261 / 384),
)

const HARD_CODED_B = (
    (6.0, -12.0, 6.0),
    (30.0, -60.0, 30.0),
    (105.0, -225.0, 135.0, -15.0),
    (315.0, -735.0, 525.0, -105.0),
    (3465 / 4, -2205.0, 3675 / 2, -525.0, 105 / 4),
    (9009 / 4, -6237.0, 11907 / 2, -2205.0, 945 / 4),
    (45045 / 8, -135135 / 8, 72765 / 4, -33075 / 4, 11025 / 8,
     -315 / 8),
    (109395 / 8, -353925 / 8, 212355 / 4, -114345 / 4, 51975 / 8,
     -3465 / 8),
    (2078505 / 64, -3610035 / 32, 9555975 / 64, -1486485 / 16,
     1715175 / 64, -93555 / 32, 3465 / 64),
    (4849845 / 64, -9006855 / 32, 26072475 / 64, -4601025 / 16,
     6441435 / 64, -495495 / 32, 45045 / 64),
    (22309287 / 128, -88267179 / 128, 138705567 / 128,
     -109504395 / 128, 45090045 / 128, -9018009 / 128,
     693693 / 128, -9009 / 128),
)

const HARD_CODED_SERIES_LEADING = (
    8 / 5, 8 / 7, 16 / 21, 16 / 33, 128 / 429, 128 / 715,
    256 / 2431, 256 / 4199, 1024 / 29393, 1024 / 52003,
    2048 / 185725,
)

function trim_polynomial(coefficients)
    result = copy(coefficients)
    while length(result) > 1 && iszero(result[end])
        pop!(result)
    end
    return result
end

function add_polynomials(left, right)
    result = zeros(eltype(left), max(length(left), length(right)))
    result[1:length(left)] .+= left
    result[1:length(right)] .+= right
    return trim_polynomial(result)
end

function scale_polynomial(coefficients, factor)
    return trim_polynomial(factor .* coefficients)
end

function shift_polynomial(coefficients, degree::Int=1)
    degree >= 0 || error("polynomial shift must be nonnegative")
    return vcat(zeros(eltype(coefficients), degree), coefficients)
end

function multiply_polynomials(left, right)
    result = zeros(promote_type(eltype(left), eltype(right)),
                   length(left) + length(right) - 1)
    for i in eachindex(left), j in eachindex(right)
        result[i + j - 1] += left[i] * right[j]
    end
    return trim_polynomial(result)
end

function derivative_polynomial(coefficients, order::Int=1)
    order >= 0 || error("derivative order must be nonnegative")
    result = copy(coefficients)
    for _ in 1:order
        length(result) == 1 && return [zero(eltype(result))]
        result = [n * result[n + 1] for n in 1:(length(result) - 1)]
    end
    return trim_polynomial(result)
end

function legendre_coefficients(ell::Int)
    ell >= 0 || error("ell must be nonnegative")
    rational = Rational{BigInt}
    ell == 0 && return rational[1]
    ell == 1 && return rational[0, 1]
    previous_previous = rational[1]
    previous = rational[0, 1]
    for degree in 2:ell
        leading = scale_polynomial(
            shift_polynomial(previous), rational(2degree - 1, degree),
        )
        trailing = scale_polynomial(
            previous_previous, -rational(degree - 1, degree),
        )
        current = add_polynomials(leading, trailing)
        previous_previous, previous = previous, current
    end
    return previous
end

function build_fl_kernel(ell::Int)
    ell >= 2 || error("F_l is defined here only for ell >= 2")
    legendre = legendre_coefficients(ell)
    second = derivative_polynomial(legendre, 2)
    window = Rational{BigInt}[1, 0, -2, 0, 1]
    numerator = multiply_polynomials(second, window)
    series_moments = Float64[]
    for k in 0:511
        moment = zero(Rational{BigInt})
        for n0 in 0:(length(numerator) - 1)
            iseven(n0 + k) || continue
            moment += numerator[n0 + 1] * Rational{BigInt}(2, n0 + k + 1)
        end
        push!(series_moments, Float64(moment))
    end
    normalization = inv(sqrt(Float64((ell - 1) * ell * (ell + 1) * (ell + 2))))
    return AnalyticFlKernel(
        ell,
        normalization,
        Float64.(second),
        Float64.(numerator),
        series_moments,
    )
end

analytic_fl_kernel(ell::Int) = get!(FL_KERNEL_CACHE, ell) do
    build_fl_kernel(ell)
end

function clear_fl_kernel_cache!()
    empty!(FL_KERNEL_CACHE)
    empty!(FL_VALUE_CACHE)
    return nothing
end

monomial_integral(power::Int) = iseven(power) ? 2.0 / (power + 1) : 0.0

function evaluate_polynomial(coefficients, x)
    value = zero(x)
    for coefficient in Iterators.reverse(coefficients)
        value = muladd(value, x, coefficient)
    end
    return value
end

function hardcoded_fl_series(ell::Int, v::Float64;
                             rtol::Float64=4eps(Float64))
    index = ell - 1
    term = HARD_CODED_SERIES_LEADING[index] * v^(ell - 2)
    total = term
    n = ell
    for _ in 1:512
        term *= v^2 * (n + 4) * (n + 3) /
                ((n - ell + 2) * (n + ell + 3))
        total += term
        abs(term) <= rtol * max(abs(total), eps(Float64)) && break
        n += 2
    end
    normalization = inv(sqrt(Float64(
        (ell - 1) * ell * (ell + 1) * (ell + 2),
    )))
    return 2pi * normalization * (1 - v^2)^2 * total
end

function hardcoded_fl_closed(ell::Int, v::Float64)
    index = ell - 1
    u = v^2
    numerator = v * evaluate_polynomial(HARD_CODED_A[index], u) +
                atanh(v) * evaluate_polynomial(HARD_CODED_B[index], u)
    normalization = inv(sqrt(Float64(
        (ell - 1) * ell * (ell + 1) * (ell + 2),
    )))
    return pi * normalization * numerator / v^(ell + 3)
end

function hardcoded_F_l(ell::Int, velocity::Real)
    2 <= ell <= HARD_CODED_FL_MAX ||
        error("hard-coded F_l is available only for 2 <= ell <= $HARD_CODED_FL_MAX")
    v = Float64(velocity)
    0.0 <= v <= 1.0 || error("velocity must satisfy 0 <= v <= 1")
    v == 1.0 && return 4pi / sqrt(Float64(
        (ell - 1) * ell * (ell + 1) * (ell + 2),
    ))
    return v <= 0.72 ? hardcoded_fl_series(ell, v) :
                       hardcoded_fl_closed(ell, v)
end

function analytic_integral_series(kernel::AnalyticFlKernel, v::Float64;
                                  rtol::Float64=4eps(Float64))
    total = 0.0
    power_v = 1.0
    small_count = 0
    for k in 0:(length(kernel.series_moments) - 1)
        moment = kernel.series_moments[k + 1]
        term = power_v * moment
        total += term
        if k >= max(16, kernel.ell + 6) &&
           abs(term) <= rtol * max(abs(total), eps(Float64))
            small_count += 1
            small_count >= 4 && return total
        else
            small_count = 0
        end
        power_v *= v
    end
    error("analytic F_l series cache was exhausted for ell=$(kernel.ell), v=$v")
end

function divide_by_linear_denominator(coefficients, v::Float64)
    degree = length(coefficients) - 1
    quotient = zeros(Float64, degree)
    quotient[end] = -coefficients[end] / v
    for power in (degree - 1):-1:1
        quotient[power] = (quotient[power + 1] - coefficients[power + 1]) / v
    end
    return quotient
end

function analytic_integral_log(kernel::AnalyticFlKernel, v::Float64)
    quotient = divide_by_linear_denominator(kernel.numerator, v)
    polynomial_part = 0.0
    @inbounds for power0 in 0:(length(quotient) - 1)
        polynomial_part += quotient[power0 + 1] * monomial_integral(power0)
    end
    inverse_v = inv(v)
    second_at_inverse_v = evaluate_polynomial(kernel.legendre_second, inverse_v)
    remainder = ((v * v - 1) / (v * v))^2 * second_at_inverse_v
    logarithmic_part = remainder * (log1p(v) - log1p(-v)) / v
    return polynomial_part + logarithmic_part
end

function F_l(kernel::AnalyticFlKernel, velocity::Real)
    v = Float64(velocity)
    0.0 <= v <= 1.0 || error("velocity must satisfy 0 <= v <= 1")
    v == 1.0 && return 4pi * kernel.normalization
    integral = v <= 0.72 ?
               analytic_integral_series(kernel, v) :
               analytic_integral_log(kernel, v)
    return pi * kernel.normalization * integral
end

function F_l(ell::Int, velocity::Real)
    v = Float64(velocity)
    key = (ell, reinterpret(UInt64, v))
    return get!(FL_VALUE_CACHE, key) do
        2 <= ell <= HARD_CODED_FL_MAX ? hardcoded_F_l(ell, v) :
                                       F_l(analytic_fl_kernel(ell), v)
    end
end

function velocity_angles(velocity)
    length(velocity) == 3 || error("velocity must have three Cartesian components")
    vx, vy, vz = Float64.(velocity)
    speed = sqrt(vx * vx + vy * vy + vz * vz)
    speed <= 1 + 32eps(Float64) || error("velocity norm must not exceed one")
    speed == 0 && return (speed=0.0, theta=0.0, phi=0.0)
    return (
        speed=min(speed, 1.0),
        theta=acos(clamp(vz / speed, -1.0, 1.0)),
        phi=atan(vy, vx),
    )
end

function single_particle_memory_mode(ell::Int,
                                     m::Int,
                                     velocity;
                                     rest_mass::Real=1.0,
                                     particle_energy=nothing,
                                     radius::Real=1.0)
    ell >= 2 || error("ell must be at least 2")
    abs(m) <= ell || error("mode requires |m| <= ell")
    radius > 0 || error("radius must be positive")
    direction = velocity_angles(velocity)
    direction.speed == 0 && return 0.0 + 0.0im
    energy = if particle_energy === nothing
        direction.speed < 1 || error("particle_energy is required at v=1")
        Float64(rest_mass) / sqrt(1 - direction.speed^2)
    else
        Float64(particle_energy)
    end
    harmonic = spin_weighted_spherical_harmonic(0, ell, m)
    ylm = harmonic(direction.theta, direction.phi)
    return ComplexF64(
        4 / Float64(radius) * energy * direction.speed^2 *
        F_l(ell, direction.speed) * conj(ylm)
    )
end

function displacement_memory_mode(ell::Int,
                                  m::Int,
                                  incoming_velocity,
                                  outgoing_velocity;
                                  rest_mass::Real=1.0,
                                  particle_energy=nothing,
                                  radius::Real=1.0)
    common = (
        rest_mass=rest_mass,
        particle_energy=particle_energy,
        radius=radius,
    )
    outgoing = single_particle_memory_mode(
        ell, m, outgoing_velocity; common...,
    )
    incoming = single_particle_memory_mode(
        ell, m, incoming_velocity; common...,
    )
    return outgoing - incoming
end

end
