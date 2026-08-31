module UnboundGeodesicSNSources

export KerrParams,
       GeodesicConstants,
       BLState,
       SourceProjections,
       OrbitPiece,
       HorizonRadii,
       TrajectoryPoint,
       PieceTrajectory,
       AngularMode,
       TeukolskyATerms,
       swsh_angular_mode,
       delta,
       sigma,
       horizon_radii,
       outer_horizon,
       radial_potential,
       radial_potential_derivative,
       axis_sigma,
       axis_radial_q,
       axis_radial_kinematics,
       equatorial_chandrasekhar_constants,
       paper2003_equatorial_shat,
       k_over_delta_antiderivative,
       paper2003_equatorial_W_constant,
       paper2003_equatorial_L_shape,
       current_convention_equatorial_W_constant,
       current_convention_equatorial_L_shape,
       generic_equatorial_high_energy_shat,
       generic_equatorial_high_energy_W_constant,
       generic_equatorial_high_energy_L_shape,
       theta_potential,
       theta_turning_points,
       theta_allowed_interval,
       finite_outer_state_from_asymptote,
       asymptote_from_finite_outer_state,
       radial_turning_points,
       outer_radial_turning_point,
       classify_unbound_motion,
       orbit_pieces,
       four_velocity,
       source_projections,
       kerr_K,
       phase_integrand,
       phi_tilde_prime,
       xi_phase,
       chi_prime_r,
       radial_ode_rhs,
       radial_anomaly_rhs,
       integrate_orbit_piece,
       integrate_orbit_piece_anomaly,
       integrate_orbit_pieces,
       supports_radius,
       interpolated_point,
       source_support,
       branch_source_sum,
       rho_np,
       rhobar_np,
       l1p_l2p_s,
       ldag_apply,
       teukolsky_a_terms,
       fill_q_distribution_integrands!,
       q_distribution_coefficients,
       ldag,
       weighted_source

Base.@kwdef struct KerrParams
    a::Float64
end

Base.@kwdef struct GeodesicConstants
    energy::Float64
    lz::Float64
    carter_q::Float64
end

Base.@kwdef struct BLState
    r::Float64
    theta::Float64
    t::Float64 = 0.0
    phi::Float64 = 0.0
    ur_sign::Float64 = -1.0
    utheta_sign::Float64 = 1.0
end

struct SourceProjections
    N::Float64
    Mbar::ComplexF64
end

Base.@kwdef struct OrbitPiece
    label::Symbol
    r_start::Float64
    r_stop::Float64
    radial_sign::Float64
    theta_sign::Float64 = 1.0
    r_turn::Float64 = NaN
end

struct HorizonRadii
    rminus::Float64
    rplus::Float64
end

struct TrajectoryPoint
    label::Symbol
    r::Float64
    theta::Float64
    t::Float64
    phi::Float64
    chi::Float64
    ur_sign::Float64
    utheta_sign::Float64
    ut::Float64
    ur::Float64
    utheta::Float64
    uphi::Float64
    radial_potential::Float64
    theta_potential::Float64
    N::Float64
    Mbar::ComplexF64
end

struct PieceTrajectory
    piece::OrbitPiece
    points::Vector{TrajectoryPoint}
end

struct AngularMode{V,D1,D2,J,L}
    m::Int
    spheroid_c::Float64
    value::V
    derivative1::D1
    derivative2::D2
    joint_value_derivative::J
    theta_step::Float64
    lambda::L
end


function AngularMode(; m, spheroid_c, value, derivative=nothing,
                      derivative1=nothing, derivative2=nothing,
                      joint_value_derivative=nothing,
                      theta_step=1e-5, lambda=nothing)
    if derivative !== nothing
        derivative1 === nothing &&
            (derivative1 = theta -> ComplexF64(derivative(theta, 1)))
        derivative2 === nothing &&
            (derivative2 = theta -> ComplexF64(derivative(theta, 2)))
    end
    return AngularMode{
        typeof(value),typeof(derivative1),typeof(derivative2),
        typeof(joint_value_derivative),typeof(lambda),
    }(
        Int(m), Float64(spheroid_c), value, derivative1, derivative2,
        joint_value_derivative,
        Float64(theta_step), lambda,
    )
end

struct AngularMonomial
    coefficient::ComplexF64
    cos_power::Int
    sin_power::Int
end

function differentiate_angular_monomials(terms::Vector{AngularMonomial})
    result = AngularMonomial[]
    sizehint!(result, 2length(terms))
    for term in terms
        a = term.cos_power
        b = term.sin_power
        b != 0 && push!(result, AngularMonomial(
            0.5 * b * term.coefficient, a + 1, b - 1,
        ))
        a != 0 && push!(result, AngularMonomial(
            -0.5 * a * term.coefficient, a - 1, b + 1,
        ))
    end
    return result
end

Base.@propagate_inbounds @inline function evaluate_angular_monomials(
    terms::Vector{AngularMonomial}, theta,
)
    ct2 = cos(0.5theta)
    st2 = sin(0.5theta)
    value = 0.0 + 0.0im
    @inbounds for term in terms
        value += term.coefficient * ct2^term.cos_power * st2^term.sin_power
    end
    return value
end

Base.@propagate_inbounds @inline function evaluate_angular_value_derivative(
    terms::Vector{AngularMonomial}, theta,
)
    ct2 = cos(0.5theta)
    st2 = sin(0.5theta)
    value = 0.0 + 0.0im
    derivative = 0.0 + 0.0im
    @inbounds for term in terms
        a = term.cos_power
        b = term.sin_power
        coefficient = term.coefficient
        value += coefficient * ct2^a * st2^b
        b != 0 && (derivative +=
            0.5 * b * coefficient * ct2^(a + 1) * st2^(b - 1))
        a != 0 && (derivative -=
            0.5 * a * coefficient * ct2^(a - 1) * st2^(b + 1))
    end
    return value, derivative
end

function spectral_angular_monomials(harmonic)
    harmonic_module = parentmodule(typeof(harmonic))
    isdefined(harmonic_module, :_summation_term_prefactors) || return nothing
    isdefined(harmonic_module, :_swsh_prefactor) || return nothing
    prefactors_fun = getfield(harmonic_module, :_summation_term_prefactors)
    swsh_prefactor_fun = getfield(harmonic_module, :_swsh_prefactor)
    normalization = getproperty(harmonic, :normalization_const)
    coefficients = getproperty(harmonic, :coeffs)
    spherical_harmonics = getproperty(harmonic, :spherical_harmonics_l)
    terms = AngularMonomial[]

    for index in eachindex(coefficients)
        spherical = spherical_harmonics[index]
        spherical === nothing && continue
        s = getproperty(spherical, :s)
        ell = getproperty(spherical, :l)
        m = getproperty(spherical, :m)
        prefactors, log_normalization = prefactors_fun(s, ell, m)
        common = coefficients[index] * exp(log_normalization) *
                 swsh_prefactor_fun(s, ell, m) / normalization
        for r in max(0, m - s):min(ell - s, ell + m)
            coefficient = common * prefactors[r + 1]
            coefficient == 0 && continue
            push!(terms, AngularMonomial(
                ComplexF64(coefficient),
                2r + s - m,
                2ell - 2r - s + m,
            ))
        end
    end
    return terms
end

function swsh_angular_mode(harmonic, m, spheroid_c; theta_step=1e-5, phi=0.0)
    lambda = hasproperty(harmonic, :lambda) ? getproperty(harmonic, :lambda) : nothing
    value = theta -> harmonic(theta, phi)
    derivative1 = theta -> ComplexF64(harmonic(theta, phi; theta_derivative=1))
    derivative2 = theta -> ComplexF64(harmonic(theta, phi; theta_derivative=2))
    joint_value_derivative = nothing

    if hasproperty(harmonic, :method) && getproperty(harmonic, :method) == :spectral &&
       hasproperty(harmonic, :spherical_harmonics_l)
        terms0 = spectral_angular_monomials(harmonic)
        if terms0 !== nothing
            terms1 = differentiate_angular_monomials(terms0)
            terms2 = differentiate_angular_monomials(terms1)
            phase = cis(m * phi)
            value = theta -> phase * evaluate_angular_monomials(terms0, theta)
            derivative1 = theta -> phase * evaluate_angular_monomials(terms1, theta)
            derivative2 = theta -> phase * evaluate_angular_monomials(terms2, theta)
            joint_value_derivative = theta -> begin
                S, Sp = evaluate_angular_value_derivative(terms0, theta)
                return phase * S, phase * Sp
            end
        end
    end

    if hasproperty(harmonic, :method) && getproperty(harmonic, :method) == :chebyshev &&
       hasproperty(harmonic, :chebyshev_solution) &&
       hasproperty(harmonic, :normalization_const)
        cheb0 = getproperty(harmonic, :chebyshev_solution)
        harmonic_module = parentmodule(typeof(harmonic))
        if isdefined(harmonic_module, :differentiate)
            differentiate_fun = getfield(harmonic_module, :differentiate)
            cheb1 = differentiate_fun(cheb0, 1)
            cheb2 = differentiate_fun(cheb0, 2)
            normalization = getproperty(harmonic, :normalization_const)

            function evaluate_chebyshev(fun, theta)
                reflected_theta = mod(theta, 2pi)
                reflected_phi = phi
                if reflected_theta > pi
                    reflected_theta = 2pi - reflected_theta
                    reflected_phi += pi
                end
                return fun(reflected_theta) * cis(m * reflected_phi) / normalization
            end

            value = theta -> evaluate_chebyshev(cheb0, theta)
            derivative1 = theta -> evaluate_chebyshev(cheb1, theta)
            derivative2 = theta -> evaluate_chebyshev(cheb2, theta)
        end
    end
    return AngularMode(
        m=m,
        spheroid_c=Float64(spheroid_c),
        value=value,
        derivative1=derivative1,
        derivative2=derivative2,
        joint_value_derivative=joint_value_derivative,
        theta_step=theta_step,
        lambda=lambda,
    )
end

struct TeukolskyATerms
    nn0::ComplexF64
    nm0::ComplexF64
    nm1::ComplexF64
    mm0::ComplexF64
    mm1::ComplexF64
    mm2::ComplexF64
end

mutable struct TeukolskyATermsBuffer
    nn0::ComplexF64
    nm0::ComplexF64
    nm1::ComplexF64
    mm0::ComplexF64
    mm1::ComplexF64
    mm2::ComplexF64
end

TeukolskyATermsBuffer() = TeukolskyATermsBuffer(
    0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im,
    0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im,
)

delta(kerr::KerrParams, r) = r^2 - 2r + kerr.a^2
sigma(kerr::KerrParams, r, theta) = r^2 + kerr.a^2 * cos(theta)^2

function horizon_radii(kerr::KerrParams)
    abs(kerr.a) > 1 && error("Kerr horizon requires |a| <= 1")
    root = sqrt(1 - kerr.a^2)
    return HorizonRadii(1 - root, 1 + root)
end

outer_horizon(kerr::KerrParams) = horizon_radii(kerr).rplus

function radial_potential(kerr::KerrParams, c::GeodesicConstants, r)
    a = kerr.a
    E = c.energy
    Lz = c.lz
    Q = c.carter_q
    P = E * (r^2 + a^2) - a * Lz
    return P^2 - delta(kerr, r) * (r^2 + (Lz - a * E)^2 + Q)
end

function radial_potential_derivative(kerr::KerrParams, c::GeodesicConstants, r)
    a = kerr.a
    E = c.energy
    Lz = c.lz
    Q = c.carter_q
    P = E * (r^2 + a^2) - a * Lz
    Pp = 2 * E * r
    d = delta(kerr, r)
    dp = 2r - 2
    A = r^2 + (Lz - a * E)^2 + Q
    Ap = 2r
    return 2 * P * Pp - dp * A - d * Ap
end

axis_sigma(kerr::KerrParams, r) = r^2 + kerr.a^2

function axis_radial_q(kerr::KerrParams, energy, r)
    return sqrt(energy^2 - delta(kerr, r) / axis_sigma(kerr, r))
end

function axis_radial_kinematics(kerr::KerrParams, energy, r)
    sig = axis_sigma(kerr, r)
    q = axis_radial_q(kerr, energy, r)
    fp = 2 * (kerr.a^2 - r^2) / sig^2
    fpp = 4 * r * (r^2 - 3kerr.a^2) / sig^3
    qp = fp / (2q)
    qpp = fpp / (2q) - fp^2 / (4q^3)

    u = -q
    up = -qp
    upp = -qpp

    n = 1 / (energy + q)
    np = -qp / (energy + q)^2
    npp = -qpp / (energy + q)^2 + 2 * qp^2 / (energy + q)^3
    return u, up, upp, n, np, npp
end

equatorial_chandrasekhar_constants(kerr::KerrParams, energy) =
    GeodesicConstants(energy=energy, lz=kerr.a * energy, carter_q=0.0)

function paper2003_equatorial_shat(lambda, S, dS, kerr::KerrParams, omega, m)
    b = m - kerr.a * omega
    return (lambda / 2 - b^2) * S + b * dS
end

function paper2003_equatorial_shat(harmonic, kerr::KerrParams, omega, m;
                                   theta=pi / 2, phi=0.0)
    lambda = getproperty(harmonic, :lambda)
    S = harmonic(theta, phi)
    dS = harmonic(theta, phi; theta_derivative=1)
    return paper2003_equatorial_shat(lambda, S, dS, kerr, omega, m)
end

function k_over_delta_antiderivative(kerr::KerrParams, r, omega, m)
    hrs = horizon_radii(kerr)
    rp = hrs.rplus
    rm = hrs.rminus
    gap = rp - rm
    gap <= 0 && error("analytic K/Delta antiderivative requires |a| < 1")
    r <= rp && error("K/Delta antiderivative is real-valued only outside r_plus")

    rstar = r + 2 * rp / gap * log(r - rp) -
            2 * rm / gap * log(r - rm)
    delta_int = log((r - rp) / (r - rm)) / gap
    return omega * rstar - kerr.a * m * delta_int
end

paper2003_equatorial_W_constant(energy, omega, shat; particle_mass=1.0) =
    -particle_mass * energy * shat / omega^2

current_convention_equatorial_W_constant(energy, omega, shat; particle_mass=1.0) =
    current_convention_sign_relative_to_2003() *
    paper2003_equatorial_W_constant(energy, omega, shat; particle_mass=particle_mass)

function paper2003_equatorial_L_shape(kerr::KerrParams, r, omega, m, energy, shat;
                                      gamma0=1.0, particle_mass=1.0,
                                      phase_constant=0.0)
    g0 = gamma0 isa Function ? gamma0(r) : gamma0
    prefactor = g0 * delta(kerr, r) / (r^2 * (r^2 + kerr.a^2)^(3 / 2))
    phase = exp(-1im * (k_over_delta_antiderivative(kerr, r, omega, m) +
                        phase_constant))
    return paper2003_equatorial_W_constant(
        energy,
        omega,
        shat;
        particle_mass=particle_mass,
    ) * prefactor * phase
end

function current_convention_equatorial_L_shape(kerr::KerrParams, r, omega, m,
                                               energy, shat;
                                               gamma0=1.0, particle_mass=1.0,
                                               phase_constant=0.0)
    return current_convention_sign_relative_to_2003() *
           paper2003_equatorial_L_shape(
               kerr,
               r,
               omega,
               m,
               energy,
               shat;
               gamma0=gamma0,
               particle_mass=particle_mass,
               phase_constant=phase_constant,
           )
end

function generic_equatorial_high_energy_shat(kerr::KerrParams, mode::AngularMode;
                                             r1=10.0, r2=30.0)
    theta = pi / 2
    abs(r2 - r1) < sqrt(eps(Float64)) &&
        error("r1 and r2 must be distinct")
    l1 = l1p_l2p_s(kerr, r1, theta, mode)
    l2 = l1p_l2p_s(kerr, r2, theta, mode)
    return (l2 - l1) / (2 * (r2 - r1))
end

function generic_equatorial_high_energy_W_constant(kerr::KerrParams,
                                                   mode::AngularMode,
                                                   omega, energy;
                                                   convention::Symbol=:current,
                                                   particle_mass=1.0)
    shat = generic_equatorial_high_energy_shat(kerr, mode)
    if convention === :current
        return particle_mass * energy * shat / omega^2
    elseif convention === :paper2003
        return -particle_mass * energy * shat / omega^2
    end
    error("unknown source convention $convention")
end

function generic_equatorial_high_energy_L_shape(kerr::KerrParams,
                                                mode::AngularMode,
                                                r, omega, m, energy;
                                                convention::Symbol=:current,
                                                gamma0=1.0,
                                                particle_mass=1.0,
                                                phase_constant=0.0)
    g0 = gamma0 isa Function ? gamma0(r) : gamma0
    prefactor = g0 * delta(kerr, r) / (r^2 * (r^2 + kerr.a^2)^(3 / 2))
    phase = exp(-1im * (k_over_delta_antiderivative(kerr, r, omega, m) +
                        phase_constant))
    return generic_equatorial_high_energy_W_constant(
        kerr,
        mode,
        omega,
        energy;
        convention=convention,
        particle_mass=particle_mass,
    ) * prefactor * phase
end

const AXIS_LIMIT_TOLERANCE = 64eps(Float64)

is_north_axis(theta) = abs(theta) <= sqrt(eps(Float64))

function is_axis_radial_geodesic(kerr::KerrParams, c::GeodesicConstants)
    abs(c.lz) <= AXIS_LIMIT_TOLERANCE * max(1.0, abs(kerr.a * c.energy)) ||
        return false
    required_q = kerr.a^2 * (1 - c.energy^2)
    return abs(c.carter_q - required_q) <=
           AXIS_LIMIT_TOLERANCE * max(1.0, abs(required_q))
end

function theta_potential(kerr::KerrParams, c::GeodesicConstants, theta)
    if is_north_axis(theta)
        is_axis_radial_geodesic(kerr, c) ||
            error("an on-axis generic orbit requires Lz=0 and Q=a^2(1-E^2)")
        return 0.0
    end
    s = sin(theta)
    abs(s) < sqrt(eps(Float64)) &&
        error("the south-axis generic limit is not implemented")
    return c.carter_q -
           cos(theta)^2 * (kerr.a^2 * (1 - c.energy^2) + c.lz^2 / s^2)
end

function asymptotic_tail_transport(kerr::KerrParams,
                                   c::GeodesicConstants,
                                   theta::Float64,
                                   phi::Float64,
                                   radial_sign::Float64,
                                   theta_sign::Float64,
                                   x_start::Float64,
                                   x_stop::Float64;
                                   steps::Int=2048)
    steps >= 16 || error("asymptotic angular transport needs at least 16 steps")
    c.energy > 1 || error("asymptotic angular transport requires an unbound orbit")
    radial_sign in (-1.0, 1.0) || error("invalid radial sign")
    theta_sign in (-1.0, 1.0) || error("invalid polar sign")
    x_start >= 0 && x_stop >= 0 || error("compact radii must be nonnegative")

    if is_north_axis(theta)
        is_axis_radial_geodesic(kerr, c) ||
            error("an axial asymptotic state requires an axial radial geodesic")
        return (theta=0.0, phi=0.0, theta_sign=theta_sign)
    end

    lower, upper = theta_allowed_interval(kerr, c, theta)
    radial_square(x) = begin
        a = kerr.a
        A = c.energy * a^2 - a * c.lz
        B = (c.lz - a * c.energy)^2 + c.carter_q
        (c.energy + A * x^2)^2 -
        (1 - 2x + a^2 * x^2) * (1 + B * x^2)
    end
    rhs(theta_value, x, polar_sign) = begin
        theta_value = clamp(theta_value, lower, upper)
        theta_value == 0.0 && error("non-axial asymptotic transport reached the axis")
        sine = sin(theta_value)
        abs(sine) > sqrt(eps(Float64)) ||
            error("non-axial asymptotic transport reached the axis")
        radial_value = radial_square(x)
        radial_value >= -1.0e-13 ||
            error("asymptotic transport encountered a forbidden radial point")
        root_radial = sqrt(max(radial_value, 0.0))
        root_radial > 0 || error("asymptotic transport encountered a radial turning point")
        theta_value_potential = theta_potential(kerr, c, theta_value)
        theta_value_potential >= -1.0e-12 ||
            error("asymptotic transport encountered a forbidden polar point")
        dtheta = -polar_sign * sqrt(max(theta_value_potential, 0.0)) /
                 (radial_sign * root_radial)
        A = c.energy * kerr.a^2 - kerr.a * c.lz
        denominator = 1 - 2x + kerr.a^2 * x^2
        dphi_numerator =
            -(kerr.a * c.energy - c.lz / sine^2) +
            kerr.a * (c.energy + A * x^2) / denominator
        dphi = -dphi_numerator / (radial_sign * root_radial)
        return dtheta, dphi
    end
    reflect(theta_value, polar_sign) = begin
        for _ in 1:8
            if theta_value < lower
                theta_value = 2lower - theta_value
                polar_sign = -polar_sign
            elseif theta_value > upper
                theta_value = 2upper - theta_value
                polar_sign = -polar_sign
            else
                return theta_value, polar_sign
            end
        end
        error("asymptotic transport crossed too many polar turning points in one step")
    end

    h = (x_stop - x_start) / steps
    x = x_start
    polar_sign = theta_sign
    for _ in 1:steps
        k1theta, k1phi = rhs(theta, x, polar_sign)
        k2theta, k2phi = rhs(theta + h * k1theta / 2, x + h / 2, polar_sign)
        k3theta, k3phi = rhs(theta + h * k2theta / 2, x + h / 2, polar_sign)
        k4theta, k4phi = rhs(theta + h * k3theta, x + h, polar_sign)
        theta += h * (k1theta + 2k2theta + 2k3theta + k4theta) / 6
        phi += h * (k1phi + 2k2phi + 2k3phi + k4phi) / 6
        theta, polar_sign = reflect(theta, polar_sign)
        x += h
    end
    return (theta=theta, phi=phi, theta_sign=polar_sign)
end

function finite_outer_state_from_asymptote(kerr::KerrParams,
                                           c::GeodesicConstants,
                                           r_outer::Real,
                                           theta_infinity::Real,
                                           phi_infinity::Real,
                                           theta_sign::Real;
                                           steps::Int=2048)
    r_outer > 0 || error("outer radius must be positive")
    return asymptotic_tail_transport(
        kerr, c, Float64(theta_infinity), Float64(phi_infinity), -1.0,
        Float64(theta_sign), 0.0, inv(Float64(r_outer)); steps=steps,
    )
end

function asymptote_from_finite_outer_state(kerr::KerrParams,
                                           c::GeodesicConstants,
                                           r_outer::Real,
                                           theta_outer::Real,
                                           phi_outer::Real,
                                           radial_sign::Real,
                                           theta_sign::Real;
                                           steps::Int=2048)
    r_outer > 0 || error("outer radius must be positive")
    return asymptotic_tail_transport(
        kerr, c, Float64(theta_outer), Float64(phi_outer), Float64(radial_sign),
        Float64(theta_sign), inv(Float64(r_outer)), 0.0; steps=steps,
    )
end

function theta_turning_points(kerr::KerrParams, c::GeodesicConstants;
                              samples::Int=4000,
                              eps_theta::Float64=1e-6)
    samples < 3 && error("samples must be at least 3")
    lo = eps_theta
    hi = pi - eps_theta
    thetas = [lo + (hi - lo) * (i - 1) / (samples - 1) for i in 1:samples]
    vals = [theta_potential(kerr, c, theta) for theta in thetas]
    roots = Float64[]
    f(theta) = theta_potential(kerr, c, theta)

    for i in 1:(samples - 1)
        v1 = vals[i]
        v2 = vals[i + 1]
        if !isfinite(v1) || !isfinite(v2)
            continue
        elseif abs(v1) < 1e-12
            push!(roots, thetas[i])
        elseif sign(v1) != sign(v2)
            push!(roots, bisect_root(f, thetas[i], thetas[i + 1]))
        end
    end
    abs(vals[end]) < 1e-12 && push!(roots, thetas[end])

    sort!(roots)
    unique_roots = Float64[]
    for root in roots
        if isempty(unique_roots) || abs(root - unique_roots[end]) > 1e-8
            push!(unique_roots, root)
        end
    end
    return unique_roots
end

function theta_allowed_interval(kerr::KerrParams, c::GeodesicConstants,
                                theta0::Float64; kwargs...)
    theta_potential(kerr, c, theta0) < -1e-10 &&
        error("theta0=$theta0 lies outside the allowed theta interval")
    roots = theta_turning_points(kerr, c; kwargs...)
    lower = 1e-8
    upper = pi - 1e-8
    for root in roots
        if root < theta0
            lower = root
        elseif root > theta0
            upper = root
            break
        end
    end
    return (theta_min=lower, theta_max=upper, roots=roots)
end

function safe_sqrt_nonnegative(x; atol=1e-6)
    x < -atol && error("potential is negative: $x")
    return sqrt(max(x, 0.0))
end

function bisect_root(f, lo, hi; maxiters=200, atol=1e-13, rtol=1e-14)
    flo = f(lo)
    fhi = f(hi)
    flo == 0 && return lo
    fhi == 0 && return hi
    sign(flo) == sign(fhi) && error("root bracket has no sign change")

    left = lo
    right = hi
    fleft = flo
    for _ in 1:maxiters
        mid = (left + right) / 2
        fmid = f(mid)
        if abs(fmid) < atol || abs(right - left) < atol + rtol * abs(mid)
            return mid
        end
        if sign(fmid) == sign(fleft)
            left = mid
            fleft = fmid
        else
            right = mid
        end
    end
    return (left + right) / 2
end

function radial_turning_points(kerr::KerrParams, c::GeodesicConstants;
                               r_min::Float64=outer_horizon(kerr) + 1e-8,
                               r_max::Float64=1.0e4,
                               samples::Int=4000)
    r_min <= outer_horizon(kerr) &&
        error("r_min must be outside the outer horizon")
    r_max <= r_min && error("r_max must be larger than r_min")
    samples < 3 && error("samples must be at least 3")

    log_min = log(r_min)
    log_max = log(r_max)
    rs = [exp(log_min + (log_max - log_min) * (i - 1) / (samples - 1))
          for i in 1:samples]
    vals = [radial_potential(kerr, c, r) for r in rs]
    roots = Float64[]
    f(r) = radial_potential(kerr, c, r)

    for i in 1:(samples - 1)
        v1 = vals[i]
        v2 = vals[i + 1]
        if !isfinite(v1) || !isfinite(v2)
            continue
        elseif v1 == 0
            push!(roots, rs[i])
        elseif sign(v1) != sign(v2)
            push!(roots, bisect_root(f, rs[i], rs[i + 1]))
        end
    end
    vals[end] == 0 && push!(roots, rs[end])

    sort!(roots)
    unique_roots = Float64[]
    for root in roots
        if isempty(unique_roots) || abs(root - unique_roots[end]) > 1e-8 * max(1.0, abs(root))
            push!(unique_roots, root)
        end
    end
    return unique_roots
end

function outer_radial_turning_point(kerr::KerrParams, c::GeodesicConstants; kwargs...)
    roots = radial_turning_points(kerr, c; kwargs...)
    isempty(roots) && return nothing
    return roots[end]
end

function classify_unbound_motion(kerr::KerrParams, c::GeodesicConstants; kwargs...)
    turn = outer_radial_turning_point(kerr, c; kwargs...)
    return turn === nothing ? :plunge : :scattering
end

function orbit_pieces(kerr::KerrParams, c::GeodesicConstants;
                      kind::Symbol=:auto,
                      r_outer::Float64=200.0,
                      r_inner::Float64=outer_horizon(kerr) + 1e-5,
                      turn_buffer::Float64=1e-5,
                      regularized_turn::Bool=false,
                      theta_sign::Float64=1.0)
    motion = kind === :auto ? classify_unbound_motion(kerr, c; r_max=r_outer) : kind
    if motion === :plunge
        radial_potential(kerr, c, r_inner) < -1e-10 &&
            error("radial potential is negative at requested plunge inner radius")
        return [OrbitPiece(
            label=:plunge_in,
            r_start=r_outer,
            r_stop=r_inner,
            radial_sign=-1.0,
            theta_sign=theta_sign,
        )]
    elseif motion === :scattering
        turn = outer_radial_turning_point(kerr, c; r_max=r_outer)
        turn === nothing && error("scattering orbit requested but no radial turn was found")
        r_turn_work = regularized_turn ? turn : turn + turn_buffer * max(1.0, abs(turn))
        r_turn_work >= r_outer && error("turning point is too close to r_outer")
        return [
            OrbitPiece(
                label=:scatter_in,
                r_start=r_outer,
                r_stop=r_turn_work,
                radial_sign=-1.0,
                theta_sign=theta_sign,
                r_turn=turn,
            ),
            OrbitPiece(
                label=:scatter_out,
                r_start=r_turn_work,
                r_stop=r_outer,
                radial_sign=1.0,
                theta_sign=theta_sign,
                r_turn=turn,
            ),
        ]
    end
    error("unknown unbound orbit kind: $motion")
end

function four_velocity(kerr::KerrParams, c::GeodesicConstants, state::BLState)
    a = kerr.a
    r = state.r
    theta = state.theta
    if is_north_axis(theta)
        is_axis_radial_geodesic(kerr, c) ||
            error("an on-axis generic orbit requires Lz=0 and Q=a^2(1-E^2)")
        sig = axis_sigma(kerr, r)
        d = delta(kerr, r)
        q = axis_radial_q(kerr, c.energy, r)
        return (
            ut=c.energy * sig / d,
            ur=state.ur_sign * q,
            utheta=0.0,
            uphi=2 * a * c.energy * r / (sig * d),
            R=sig^2 * q^2,
            Theta=0.0,
            Sigma=sig,
            Delta=d,
        )
    end
    s = sin(theta)
    abs(s) < sqrt(eps(Float64)) &&
        error("the south-axis generic limit is not implemented")

    d = delta(kerr, r)
    sig = sigma(kerr, r, theta)
    E = c.energy
    Lz = c.lz
    P = E * (r^2 + a^2) - a * Lz
    radial_second_term = d * (r^2 + (Lz - a * E)^2 + c.carter_q)
    R = P^2 - radial_second_term
    Th = theta_potential(kerr, c, theta)

    radial_roundoff = 64 * eps(Float64) * (abs(P^2) + abs(radial_second_term))

    ut = (-a * (a * E * s^2 - Lz) + (r^2 + a^2) * P / d) / sig
    ur = state.ur_sign * safe_sqrt_nonnegative(
        R; atol=max(1e-6, radial_roundoff),
    ) / sig
    utheta = state.utheta_sign * safe_sqrt_nonnegative(Th; atol=1e-7) / sig
    uphi = (-(a * E - Lz / s^2) + a * P / d) / sig

    return (
        ut=ut,
        ur=ur,
        utheta=utheta,
        uphi=uphi,
        R=R,
        Theta=Th,
        Sigma=sig,
        Delta=d,
    )
end

function source_projections(kerr::KerrParams, state::BLState, u)
    a = kerr.a
    r = state.r
    theta = state.theta
    sig = sigma(kerr, r, theta)
    d = delta(kerr, r)
    s = sin(theta)

    if is_north_axis(theta)
        energy = u.ut * d / sig
        return SourceProjections(inv(energy + abs(u.ur)), 0.0 + 0.0im)
    end

    N = u.ut - a * s^2 * u.uphi + sig * u.ur / d
    Mbar = 1im * a * s * u.ut -
           1im * (r^2 + a^2) * s * u.uphi +
           sig * u.utheta
    return SourceProjections(N, ComplexF64(Mbar))
end

kerr_K(kerr::KerrParams, r, omega, m) =
    (r^2 + kerr.a^2) * omega - kerr.a * m

phase_integrand(kerr::KerrParams, r, omega, m) =
    kerr_K(kerr, r, omega, m) / delta(kerr, r)

rho_np(kerr::KerrParams, r, theta) = -1 / (r - 1im * kerr.a * cos(theta))
rhobar_np(kerr::KerrParams, r, theta) = conj(rho_np(kerr, r, theta))

function phi_tilde_prime(kerr::KerrParams, state::BLState, u)
    abs(u.ur) < sqrt(eps(Float64)) &&
        error("phi_tilde_prime is singular at radial turning points")
    return u.uphi / u.ur + kerr.a / delta(kerr, state.r)
end

function xi_phase(kerr::KerrParams, state::BLState, u, omega, m)
    return (kerr.a * omega * sin(state.theta)^2 - m) *
           phi_tilde_prime(kerr, state, u)
end

function chi_prime_r(kerr::KerrParams, state::BLState, u, proj::SourceProjections,
                     omega, m)
    abs(u.ur) < sqrt(eps(Float64)) &&
        error("chi_prime_r is singular at radial turning points")
    return omega * proj.N / u.ur + xi_phase(kerr, state, u, omega, m)
end

function radial_ode_rhs(kerr::KerrParams, c::GeodesicConstants, r, y,
                        radial_sign, theta_sign, omega, m)
    state = BLState(
        r=r,
        theta=y[2],
        t=y[1],
        phi=y[3],
        ur_sign=radial_sign,
        utheta_sign=theta_sign,
    )
    u = four_velocity(kerr, c, state)
    abs(u.ur) < sqrt(eps(Float64)) &&
        error("radial ODE is singular at a radial turning point")
    dt_dr = u.ut / u.ur
    dtheta_dr = u.utheta / u.ur
    dphi_dr = u.uphi / u.ur
    dchi_dr = omega * dt_dr - m * dphi_dr + phase_integrand(kerr, r, omega, m)
    return (dt_dr, dtheta_dr, dphi_dr, dchi_dr)
end

function radial_anomaly_rhs(kerr::KerrParams, c::GeodesicConstants, r_turn, z, y,
                            radial_sign, theta_sign, omega, m)
    if z < -1e-12
        error("radial anomaly z must be nonnegative")
    end
    z = max(z, 0.0)
    rp = radial_potential_derivative(kerr, c, r_turn)
    rp <= 0 && error("outer radial turn must have positive R'(r_turn)")

    if abs(z) < sqrt(eps(Float64))
        r = r_turn
        theta = y[2]
        state = BLState(
            r=r,
            theta=theta,
            t=y[1],
            phi=y[3],
            ur_sign=radial_sign,
            utheta_sign=theta_sign,
        )
        s = sin(theta)
        abs(s) < sqrt(eps(Float64)) &&
            error("generic radial_anomaly_rhs is singular on the axis")
        sig = sigma(kerr, r, theta)
        d = delta(kerr, r)
        E = c.energy
        Lz = c.lz
        P = E * (r^2 + kerr.a^2) - kerr.a * Lz
        ut = (-kerr.a * (kerr.a * E * s^2 - Lz) + (r^2 + kerr.a^2) * P / d) / sig
        Th = theta_potential(kerr, c, theta)
        utheta = theta_sign * safe_sqrt_nonnegative(Th; atol=1e-7) / sig
        uphi = (-(kerr.a * E - Lz / s^2) + kerr.a * P / d) / sig
        cancel = 2 * sig / (radial_sign * sqrt(rp))
        dt_dz = cancel * ut
        dtheta_dz = cancel * utheta
        dphi_dz = cancel * uphi
        dchi_dz = omega * dt_dz - m * dphi_dz
        return (dt_dz, dtheta_dz, dphi_dz, dchi_dz)
    end

    r = r_turn + z^2
    dr_dz = 2z
    dy_dr = radial_ode_rhs(kerr, c, r, y, radial_sign, theta_sign, omega, m)
    return (
        dy_dr[1] * dr_dz,
        dy_dr[2] * dr_dz,
        dy_dr[3] * dr_dz,
        dy_dr[4] * dr_dz,
    )
end

function polar_phase_parameters(kerr::KerrParams, c::GeodesicConstants,
                                theta0, theta_sign, theta_bounds)
    zmax = max(cos(theta_bounds.theta_min)^2,
               cos(theta_bounds.theta_max)^2)
    zmax > 64eps(Float64) ||
        error("polar-phase regularization requires a non-equatorial orbit")
    umax = sqrt(zmax)
    sine_phase = clamp(cos(theta0) / umax, -1.0, 1.0)
    principal = asin(sine_phase)
    phase0 = theta_sign >= 0 ? principal : -pi - principal
    a2_one_minus_e2 = kerr.a^2 * (1 - c.energy^2)
    return (
        zmax=zmax,
        umax=umax,
        phase0=phase0,
        q_over_zmax=c.carter_q / zmax,
        a2_one_minus_e2=a2_one_minus_e2,
    )
end

function theta_from_polar_phase(phase, polar)
    cosine_theta = clamp(polar.umax * sin(phase), -1.0, 1.0)
    return acos(cosine_theta)
end

polar_sign_from_phase(phase) = cos(phase) >= 0 ? 1.0 : -1.0

function radial_anomaly_polar_rhs(kerr::KerrParams,
                                  c::GeodesicConstants,
                                  r_turn,
                                  z,
                                  y,
                                  radial_sign,
                                  polar,
                                  omega,
                                  m)
    z < -1e-12 && error("radial anomaly z must be nonnegative")
    z = max(z, 0.0)
    r = r_turn + z^2
    phase = y[2]
    theta = theta_from_polar_phase(phase, polar)
    theta_sign = polar_sign_from_phase(phase)
    state = BLState(
        r=r,
        theta=theta,
        t=y[1],
        phi=y[3],
        ur_sign=radial_sign,
        utheta_sign=theta_sign,
    )
    velocity = four_velocity(kerr, c, state)
    radial_derivative = radial_potential_derivative(kerr, c, r_turn)
    radial_derivative > 0 ||
        error("outer radial turn must have positive R'(r_turn)")
    dlambda_dz = if z < sqrt(eps(Float64))
        2 / (radial_sign * sqrt(radial_derivative))
    else
        radial_value = radial_potential(kerr, c, r)
        2z / (radial_sign * sqrt(max(radial_value, 0.0)))
    end
    sig = sigma(kerr, r, theta)
    dt_dz = sig * velocity.ut * dlambda_dz
    dphi_dz = sig * velocity.uphi * dlambda_dz
    polar_rate = sqrt(max(
        polar.q_over_zmax -
        polar.a2_one_minus_e2 * polar.zmax * sin(phase)^2,
        0.0,
    ))
    dphase_dz = -polar_rate * dlambda_dz
    dchi_dz = omega * dt_dz - m * dphi_dz +
                phase_integrand(kerr, r, omega, m) * 2z
    return (dt_dz, dphase_dz, dphi_dz, dchi_dz)
end

function add_scaled(y, scale, dy)
    return (
        y[1] + scale * dy[1],
        y[2] + scale * dy[2],
        y[3] + scale * dy[3],
        y[4] + scale * dy[4],
    )
end

function rk4_step(kerr::KerrParams, c::GeodesicConstants, r, y, h,
                  radial_sign, theta_sign, omega, m)
    k1 = radial_ode_rhs(kerr, c, r, y, radial_sign, theta_sign, omega, m)
    k2 = radial_ode_rhs(kerr, c, r + h / 2, add_scaled(y, h / 2, k1),
                        radial_sign, theta_sign, omega, m)
    k3 = radial_ode_rhs(kerr, c, r + h / 2, add_scaled(y, h / 2, k2),
                        radial_sign, theta_sign, omega, m)
    k4 = radial_ode_rhs(kerr, c, r + h, add_scaled(y, h, k3),
                        radial_sign, theta_sign, omega, m)
    return (
        y[1] + h * (k1[1] + 2k2[1] + 2k3[1] + k4[1]) / 6,
        y[2] + h * (k1[2] + 2k2[2] + 2k3[2] + k4[2]) / 6,
        y[3] + h * (k1[3] + 2k2[3] + 2k3[3] + k4[3]) / 6,
        y[4] + h * (k1[4] + 2k2[4] + 2k3[4] + k4[4]) / 6,
    )
end

function rk4_step_anomaly(kerr::KerrParams, c::GeodesicConstants, r_turn, z, y, h,
                          radial_sign, theta_sign, omega, m)
    k1 = radial_anomaly_rhs(kerr, c, r_turn, z, y, radial_sign, theta_sign, omega, m)
    k2 = radial_anomaly_rhs(kerr, c, r_turn, z + h / 2, add_scaled(y, h / 2, k1),
                            radial_sign, theta_sign, omega, m)
    k3 = radial_anomaly_rhs(kerr, c, r_turn, z + h / 2, add_scaled(y, h / 2, k2),
                            radial_sign, theta_sign, omega, m)
    k4 = radial_anomaly_rhs(kerr, c, r_turn, z + h, add_scaled(y, h, k3),
                            radial_sign, theta_sign, omega, m)
    return (
        y[1] + h * (k1[1] + 2k2[1] + 2k3[1] + k4[1]) / 6,
        y[2] + h * (k1[2] + 2k2[2] + 2k3[2] + k4[2]) / 6,
        y[3] + h * (k1[3] + 2k2[3] + 2k3[3] + k4[3]) / 6,
        y[4] + h * (k1[4] + 2k2[4] + 2k3[4] + k4[4]) / 6,
    )
end

function rk4_step_anomaly_polar(kerr::KerrParams,
                                c::GeodesicConstants,
                                r_turn,
                                z,
                                y,
                                h,
                                radial_sign,
                                polar,
                                omega,
                                m)
    rhs(zvalue, yvalue) = radial_anomaly_polar_rhs(
        kerr, c, r_turn, zvalue, yvalue, radial_sign, polar, omega, m,
    )
    k1 = rhs(z, y)
    k2 = rhs(z + h / 2, add_scaled(y, h / 2, k1))
    k3 = rhs(z + h / 2, add_scaled(y, h / 2, k2))
    k4 = rhs(z + h, add_scaled(y, h, k3))
    return (
        y[1] + h * (k1[1] + 2k2[1] + 2k3[1] + k4[1]) / 6,
        y[2] + h * (k1[2] + 2k2[2] + 2k3[2] + k4[2]) / 6,
        y[3] + h * (k1[3] + 2k2[3] + 2k3[3] + k4[3]) / 6,
        y[4] + h * (k1[4] + 2k2[4] + 2k3[4] + k4[4]) / 6,
    )
end

function mixed_radial_anomaly_nodes(z_start::Float64, z_stop::Float64, nsteps::Int)
    nsteps < 1 && error("nsteps must be positive")
    zlo = min(z_start, z_stop)
    zhi = max(z_start, z_stop)
    zlo == zhi && return [z_start, z_stop]

    npoints = nsteps + 1
    n_near = max(2, cld(npoints, 2))
    n_far = max(2, npoints - n_near + 1)
    z_near = collect(range(zlo, zhi; length=n_near))
    z_far = [u^(1 / 3) for u in range(zlo^3, zhi^3; length=n_far)]
    z_all = sort!(vcat(z_near, z_far))

    nodes = Float64[]
    tol = 1e-12 * max(1.0, zhi)
    for z in z_all
        if isempty(nodes) || z > nodes[end] + tol
            push!(nodes, z)
        end
    end
    nodes[1] = zlo
    nodes[end] = zhi
    z_start > z_stop && reverse!(nodes)
    return nodes
end

function reflect_theta(theta, theta_sign, theta_bounds)
    lo = theta_bounds.theta_min
    hi = theta_bounds.theta_max
    sign_now = theta_sign
    theta_now = theta
    turn_tol = 64 * eps(Float64) * max(1.0, abs(lo), abs(hi))
    for _ in 1:20
        if theta_now < lo
            theta_now = 2lo - theta_now
            sign_now = -sign_now
        elseif theta_now > hi
            theta_now = 2hi - theta_now
            sign_now = -sign_now
        elseif theta_now <= lo + turn_tol && sign_now < 0
            theta_now = lo + turn_tol
            sign_now = -sign_now
        elseif theta_now >= hi - turn_tol && sign_now > 0
            theta_now = hi - turn_tol
            sign_now = -sign_now
        else
            return theta_now, sign_now
        end
    end
    error("theta reflection did not converge: theta=$theta bounds=[$lo,$hi]")
end

function reflect_state_theta(y, theta_sign, theta_bounds)
    theta, next_sign = reflect_theta(y[2], theta_sign, theta_bounds)
    return (y[1], theta, y[3], y[4]), next_sign
end

function rk4_step_dynamic_theta(kerr::KerrParams, c::GeodesicConstants, r, y, h,
                                radial_sign, theta_sign, omega, m,
                                theta_bounds; depth::Int=0)
    try
        ytrial = rk4_step(kerr, c, r, y, h, radial_sign, theta_sign, omega, m)
        yref, next_sign = reflect_state_theta(ytrial, theta_sign, theta_bounds)
        theta_potential(kerr, c, yref[2]) < -1e-7 &&
            error("reflected theta is outside allowed region")
        return yref, next_sign
    catch err
        depth >= 24 && rethrow(err)
        half = h / 2
        ymid, mid_sign = rk4_step_dynamic_theta(
            kerr, c, r, y, half,
            radial_sign, theta_sign, omega, m, theta_bounds;
            depth=depth + 1,
        )
        return rk4_step_dynamic_theta(
            kerr, c, r + half, ymid, half,
            radial_sign, mid_sign, omega, m, theta_bounds;
            depth=depth + 1,
        )
    end
end

function rk4_step_anomaly_dynamic_theta(kerr::KerrParams, c::GeodesicConstants,
                                        r_turn, z, y, h,
                                        radial_sign, theta_sign, omega, m,
                                        theta_bounds; depth::Int=0)
    try
        ytrial = rk4_step_anomaly(kerr, c, r_turn, z, y, h,
                                  radial_sign, theta_sign, omega, m)
        yref, next_sign = reflect_state_theta(ytrial, theta_sign, theta_bounds)
        theta_potential(kerr, c, yref[2]) < -1e-7 &&
            error("reflected theta is outside allowed region")
        return yref, next_sign
    catch err
        depth >= 24 && rethrow(err)
        half = h / 2
        ymid, mid_sign = rk4_step_anomaly_dynamic_theta(
            kerr, c, r_turn, z, y, half,
            radial_sign, theta_sign, omega, m, theta_bounds;
            depth=depth + 1,
        )
        return rk4_step_anomaly_dynamic_theta(
            kerr, c, r_turn, z + half, ymid, half,
            radial_sign, mid_sign, omega, m, theta_bounds;
            depth=depth + 1,
        )
    end
end

function trajectory_point(kerr::KerrParams, c::GeodesicConstants, piece::OrbitPiece,
                          r, y)
    return trajectory_point(kerr, c, piece, r, y, piece.theta_sign)
end

function trajectory_point(kerr::KerrParams, c::GeodesicConstants, piece::OrbitPiece,
                          r, y, theta_sign)
    state = BLState(
        r=r,
        theta=y[2],
        t=y[1],
        phi=y[3],
        ur_sign=piece.radial_sign,
        utheta_sign=theta_sign,
    )
    u = four_velocity(kerr, c, state)
    proj = source_projections(kerr, state, u)
    return TrajectoryPoint(
        piece.label,
        r,
        y[2],
        y[1],
        y[3],
        y[4],
        piece.radial_sign,
        theta_sign,
        u.ut,
        u.ur,
        u.utheta,
        u.uphi,
        u.R,
        u.Theta,
        proj.N,
        proj.Mbar,
    )
end

function trajectory_point_from_polar_phase(kerr::KerrParams,
                                           c::GeodesicConstants,
                                           piece::OrbitPiece,
                                           r,
                                           y,
                                           polar)
    phase = y[2]
    theta = theta_from_polar_phase(phase, polar)
    theta_sign = polar_sign_from_phase(phase)
    standard_state = (y[1], theta, y[3], y[4])
    return trajectory_point(
        kerr, c, piece, r, standard_state, theta_sign,
    )
end

function integrate_orbit_piece(kerr::KerrParams, c::GeodesicConstants,
                               piece::OrbitPiece, omega, m;
                               t0::Float64=0.0,
                               theta0::Float64=pi / 2,
                               phi0::Float64=0.0,
                               chi0::Float64=0.0,
                               nsteps::Int=1000,
                               radial_nodes=nothing,
                               allow_theta_turns::Bool=false,
                               theta_bounds=nothing)
    nodes = if radial_nodes === nothing
        nsteps < 1 && error("nsteps must be positive")
        range(piece.r_start, piece.r_stop; length=nsteps + 1)
    else
        values = Float64.(collect(radial_nodes))
        length(values) >= 2 || error("radial_nodes needs at least two radii")
        isapprox(values[1], piece.r_start; atol=1e-10, rtol=1e-12) ||
            error("radial_nodes must start at piece.r_start")
        isapprox(values[end], piece.r_stop; atol=1e-10, rtol=1e-12) ||
            error("radial_nodes must end at piece.r_stop")
        direction = sign(piece.r_stop - piece.r_start)
        direction != 0 || error("orbit piece endpoints must differ")
        @inbounds for i in 2:length(values)
            direction * (values[i] - values[i - 1]) > 0 ||
                error("radial_nodes must be strictly monotone")
        end
        values
    end
    r = Float64(nodes[1])
    y = (t0, theta0, phi0, chi0)
    theta_sign = piece.theta_sign
    if allow_theta_turns && theta_bounds === nothing
        theta_bounds = theta_allowed_interval(kerr, c, theta0)
    end
    points = TrajectoryPoint[trajectory_point(kerr, c, piece, r, y, theta_sign)]
    for step in 1:(length(nodes) - 1)
        r_next = Float64(nodes[step + 1])
        h = r_next - r
        if allow_theta_turns
            y, theta_sign = rk4_step_dynamic_theta(
                kerr, c, r, y, h, piece.radial_sign, theta_sign,
                omega, m, theta_bounds,
            )
        else
            y = rk4_step(kerr, c, r, y, h, piece.radial_sign, theta_sign, omega, m)
        end
        r = r_next
        push!(points, trajectory_point(kerr, c, piece, r, y, theta_sign))
    end
    return PieceTrajectory(piece, points)
end

function integrate_orbit_piece_anomaly(kerr::KerrParams, c::GeodesicConstants,
                                       piece::OrbitPiece, omega, m;
                                       t0::Float64=0.0,
                                       theta0::Float64=pi / 2,
                                       phi0::Float64=0.0,
                                       chi0::Float64=0.0,
                                       nsteps::Int=1000,
                                       allow_theta_turns::Bool=false,
                                       theta_bounds=nothing)
    nsteps < 1 && error("nsteps must be positive")
    isfinite(piece.r_turn) || error("anomaly integration requires piece.r_turn")
    piece.r_start < piece.r_turn - 1e-12 &&
        error("piece.r_start lies inside the radial turn")
    piece.r_stop < piece.r_turn - 1e-12 &&
        error("piece.r_stop lies inside the radial turn")

    z_start = sqrt(max(piece.r_start - piece.r_turn, 0.0))
    z_stop = sqrt(max(piece.r_stop - piece.r_turn, 0.0))
    nodes = mixed_radial_anomaly_nodes(z_start, z_stop, nsteps)
    z = nodes[1]
    y = (t0, theta0, phi0, chi0)
    theta_sign = piece.theta_sign
    if allow_theta_turns && theta_bounds === nothing
        theta_bounds = theta_allowed_interval(kerr, c, theta0)
    end
    if allow_theta_turns && c.carter_q > 64eps(Float64)
        polar = polar_phase_parameters(
            kerr, c, theta0, piece.theta_sign, theta_bounds,
        )
        ypolar = (t0, polar.phase0, phi0, chi0)
        r = piece.r_turn + z^2
        points = TrajectoryPoint[
            trajectory_point_from_polar_phase(
                kerr, c, piece, r, ypolar, polar,
            ),
        ]
        for step in 1:(length(nodes) - 1)
            z_next = nodes[step + 1]
            h = z_next - z
            ypolar = rk4_step_anomaly_polar(
                kerr, c, piece.r_turn, z, ypolar, h,
                piece.radial_sign, polar, omega, m,
            )
            z = z_next
            r = piece.r_turn + max(z, 0.0)^2
            push!(points, trajectory_point_from_polar_phase(
                kerr, c, piece, r, ypolar, polar,
            ))
        end
        return PieceTrajectory(piece, points)
    end
    r = piece.r_turn + z^2
    points = TrajectoryPoint[trajectory_point(kerr, c, piece, r, y, theta_sign)]
    for step in 1:(length(nodes) - 1)
        z_next = nodes[step + 1]
        h = z_next - z
        if allow_theta_turns
            y, theta_sign = rk4_step_anomaly_dynamic_theta(
                kerr, c, piece.r_turn, z, y, h,
                piece.radial_sign, theta_sign, omega, m, theta_bounds,
            )
        else
            y = rk4_step_anomaly(
                kerr, c, piece.r_turn, z, y, h,
                piece.radial_sign, theta_sign, omega, m,
            )
        end
        z = z_next
        r = piece.r_turn + max(z, 0.0)^2
        push!(points, trajectory_point(kerr, c, piece, r, y, theta_sign))
    end
    return PieceTrajectory(piece, points)
end

function should_use_anomaly(piece::OrbitPiece)
    !isfinite(piece.r_turn) && return false
    return isapprox(piece.r_start, piece.r_turn; atol=1e-10, rtol=1e-10) ||
           isapprox(piece.r_stop, piece.r_turn; atol=1e-10, rtol=1e-10)
end

function integrate_orbit_pieces(kerr::KerrParams, c::GeodesicConstants,
                                pieces::Vector{OrbitPiece}, omega, m;
                                t0::Float64=0.0,
                                theta0::Float64=pi / 2,
                                phi0::Float64=0.0,
                                chi0::Float64=0.0,
                                nsteps_per_piece::Int=1000,
                                regularize_turns::Bool=false,
                                allow_theta_turns::Bool=false,
                                radial_nodes_per_piece=nothing)
    radial_nodes_per_piece === nothing ||
        length(radial_nodes_per_piece) == length(pieces) ||
        error("radial_nodes_per_piece must match the orbit-piece count")
    trajectories = PieceTrajectory[]
    y0 = (t=t0, theta=theta0, phi=phi0, chi=chi0)
    theta_sign0 = pieces[1].theta_sign
    theta_bounds = allow_theta_turns ? theta_allowed_interval(kerr, c, theta0) : nothing
    for (piece_index, piece) in enumerate(pieces)
        piece_with_theta = OrbitPiece(
            label=piece.label,
            r_start=piece.r_start,
            r_stop=piece.r_stop,
            radial_sign=piece.radial_sign,
            theta_sign=theta_sign0,
            r_turn=piece.r_turn,
        )
        traj = if regularize_turns && should_use_anomaly(piece)
            integrate_orbit_piece_anomaly(
                kerr, c, piece_with_theta, omega, m;
                t0=y0.t,
                theta0=y0.theta,
                phi0=y0.phi,
                chi0=y0.chi,
                nsteps=nsteps_per_piece,
                allow_theta_turns=allow_theta_turns,
                theta_bounds=theta_bounds,
            )
        else
            integrate_orbit_piece(
                kerr, c, piece_with_theta, omega, m;
                t0=y0.t,
                theta0=y0.theta,
                phi0=y0.phi,
                chi0=y0.chi,
                nsteps=nsteps_per_piece,
                radial_nodes=radial_nodes_per_piece === nothing ?
                             nothing : radial_nodes_per_piece[piece_index],
                allow_theta_turns=allow_theta_turns,
                theta_bounds=theta_bounds,
            )
        end
        push!(trajectories, traj)
        last_point = traj.points[end]
        y0 = (
            t=last_point.t,
            theta=last_point.theta,
            phi=last_point.phi,
            chi=last_point.chi,
        )
        theta_sign0 = last_point.utheta_sign
    end
    return trajectories
end

lerp(a, b, x) = a + x * (b - a)

function supports_radius(traj::PieceTrajectory, r; atol=1e-10)
    r1 = traj.points[1].r
    r2 = traj.points[end].r
    lo = min(r1, r2) - atol
    hi = max(r1, r2) + atol
    return lo <= r <= hi
end

function interpolated_point(traj::PieceTrajectory, r)
    supports_radius(traj, r) || error("radius $r is outside trajectory piece")
    points = traj.points
    length(points) == 1 && return points[1]

    for i in 1:(length(points) - 1)
        p1 = points[i]
        p2 = points[i + 1]
        lo = min(p1.r, p2.r)
        hi = max(p1.r, p2.r)
        if lo <= r <= hi
            x = p2.r == p1.r ? 0.0 : (r - p1.r) / (p2.r - p1.r)
            utheta = lerp(p1.utheta, p2.utheta, x)
            utheta_sign = abs(utheta) > sqrt(eps(Float64)) ?
                sign(utheta) :
                (x < 0.5 ? p1.utheta_sign : p2.utheta_sign)
            return TrajectoryPoint(
                traj.piece.label,
                r,
                lerp(p1.theta, p2.theta, x),
                lerp(p1.t, p2.t, x),
                lerp(p1.phi, p2.phi, x),
                lerp(p1.chi, p2.chi, x),
                traj.piece.radial_sign,
                utheta_sign,
                lerp(p1.ut, p2.ut, x),
                lerp(p1.ur, p2.ur, x),
                utheta,
                lerp(p1.uphi, p2.uphi, x),
                lerp(p1.radial_potential, p2.radial_potential, x),
                lerp(p1.theta_potential, p2.theta_potential, x),
                lerp(p1.N, p2.N, x),
                lerp(p1.Mbar, p2.Mbar, x),
            )
        end
    end
    return points[end]
end

function source_support(trajectories::Vector{PieceTrajectory})
    isempty(trajectories) && error("empty trajectory list")
    lows = Float64[]
    highs = Float64[]
    for traj in trajectories
        r1 = traj.points[1].r
        r2 = traj.points[end].r
        push!(lows, min(r1, r2))
        push!(highs, max(r1, r2))
    end
    return (r_minimum=minimum(lows), r_maximum=maximum(highs))
end

function branch_source_sum(trajectories::Vector{PieceTrajectory}, r,
                           Wnn, Wnmb, Wmbmb)
    total = 0.0 + 0.0im
    contributors = 0
    for traj in trajectories
        supports_radius(traj, r) || continue
        point = interpolated_point(traj, r)
        proj = SourceProjections(point.N, point.Mbar)
        total += weighted_source(proj, Wnn(point), Wnmb(point), Wmbmb(point))
        contributors += 1
    end
    return (value=total, contributors=contributors)
end

function ldag(s_weight, m, spheroid_c, theta, S, dS)
    st = sin(theta)
    abs(st) < sqrt(eps(Float64)) &&
        error("Ldag is singular on the axis; use an axial angular limit")
    return dS - (m / st) * S + spheroid_c * st * S +
           s_weight * (cos(theta) / st) * S
end

function theta_derivative(mode::AngularMode, theta, f)
    h = min(mode.theta_step, 0.2 * max(1e-8, theta), 0.2 * max(1e-8, pi - theta))
    h <= 0 && error("theta derivative is singular on the polar axis")
    return (-f(theta + 2h) + 8f(theta + h) - 8f(theta - h) + f(theta - 2h)) / (12h)
end

function ldag_apply(mode::AngularMode, s_weight, theta, f)
    return ldag(
        s_weight,
        mode.m,
        mode.spheroid_c,
        theta,
        f(theta),
        theta_derivative(mode, theta, f),
    )
end

function local_shifted_state(kerr::KerrParams, c::GeodesicConstants,
                             state::BLState, r_new)
    if r_new == state.r
        return state
    end
    u = four_velocity(kerr, c, state)
    abs(u.ur) < 1e-10 &&
        error("cannot build local r derivative at a radial turning point")
    theta_new = state.theta + (u.utheta / u.ur) * (r_new - state.r)
    return BLState(
        r=r_new,
        theta=theta_new,
        t=state.t,
        phi=state.phi,
        ur_sign=state.ur_sign,
        utheta_sign=state.utheta_sign,
    )
end

function derivative_r(kerr::KerrParams, c::GeodesicConstants, state::BLState,
                      f; relstep=1e-5)
    h = relstep * max(1.0, abs(state.r))
    rplus = outer_horizon(kerr)
    left_forbidden = state.r - 2h <= rplus ||
                     radial_potential(kerr, c, state.r - h) < -1e-9 ||
                     radial_potential(kerr, c, state.r - 2h) < -1e-9
    if left_forbidden
        f0 = f(state)
        fp = f(local_shifted_state(kerr, c, state, state.r + h))
        fp2 = f(local_shifted_state(kerr, c, state, state.r + 2h))
        return (-3 * f0 + 4 * fp - fp2) / (2h)
    end
    sp1 = local_shifted_state(kerr, c, state, state.r + h)
    sm1 = local_shifted_state(kerr, c, state, state.r - h)
    sp2 = local_shifted_state(kerr, c, state, state.r + 2h)
    sm2 = local_shifted_state(kerr, c, state, state.r - 2h)
    return (-f(sp2) + 8 * f(sp1) - 8 * f(sm1) + f(sm2)) / (12h)
end

function radial_partial_rhobar2_rho_minus4(kerr::KerrParams, r, theta)
    rho = rho_np(kerr, r, theta)
    rhob = rhobar_np(kerr, r, theta)
    return 2 * rhob^3 * rho^(-4) - 4 * rhob^2 * rho^(-3)
end

function angular_S(mode::AngularMode, theta)
    return ComplexF64(mode.value(theta))
end

function angular_dS(mode::AngularMode, theta, order::Int)
    order == 0 && return angular_S(mode, theta)
    if order == 1 && mode.derivative1 !== nothing
        return ComplexF64(mode.derivative1(theta))
    end
    if order == 1
        return theta_derivative(mode, theta, t -> angular_S(mode, t))
    elseif order == 2
        mode.derivative2 !== nothing && mode.lambda === nothing &&
            return ComplexF64(mode.derivative2(theta))
        if mode.lambda !== nothing
            st = sin(theta)
            abs(st) < sqrt(eps(Float64)) &&
                error("angular ODE second derivative is singular on the polar axis")
            ct = cos(theta)
            c = mode.spheroid_c
            m = mode.m
            angular_A = mode.lambda - c^2 + 2 * m * c
            bracket = c^2 * ct^2 + 4 * c * ct + angular_A -
                      (m - 2 * ct)^2 / st^2 - 2
            return -ct / st * angular_dS(mode, theta, 1) -
                   bracket * angular_S(mode, theta)
        end
        return theta_derivative(
            mode,
            theta,
            t -> theta_derivative(mode, t, x -> angular_S(mode, x)),
        )
    end
    error("unsupported angular derivative order $order")
end

@inline function angular_values(mode::AngularMode, theta)
    if mode.joint_value_derivative === nothing
        S = angular_S(mode, theta)
        Sp = angular_dS(mode, theta, 1)
    else
        S, Sp = mode.joint_value_derivative(theta)
        S = ComplexF64(S)
        Sp = ComplexF64(Sp)
    end
    if mode.lambda !== nothing
        st = sin(theta)
        abs(st) < sqrt(eps(Float64)) &&
            error("angular ODE second derivative is singular on the polar axis")
        ct = cos(theta)
        c = mode.spheroid_c
        m = mode.m
        angular_A = mode.lambda - c^2 + 2 * m * c
        bracket = c^2 * ct^2 + 4 * c * ct + angular_A -
                  (m - 2 * ct)^2 / st^2 - 2
        Spp = -ct / st * Sp - bracket * S
    else
        Spp = angular_dS(mode, theta, 2)
    end
    return S, Sp, Spp
end

function teukolsky_a_terms_from_kinematics!(
    result::TeukolskyATermsBuffer,
    kerr::KerrParams, r, theta, N, Mbar, mode::AngularMode, omega,
)
    if is_north_axis(theta)
        mode.m == 0 || error("the generic north-axis limit requires m=0")
        abs(Mbar) <= 128eps(Float64) * max(1.0, abs(N)) ||
            error("the generic north-axis limit requires vanishing Mbar")
        mode.derivative2 === nothing &&
            error("the generic north-axis limit requires an angular second derivative")

        rho = rho_np(kerr, r, theta)
        rhob = rhobar_np(kerr, r, theta)
        Spp_axis = ComplexF64(mode.derivative2(0.0))
        l1l2rhoS_axis = 4 * Spp_axis / rho
        result.nn0 = ComplexF64(-0.5 * rho * rhob^2 * N^2 * l1l2rhoS_axis)
        result.nm0 = 0.0 + 0.0im
        result.nm1 = 0.0 + 0.0im
        result.mm0 = 0.0 + 0.0im
        result.mm1 = 0.0 + 0.0im
        result.mm2 = 0.0 + 0.0im
        return result
    end
    d = delta(kerr, r)
    dp = 2r - 2
    K = kerr_K(kerr, r, omega, mode.m)
    Kp = 2r * omega
    k_over_d = K / d
    k_over_d_prime = (Kp * d - K * dp) / d^2
    rho = rho_np(kerr, r, theta)
    rhob = rhobar_np(kerr, r, theta)
    S, Sp, Spp = angular_values(mode, theta)
    a2 = angular_operator_coeff(mode, 2, theta)
    L2S_value = Sp + a2 * S
    L1L2rhoS = l1p_l2p_s_from_values(kerr, r, theta, mode, S, Sp, Spp)
    sth = sin(theta)
    A = -1.0

    nn0 = A / 2 * rho * rhob^2 * N^2 * L1L2rhoS
    nm0 = A * rhob^2 * N * Mbar * (
        L2S_value * (1im * k_over_d - rho - rhob) -
        kerr.a * sth * S * k_over_d * (rho - rhob)
    )
    mm0 = A / 2 * rhob^2 * Mbar^2 * S * (
        -1im * k_over_d_prime - k_over_d^2 - 2im * rho * k_over_d
    )
    nm1 = A * rhob^2 * N * Mbar * (
        L2S_value + 1im * kerr.a * sth * (rho - rhob) * S
    )
    mm1 = A * rhob^2 * Mbar^2 * S * (1im * k_over_d - rho)
    mm2 = A / 2 * rhob^2 * Mbar^2 * S
    result.nn0 = ComplexF64(nn0)
    result.nm0 = ComplexF64(nm0)
    result.nm1 = ComplexF64(nm1)
    result.mm0 = ComplexF64(mm0)
    result.mm1 = ComplexF64(mm1)
    result.mm2 = ComplexF64(mm2)
    return result
end

function teukolsky_a_terms_from_kinematics(
    kerr::KerrParams, r, theta, N, Mbar, mode::AngularMode, omega,
)
    buffer = teukolsky_a_terms_from_kinematics!(
        TeukolskyATermsBuffer(), kerr, r, theta, N, Mbar, mode, omega,
    )
    return TeukolskyATerms(
        buffer.nn0, buffer.nm0, buffer.nm1,
        buffer.mm0, buffer.mm1, buffer.mm2,
    )
end

function teukolsky_a_terms(kerr::KerrParams, c::GeodesicConstants,
                           state::BLState, mode::AngularMode, omega)
    u = four_velocity(kerr, c, state)
    proj = source_projections(kerr, state, u)
    return teukolsky_a_terms_from_kinematics(
        kerr, state.r, state.theta, proj.N, proj.Mbar, mode, omega,
    )
end

function teukolsky_a_terms(kerr::KerrParams, c::GeodesicConstants,
                           point::TrajectoryPoint, mode::AngularMode, omega)
    return teukolsky_a_terms_from_kinematics(
        kerr, point.r, point.theta, point.N, point.Mbar, mode, omega,
    )
end


@inline distribution_q0(A0, A1, A2, F, Fp, Fpp) =
    F * A0 - Fp * A1 + Fpp * A2
@inline distribution_q1(A1, A2, F, Fp) = F * A1 - 2Fp * A2
@inline distribution_q2(A2, F) = F * A2

function q_distribution_coefficients(
    kerr::KerrParams, c::GeodesicConstants, mode::AngularMode,
    point::TrajectoryPoint, omega, m;
    t_origin::Float64=0.0, phi_origin::Float64=0.0,
)
    terms = teukolsky_a_terms(kerr, c, point, mode, omega)
    r = point.r
    d = delta(kerr, r)
    dp = 2r - 2
    K = kerr_K(kerr, r, omega, m)
    Kp = 2r * omega
    kappa = K / d
    kappap = (Kp * d - K * dp) / d^2
    chi = k_over_delta_antiderivative(kerr, r, omega, m)
    echi = cis(chi)
    F = -r^2 * echi
    Fp = -echi * (2r + 1im * r^2 * kappa)
    Fpp = -echi * (
        2 + 4im * r * kappa + 1im * r^2 * kappap - r^2 * kappa^2
    )
    orbit_phase = cis(
        omega * (point.t - t_origin) - m * (point.phi - phi_origin),
    )
    function transform(A0, A1, A2)
        return (
            q0=ComplexF64(orbit_phase * distribution_q0(A0, A1, A2, F, Fp, Fpp)),
            q1=ComplexF64(orbit_phase * distribution_q1(A1, A2, F, Fp)),
            q2=ComplexF64(orbit_phase * distribution_q2(A2, F)),
        )
    end
    nn = transform(terms.nn0, 0.0 + 0.0im, 0.0 + 0.0im)
    nm = transform(terms.nm0, terms.nm1, 0.0 + 0.0im)
    mm = transform(terms.mm0, terms.mm1, terms.mm2)
    total = (
        q0=nn.q0 + nm.q0 + mm.q0,
        q1=nn.q1 + nm.q1 + mm.q1,
        q2=nn.q2 + nm.q2 + mm.q2,
    )
    return (nn=nn, nm=nm, mm=mm, total=total)
end

function fill_q_distribution_integrands!(
    f2, g1, g2, h0, h1, h2, i,
    terms::TeukolskyATermsBuffer,
    kerr::KerrParams, mode::AngularMode, point::TrajectoryPoint,
    omega, m, orbit_phase, echi,
)
    teukolsky_a_terms_from_kinematics!(
        terms, kerr, point.r, point.theta, point.N, point.Mbar, mode, omega,
    )
    r = point.r
    d = delta(kerr, r)
    dp = 2r - 2
    K = kerr_K(kerr, r, omega, m)
    Kp = 2r * omega
    kappa = K / d
    kappap = (Kp * d - K * dp) / d^2

    F = -r^2 * echi
    Fp = -echi * (2r + 1im * r^2 * kappa)
    Fpp = -echi * (
        2 + 4im * r * kappa + 1im * r^2 * kappap - r^2 * kappa^2
    )
    proper_time_per_r = inv(abs(point.ur))

    f2[i] = ComplexF64(orbit_phase * distribution_q0(
        terms.nn0, 0.0 + 0.0im, 0.0 + 0.0im, F, Fp, Fpp,
    )) * proper_time_per_r

    g1[i] = -ComplexF64(orbit_phase * distribution_q1(
        terms.nm1, 0.0 + 0.0im, F, Fp,
    )) * proper_time_per_r
    g2[i] = ComplexF64(orbit_phase * distribution_q0(
        terms.nm0, terms.nm1, 0.0 + 0.0im, F, Fp, Fpp,
    )) * proper_time_per_r

    h0[i] = ComplexF64(orbit_phase * distribution_q2(terms.mm2, F)) * proper_time_per_r
    h1[i] = -ComplexF64(orbit_phase * distribution_q1(
        terms.mm1, terms.mm2, F, Fp,
    )) * proper_time_per_r
    h2[i] = ComplexF64(orbit_phase * distribution_q0(
        terms.mm0, terms.mm1, terms.mm2, F, Fp, Fpp,
    )) * proper_time_per_r
    return nothing
end

angular_operator_coeff(mode::AngularMode, s_weight, theta) =
    -mode.m / sin(theta) + mode.spheroid_c * sin(theta) +
    s_weight * cos(theta) / sin(theta)

angular_operator_coeff_derivative(mode::AngularMode, s_weight, theta) =
    mode.m * cos(theta) / sin(theta)^2 +
    mode.spheroid_c * cos(theta) -
    s_weight / sin(theta)^2

function ldag_on_S(mode::AngularMode, s_weight, theta)
    return angular_dS(mode, theta, 1) +
           angular_operator_coeff(mode, s_weight, theta) * angular_S(mode, theta)
end

function rho_theta_derivatives(kerr::KerrParams, r, theta)
    a = kerr.a
    rho = rho_np(kerr, r, theta)
    rhob = rhobar_np(kerr, r, theta)
    rp = 1im * a * sin(theta) * rho^2
    rbp = -1im * a * sin(theta) * rhob^2
    rpp = 1im * a * cos(theta) * rho^2 - 2 * a^2 * sin(theta)^2 * rho^3
    rbpp = -1im * a * cos(theta) * rhob^2 - 2 * a^2 * sin(theta)^2 * rhob^3
    return rho, rhob, rp, rbp, rpp, rbpp
end

function l1p_l2p_s_from_values(kerr::KerrParams, r, theta, mode::AngularMode,
                               S, Sp, Spp)
    rho = rho_np(kerr, r, theta)
    a2 = angular_operator_coeff(mode, 2, theta)
    a2p = angular_operator_coeff_derivative(mode, 2, theta)
    a1 = angular_operator_coeff(mode, 1, theta)

    L2S = Sp + a2 * S
    L1Sp = Spp + a1 * Sp
    L1L2S = L1Sp + a2p * S + a2 * Sp + a1 * a2 * S

    return rho^(-1) * L1L2S +
           3im * kerr.a * sin(theta) * a1 * S +
           3im * kerr.a * cos(theta) * S +
           2im * kerr.a * sin(theta) * Sp -
           1im * kerr.a * sin(theta) * a2 * S
end


function l1p_l2p_s(kerr::KerrParams, r, theta, mode::AngularMode)
    S, Sp, Spp = angular_values(mode, theta)
    return l1p_l2p_s_from_values(kerr, r, theta, mode, S, Sp, Spp)
end

L2S(mode::AngularMode, theta) = ldag_on_S(mode, 2, theta)

function weighted_source(proj::SourceProjections, Wnn, Wnmb, Wmbmb)
    return proj.N^2 * Wnn + proj.N * proj.Mbar * Wnmb + proj.Mbar^2 * Wmbmb
end

end
