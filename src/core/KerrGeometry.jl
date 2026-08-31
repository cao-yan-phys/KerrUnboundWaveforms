module KerrGeometry

export rplus, rminus, delta, rstar_from_r, r_from_rstar

function rplus(a::Real)
    a = Float64(a)
    abs(a) < 1 || throw(DomainError(a, "require |a| < 1"))
    return 1 + sqrt(1 - a^2)
end

rminus(a::Real) = 1 - sqrt(1 - Float64(a)^2)

delta(a::Real, r) = r^2 - 2r + Float64(a)^2

function rstar_from_r(a::Real, r::Real)
    rp = rplus(a)
    rm = rminus(a)
    r > rp || throw(DomainError(r, "require r > r_+"))
    return r + 2rp / (rp - rm) * log((r - rp) / 2) -
           2rm / (rp - rm) * log((r - rm) / 2)
end

function r_from_rstar(a::Real, target::Real)
    a = Float64(a)
    target = Float64(target)
    rp = rplus(a)
    rm = rminus(a)
    inner_log_coefficient = 2rp / (rp - rm)
    outer_log_coefficient = 2rm / (rp - rm)
    horizon_constant = rp - outer_log_coefficient * log((rp - rm) / 2)
    r = if target < 5
        rp + 2 * exp((target - horizon_constant) / inner_log_coefficient)
    else
        max(target - 2 * log(target), rp + 0.1)
    end
    for _ in 1:7
        correction = (rstar_from_r(a, r) - target) * delta(a, r) / (r^2 + a^2)
        next_r = r - correction
        r = next_r > rp ? next_r : (r + rp) / 2
    end
    return r
end

end


