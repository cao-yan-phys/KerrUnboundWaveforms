module NativeSNLinear

using OrdinaryDiffEqRosenbrock: Rosenbrock23
using OrdinaryDiffEqVerner: AutoVern9
using SciMLBase: ODEProblem, solve
using StaticArrays: SVector

using ..NativeSN

export InSolution, solve_in, match_infinity, evaluate_in

struct InSolution
    mode::SNMode
    rsin::Float64
    rsout::Float64
    rin::Float64
    rout::Float64
    numerical_solution
end

function rstar_rhs(u, mode::SNMode, rs)
    r = r_from_rstar(mode, rs)
    F, U = potentials(mode, r)
    return SVector(u[2], F * u[2] + U * u[1])
end

function solve_in(mode::SNMode;
                  rsin::Real=-50.0,
                  rsout::Real=1000.0,
                  tolerance::Real=1.0e-12,
                  horizon_order::Int=0,
                  maxiters::Integer=100_000)
    rsin < rsout || throw(ArgumentError("require rsin < rsout"))
    mode.omega != 0 || throw(DomainError(mode.omega, "static modes are not covered"))
    rin = r_from_rstar(mode, rsin)
    rout = r_from_rstar(mode, rsout)
    p = mode.omega - mode.m * mode.a / (2rplus(mode))
    horizon_value, horizon_derivative = horizon_factor(mode, rin; order=horizon_order)
    phase = exp(-im * p * rsin)
    initial_X = phase * horizon_value
    initial_Y = phase * (-im * p * horizon_value +
        delta(mode, rin) / (rin^2 + mode.a^2) * horizon_derivative)
    problem = ODEProblem(rstar_rhs, SVector(initial_X, initial_Y), (rsin, rsout), mode)
    numerical_solution = solve(
        problem,
        AutoVern9(Rosenbrock23(autodiff=false));
        reltol=tolerance,
        abstol=tolerance,
        maxiters=maxiters,
    )
    return InSolution(mode, Float64(rsin), Float64(rsout), rin, rout, numerical_solution)
end

function evaluate_in(solution::InSolution, rs::Real)
    solution.rsin <= rs <= solution.rsout ||
        throw(DomainError(rs, "r_* lies outside the numerical interval"))
    solution.numerical_solution(rs)
end

function match_infinity(solution::InSolution; order::Int=3)
    mode = solution.mode
    r = solution.rout
    rs = solution.rsout
    D = delta(mode, r) / (r^2 + mode.a^2)
    fin, dfin = infinity_factor(mode, r, :ingoing; order=order)
    fout, dfout = infinity_factor(mode, r, :outgoing; order=order)
    incoming = fin * exp(-im * mode.omega * rs)
    outgoing = fout * exp(im * mode.omega * rs)
    incoming_derivative = (D * dfin - im * mode.omega * fin) * exp(-im * mode.omega * rs)
    outgoing_derivative = (D * dfout + im * mode.omega * fout) * exp(im * mode.omega * rs)
    boundary = evaluate_in(solution, rs)
    amplitudes = [outgoing incoming; outgoing_derivative incoming_derivative] \
        [boundary[1], boundary[2]]
    return (bref=amplitudes[1], binc=amplitudes[2])
end

end


