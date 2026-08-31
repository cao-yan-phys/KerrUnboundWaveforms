module SpheroidalSphericalMixing

using LinearAlgebra
using SpinWeightedSpheroidalHarmonics

export spheroidal_expansion_coefficients,
       spheroidal_spherical_mixing_matrix,
       spheroidal_to_spherical_modes,
       write_mixing_coefficients_csv,
       write_spherical_modes_csv

csv_value(x::AbstractFloat) = isnan(x) ? "NaN" : repr(Float64(x))
csv_value(x::Integer) = string(x)
function csv_value(x::Complex)
    error("write real and imaginary parts explicitly, not a raw Complex value")
end
function csv_value(x::AbstractString)
    escaped = replace(x, "\"" => "\"\"")
    if occursin(",", escaped) || occursin("\"", escaped) ||
       occursin("\n", escaped) || occursin("\r", escaped)
        return "\"$escaped\""
    end
    return escaped
end

function validate_mode(spin::Int, ell::Int, m::Int)
    ell >= max(abs(spin), abs(m)) ||
        error("invalid mode: ell=$ell must be >= max(|spin|, |m|)")
end

function spheroidal_expansion_coefficients(spin::Int,
                                           spheroidal_l::Int,
                                           m::Int,
                                           spheroid_c;
                                           N::Int=-1,
                                           method="auto",
                                           coefficient_cutoff::Float64=0.0)
    validate_mode(spin, spheroidal_l, m)
    harmonic = spin_weighted_spheroidal_harmonic(
        spin,
        spheroidal_l,
        m,
        spheroid_c;
        N=N,
        method=method,
    )
    rows = NamedTuple[]
    for (coeff, spherical_harmonic) in zip(harmonic.coeffs,
                                           harmonic.spherical_harmonics_l)
        spherical_harmonic === nothing && continue
        abs(coeff) < coefficient_cutoff && continue
        push!(rows, (
            spherical_l=spherical_harmonic.l,
            spheroidal_l=spheroidal_l,
            m=m,
            spin=spin,
            spheroid_c=spheroid_c,
            coefficient=ComplexF64(coeff),
        ))
    end
    sort!(rows; by=row -> row.spherical_l)
    return rows
end

function inferred_spherical_ls(spin::Int,
                               spheroidal_ls,
                               m::Int,
                               spheroid_c;
                               N::Int=-1,
                               method="auto",
                               coefficient_cutoff::Float64=0.0)
    values = Set{Int}()
    for ell in spheroidal_ls
        for row in spheroidal_expansion_coefficients(
            spin,
            ell,
            m,
            spheroid_c;
            N=N,
            method=method,
            coefficient_cutoff=coefficient_cutoff,
        )
            push!(values, row.spherical_l)
        end
    end
    return sort!(collect(values))
end

function spheroidal_spherical_mixing_matrix(spin::Int,
                                            spherical_ls,
                                            spheroidal_ls,
                                            m::Int,
                                            spheroid_c;
                                            N::Int=-1,
                                            method="auto",
                                            coefficient_cutoff::Float64=0.0)
    spherical = Int.(collect(spherical_ls))
    spheroidal = Int.(collect(spheroidal_ls))
    row_index = Dict(ell => i for (i, ell) in enumerate(spherical))
    matrix = zeros(ComplexF64, length(spherical), length(spheroidal))
    for (j, ell_sph) in enumerate(spheroidal)
        rows = spheroidal_expansion_coefficients(
            spin,
            ell_sph,
            m,
            spheroid_c;
            N=N,
            method=method,
            coefficient_cutoff=coefficient_cutoff,
        )
        for row in rows
            i = get(row_index, row.spherical_l, nothing)
            i === nothing && continue
            matrix[i, j] += row.coefficient
        end
    end
    return matrix
end

function spheroidal_to_spherical_modes(spheroidal_amplitudes::AbstractDict;
                                       spin::Int=-2,
                                       m::Int,
                                       spheroid_c,
                                       spherical_ls=nothing,
                                       N::Int=-1,
                                       method="auto",
                                       coefficient_cutoff::Float64=0.0)
    spheroidal_ls = sort!(Int.(collect(keys(spheroidal_amplitudes))))
    if spherical_ls === nothing
        spherical_ls = inferred_spherical_ls(
            spin,
            spheroidal_ls,
            m,
            spheroid_c;
            N=N,
            method=method,
            coefficient_cutoff=coefficient_cutoff,
        )
    else
        spherical_ls = Int.(collect(spherical_ls))
    end
    matrix = spheroidal_spherical_mixing_matrix(
        spin,
        spherical_ls,
        spheroidal_ls,
        m,
        spheroid_c;
        N=N,
        method=method,
        coefficient_cutoff=coefficient_cutoff,
    )
    spheroidal_vector = ComplexF64[
        ComplexF64(spheroidal_amplitudes[ell])
        for ell in spheroidal_ls
    ]
    spherical_vector = matrix * spheroidal_vector
    return Dict(
        spherical_ls[i] => spherical_vector[i]
        for i in eachindex(spherical_ls)
        if abs(spherical_vector[i]) >= coefficient_cutoff
    )
end

function write_mixing_coefficients_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    columns = [
        :spin,
        :m,
        :spheroid_c,
        :spheroidal_l,
        :spherical_l,
        :re_coefficient,
        :im_coefficient,
        :abs_coefficient,
    ]
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            output = (
                spin=row.spin,
                m=row.m,
                spheroid_c=Float64(real(row.spheroid_c)),
                spheroidal_l=row.spheroidal_l,
                spherical_l=row.spherical_l,
                re_coefficient=real(row.coefficient),
                im_coefficient=imag(row.coefficient),
                abs_coefficient=abs(row.coefficient),
            )
            println(io, join((csv_value(getproperty(output, col)) for col in columns), ","))
        end
    end
    return path
end

function write_spherical_modes_csv(path::AbstractString;
                                   spin::Int,
                                   m::Int,
                                   spheroid_c,
                                   spherical_modes::AbstractDict)
    mkpath(dirname(path))
    columns = [
        :spin,
        :m,
        :spheroid_c,
        :spherical_l,
        :re_h_spherical,
        :im_h_spherical,
        :abs_h_spherical,
    ]
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for ell in sort!(collect(keys(spherical_modes)))
            value = spherical_modes[ell]
            row = (
                spin=spin,
                m=m,
                spheroid_c=Float64(real(spheroid_c)),
                spherical_l=ell,
                re_h_spherical=real(value),
                im_h_spherical=imag(value),
                abs_h_spherical=abs(value),
            )
            println(io, join((csv_value(getproperty(row, col)) for col in columns), ","))
        end
    end
    return path
end

end
