module KerrUnboundWaveforms

include(joinpath(@__DIR__, "core", "KerrGeometry.jl"))
include(joinpath(@__DIR__, "core", "NativeSN.jl"))
include(joinpath(@__DIR__, "core", "NativeSNLinear.jl"))
include(joinpath(@__DIR__, "core", "UnboundOrbitSpectrum.jl"))

using .UnboundOrbitSpectrum

const SpectrumConfig = UnboundOrbitSpectrum.UnboundSpectrumConfig
const Internal = UnboundOrbitSpectrum

export SpectrumConfig,
       solve,
       save,
       automatic_frequency_window,
       logarithmic_frequency_grid,
       hybrid_frequency_grid,
       fft_aligned_frequency_grid

const automatic_frequency_window = UnboundOrbitSpectrum.automatic_frequency_window
const logarithmic_frequency_grid = UnboundOrbitSpectrum.logarithmic_frequency_grid
const hybrid_frequency_grid = UnboundOrbitSpectrum.hybrid_frequency_grid
const fft_aligned_frequency_grid = UnboundOrbitSpectrum.fft_aligned_frequency_grid

function solve(config::SpectrumConfig)
    return UnboundOrbitSpectrum.run_unbound_spectrum(config; write_output=false)
end

function save(result, output_directory::AbstractString)
    path = abspath(output_directory)
    UnboundOrbitSpectrum.write_spectrum_outputs(path, result)
    result.config.inverse_fft &&
        UnboundOrbitSpectrum.write_time_domain_outputs(path, result)
    return path
end

include("precompile.jl")

end
