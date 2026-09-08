using PrecompileTools: @compile_workload

@compile_workload begin
    config = Internal.FR.FastReducedWaveformConfig(
        a=0.9,
        ell=2,
        m=2,
        omega=0.1,
        energy=1.2,
        lz=8.0,
        carter_q=4.0,
        theta_infinity=pi / 2,
        phi_infinity=0.0,
        theta_sign=1.0,
        orbit_kind="scattering",
        r_outer_min=1000.0,
        r_outer_floor=400.0,
        nsteps_per_branch=2000,
        asymptotic_tail_correction=true,
        asymptotic_match_phase=20.0,
        source_tail_order=3,
        green_tail_correction=true,
        scattering_green_tail_slow_order=3,
        scattering_green_tail_fast_order=2,
        source_grid="table",
        npoints=2001,
        homogeneous_rsin=-50.0,
        homogeneous_rsout_min=500.0,
        homogeneous_method="linear",
        homogeneous_tolerance=1.0e-12,
        integration_rule="trapezoid",
    )
    Internal.FR.compute_fast_reduced_waveform(config)
end
