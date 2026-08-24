module DeepBoltzmannMachines
    using Base: front, tail
    using RestrictedBoltzmannMachines: AbstractLayer, inputs_h_from_v, inputs_v_from_h,
        interaction_energy, sample_from_inputs, batch_size
    import RestrictedBoltzmannMachines: RBM, energy
    include("dbm.jl")
end # module
