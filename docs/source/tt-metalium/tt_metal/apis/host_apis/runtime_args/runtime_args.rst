Runtime Arguments
==================

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, std::shared_ptr<RuntimeArgs> runtime_args)

.. doxygenfunction:: SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::vector< CoreCoord > & core_spec, const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args)

.. doxygenfunction:: UpdateRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args)

.. doxygenfunction:: GetRuntimeArgs

.. doxygenfunction:: SetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::vector<uint32_t> &runtime_args)