#include "llama_cpp_hibiki.h"
#include "sampling.h"
#include <iostream>

CommonParamsSampling * hibiki_common_params_sampling_init() {
    common_params_sampling * params_sampling = new common_params_sampling;
    return reinterpret_cast<CommonParamsSampling*>(params_sampling);
}

CommonSampler * hibiki_common_sampler_init(const llama_model * model, const CommonParamsSampling *common_params) {
    const common_params_sampling & params = *reinterpret_cast<const common_params_sampling*>(common_params);
    const  common_sampler_init(model, params);
}
