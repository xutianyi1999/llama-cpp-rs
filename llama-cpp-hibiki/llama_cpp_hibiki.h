#ifndef LLAMA_CPP_HIBIKI_LIBRARY_H
#define LLAMA_CPP_HIBIKI_LIBRARY_H

#include "llama.h"

struct CommonParamsSampling;

struct CommonSampler;

extern "C" {
    CommonParamsSampling * hibiki_common_params_sampling_init();

    CommonSampler * hibiki_common_sampler_init(const llama_model * model, const CommonParamsSampling * common_params);

    void hibiki_common_sampler_free(CommonSampler * common_sampler);

    void hibiki_common_sampler_accept(CommonSampler * gsmpl, llama_token token, bool accept_grammar);

    void hibiki_common_sampler_reset(CommonSampler * gsmpl);

    CommonSampler * hibiki_common_sampler_clone(CommonSampler * gsmpl);

    llama_token hibiki_common_sampler_sample(CommonSampler * gsmpl, llama_context * ctx, int idx, bool grammar_first = false);
}

#endif //LLAMA_CPP_HIBIKI_LIBRARY_H