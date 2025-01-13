#ifndef LLAMA_CPP_HIBIKI_LIBRARY_H
#define LLAMA_CPP_HIBIKI_LIBRARY_H

#include "llama.h"

// llama.cpp/common/build-info.cpp
int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "";
char const *LLAMA_COMPILER = "";
char const *LLAMA_BUILD_TARGET = "";

#ifdef __cplusplus
extern "C" {
#endif

struct HibikiCommonParamsSampling;

struct HibikiCommonSampler;

struct HibikiCommonParamsSampling * hibiki_common_params_sampling_init();

void hibiki_common_params_sampling_free(struct HibikiCommonParamsSampling * params);

void hibiki_common_params_sampling_set_frequency_penalty(struct HibikiCommonParamsSampling * params, float frequency_penalty);

void hibiki_common_params_sampling_set_presence_penalty(struct HibikiCommonParamsSampling * params, float presence_penalty);

void hibiki_common_params_sampling_set_seed(struct HibikiCommonParamsSampling * params, int32_t seed);

void hibiki_common_params_sampling_set_temperature(struct HibikiCommonParamsSampling * params, float temperature);

void hibiki_common_params_sampling_set_top_p(struct HibikiCommonParamsSampling * params, float top_p);

struct HibikiCommonSampler * hibiki_common_sampler_init(const struct llama_model * model, const struct HibikiCommonParamsSampling * common_params);

void hibiki_common_sampler_free(struct HibikiCommonSampler * common_sampler);

void hibiki_common_sampler_accept(struct HibikiCommonSampler * gsmpl, llama_token token, bool accept_grammar);

void hibiki_common_sampler_reset(struct HibikiCommonSampler * gsmpl);

struct HibikiCommonSampler * hibiki_common_sampler_clone(struct HibikiCommonSampler * gsmpl);

llama_token hibiki_common_sampler_sample(struct HibikiCommonSampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first);
#ifdef __cplusplus
}
#endif
#endif //LLAMA_CPP_HIBIKI_LIBRARY_H