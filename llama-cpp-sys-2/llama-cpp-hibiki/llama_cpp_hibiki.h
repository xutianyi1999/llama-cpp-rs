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

llama_token_data_array * hibiki_common_sampler_get_candidates(HibikiCommonSampler *gsmpl);

struct HibikiCommonSampler * hibiki_common_sampler_init(const struct llama_model * model, const struct HibikiCommonParamsSampling * common_params);

void hibiki_common_sampler_free(struct HibikiCommonSampler * common_sampler);

void hibiki_common_sampler_accept(struct HibikiCommonSampler * gsmpl, llama_token token, bool accept_grammar);

void hibiki_common_sampler_reset(struct HibikiCommonSampler * gsmpl);

struct HibikiCommonSampler * hibiki_common_sampler_clone(struct HibikiCommonSampler * gsmpl);

llama_token hibiki_common_sampler_sample(struct HibikiCommonSampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first);

bool hibiki_common_speculative_are_compatible( const struct llama_context * ctx_tgt, const struct llama_context * ctx_dft);

struct HibikiCommonNgramCache;

void hibiki_common_ngram_cache_update(
    struct HibikiCommonNgramCache* ngram_cache,
    int ngram_min,
    int ngram_max,
    const llama_token * inp_data,
    int inp_data_len,
    int nnew,
    bool print_progress
);

void hibiki_common_ngram_cache_draft(
    const llama_token * inp_data,
    int inp_data_len,
    const llama_token * draft,
    int draft_len,
    int n_draft,
    int ngram_min,
    int ngram_max,
    struct HibikiCommonNgramCache* nc_context,
    struct HibikiCommonNgramCache* nc_dynamic,
    struct HibikiCommonNgramCache* nc_static
);

void hibiki_common_ngram_cache_save(const struct HibikiCommonNgramCache* ngram_cache, const char *filename);

#ifdef __cplusplus
}
#endif
#endif //LLAMA_CPP_HIBIKI_LIBRARY_H