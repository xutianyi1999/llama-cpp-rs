#include "llama_cpp_hibiki.h"
#include "sampling.h"
#include "speculative.h"

HibikiCommonParamsSampling * hibiki_common_params_sampling_init() {
    common_params_sampling * params_sampling = new common_params_sampling;
    return reinterpret_cast<HibikiCommonParamsSampling*>(params_sampling);
}

void hibiki_common_params_sampling_free(HibikiCommonParamsSampling *params) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    delete p;
}

HibikiCommonSampler * hibiki_common_sampler_init(const llama_model * model, const HibikiCommonParamsSampling *common_params) {
    const common_params_sampling & params = *reinterpret_cast<const common_params_sampling*>(common_params);
    common_sampler * sampler = common_sampler_init(model, params);
    return reinterpret_cast<HibikiCommonSampler *>(sampler);
}

void hibiki_common_sampler_free(HibikiCommonSampler *sampler) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(sampler);
    common_sampler_free(common_sampler);
}

void hibiki_common_sampler_accept(HibikiCommonSampler *gsmpl, llama_token token, bool accept_grammar) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(gsmpl);
    common_sampler_accept(common_sampler, token, accept_grammar);
}

void hibiki_common_sampler_reset(HibikiCommonSampler *gsmpl) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(gsmpl);
    common_sampler_reset(common_sampler);
}

HibikiCommonSampler *hibiki_common_sampler_clone(HibikiCommonSampler *gsmpl) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(gsmpl);
    struct common_sampler * clone_sampler = common_sampler_clone(common_sampler);
    return reinterpret_cast<HibikiCommonSampler *>(clone_sampler);
}

llama_token hibiki_common_sampler_sample(HibikiCommonSampler *gsmpl, llama_context *ctx, int idx, bool grammar_first) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(gsmpl);
    return common_sampler_sample(common_sampler, ctx, idx, grammar_first);
}

void hibiki_common_params_sampling_set_frequency_penalty(struct HibikiCommonParamsSampling *params,
                                                         float frequency_penalty) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    p->penalty_freq = frequency_penalty;
}

void hibiki_common_params_sampling_set_presence_penalty(struct HibikiCommonParamsSampling *params, float presence_penalty) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    p->penalty_present = presence_penalty;
}

void hibiki_common_params_sampling_set_seed(struct HibikiCommonParamsSampling *params, int32_t seed) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    p->seed = seed;
}

void hibiki_common_params_sampling_set_temperature(struct HibikiCommonParamsSampling *params, float temperature) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    p->temp = temperature;
}

void hibiki_common_params_sampling_set_top_p(struct HibikiCommonParamsSampling *params, float top_p) {
    common_params_sampling * p = reinterpret_cast<common_params_sampling*>(params);
    p->top_p = top_p;
}

bool hibiki_common_speculative_are_compatible(const struct llama_context *ctx_tgt,
    const struct llama_context *ctx_dft) {
    return common_speculative_are_compatible(ctx_tgt, ctx_dft);
}
