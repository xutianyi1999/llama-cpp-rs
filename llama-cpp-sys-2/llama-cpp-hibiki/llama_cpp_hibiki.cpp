#include "llama_cpp_hibiki.h"
#include "sampling.h"
#include "speculative.h"
#include "json.hpp"
#include "chat.h"
#include "ngram-cache.h"

using json = nlohmann::ordered_json;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            return default_value;
        }
    } else {
        return default_value;
    }
}

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

llama_token_data_array * hibiki_common_sampler_get_candidates(struct HibikiCommonSampler *gsmpl) {
    common_sampler * common_sampler = reinterpret_cast<struct common_sampler*>(gsmpl);
    return common_sampler_get_candidates(common_sampler);
}

bool hibiki_common_speculative_are_compatible(const struct llama_context *ctx_tgt,
    const struct llama_context *ctx_dft) {
    return common_speculative_are_compatible(ctx_tgt, ctx_dft);
}

void hibiki_common_ngram_cache_update(struct HibikiCommonNgramCache *ngram_cache, int ngram_min, int ngram_max,
    const llama_token *inp_data, int inp_data_len, int nnew, bool print_progress) {
    common_ngram_cache * cache = reinterpret_cast<common_ngram_cache *>(ngram_cache);
    std::vector<llama_token> vec_data(inp_data, inp_data + inp_data_len);

    common_ngram_cache_update(
        *cache,
        ngram_min,
        ngram_max,
        vec_data,
        nnew,
        print_progress
    );
}

void hibiki_common_ngram_cache_draft(const llama_token *inp_data, int inp_data_len, llama_token *out,
    int * draft_len, int n_draft, int ngram_min, int ngram_max, struct HibikiCommonNgramCache *nc_context,
    struct HibikiCommonNgramCache *nc_dynamic, struct HibikiCommonNgramCache *nc_static) {
    std::vector<llama_token> inp(inp_data, inp_data + inp_data_len);
    std::vector<llama_token> draft = { out[0] };

    common_ngram_cache * context = reinterpret_cast<common_ngram_cache *>(nc_context);
    common_ngram_cache * dynamic = reinterpret_cast<common_ngram_cache *>(nc_dynamic);
    common_ngram_cache * static_ = reinterpret_cast<common_ngram_cache *>(nc_static);

    common_ngram_cache_draft(
        inp,
        draft,
        n_draft,
        ngram_min,
        ngram_max,
        *context,
        *dynamic,
        *static_
    );

    std::memcpy(out, draft.data(), draft.size() * sizeof(llama_token));
    *draft_len = draft.size();
}

void hibiki_common_ngram_cache_save(struct HibikiCommonNgramCache *ngram_cache, const char *c_filename) {
    common_ngram_cache * cache = reinterpret_cast<common_ngram_cache *>(ngram_cache);
    std::string filename = std::string(c_filename);
    common_ngram_cache_save(*cache, filename);
}

struct HibikiCommonNgramCache * hibiki_common_ngram_cache_load(const char *c_filename) {
    std::string filename = std::string(c_filename);
    common_ngram_cache cache = common_ngram_cache_load(filename);
    common_ngram_cache *p = new common_ngram_cache(cache);

    struct HibikiCommonNgramCache * hibiki_cache = reinterpret_cast<struct HibikiCommonNgramCache *>(p);
    return hibiki_cache;
}

struct HibikiCommonNgramCache * hibiki_common_ngram_cache_new() {
    common_ngram_cache *p = new common_ngram_cache;
    return reinterpret_cast<struct HibikiCommonNgramCache *>(p);
}

void hibiki_common_ngram_cache_free(struct HibikiCommonNgramCache *nc) {
    common_ngram_cache * cache = reinterpret_cast<common_ngram_cache *>(nc);
    delete cache;
}

struct HibikiCommonChatTemplates * hibiki_common_chat_templates_from_model(const struct llama_model *model, const char *template_name) {
    std::string chat_template_name;

    if (template_name != nullptr) {
        chat_template_name = std::string(template_name);
    }
    common_chat_templates_ptr t = common_chat_templates_init(model, chat_template_name);
    common_chat_templates_ptr *p = new common_chat_templates_ptr(std::move(t));
    return reinterpret_cast<struct HibikiCommonChatTemplates *>(p);
}

void hibiki_common_chat_templates_free(struct HibikiCommonChatTemplates *p) {
    common_chat_templates_ptr *t = reinterpret_cast<common_chat_templates_ptr*>(p);
    delete t;
}

struct HibikiCommonChatParams * hibiki_body_to_chat_params(const struct HibikiCommonChatTemplates *hibiki_tmpls, const char *json_str) {
    const common_chat_templates_ptr *chat_templates_ptr = reinterpret_cast<const common_chat_templates_ptr*>(hibiki_tmpls);
    common_chat_templates * chat_templates = chat_templates_ptr->get();

    json body = json::parse(json_str);

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(body.at("messages"));

    auto tools = json_value(body, "tools", json());
    inputs.tools = common_chat_tools_parse_oaicompat(tools);

    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);

    auto json_schema = json_value(body, "json_schema", json());
    inputs.json_schema = json_schema.is_null() ? "" : json_schema.dump();

    auto grammar = json_value(body, "grammar", std::string());

    if (!json_schema.is_null() && !grammar.empty()) {
        printf("Cannot use both json_schema and grammar\n");
        grammar.clear();
    }
    inputs.grammar = grammar;

    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.use_jinja = true;

    inputs.parallel_tool_calls = json_value(body, "parallel_tool_calls", false);

    common_chat_params chat_params = common_chat_templates_apply(chat_templates, inputs);
    common_chat_params *p = new common_chat_params(chat_params);
    return reinterpret_cast<struct HibikiCommonChatParams *>(p);
}

void hibiki_common_chat_params_free(struct HibikiCommonChatParams *p) {
    struct common_chat_params *params = reinterpret_cast<struct common_chat_params*>(p);
    delete params;
}

size_t hibiki_get_common_chat_params_prompt_length(struct HibikiCommonChatParams *params) {
    const struct common_chat_params *chat_params = reinterpret_cast<const struct common_chat_params*>(params);
    auto prompt = chat_params->prompt;
    return prompt.size();
}

void hibiki_get_common_chat_params_prompt(const struct HibikiCommonChatParams *params, char *out) {
    const struct common_chat_params *chat_params = reinterpret_cast<const struct common_chat_params*>(params);
    auto prompt = chat_params->prompt;
    strcpy(out, prompt.c_str());
}

enum HibikiCommonChatFormat hibiki_get_common_chat_params_format(const struct HibikiCommonChatParams *params) {
    const struct common_chat_params *chat_params = reinterpret_cast<const struct common_chat_params*>(params);
    common_chat_format out = chat_params->format;
    return (enum HibikiCommonChatFormat)out;
}

nlohmann::ordered_json to_json(const common_chat_msg& msg) {
    std::vector<nlohmann::ordered_json> tool_calls_json;
    for (const auto& tool_call : msg.tool_calls) {
        tool_calls_json.push_back({
            {"name", tool_call.name},
            {"arguments", tool_call.arguments},
            {"id", tool_call.id}
        });
    }

    std::vector<nlohmann::ordered_json> content_parts_json;
    for (const auto& part : msg.content_parts) {
        content_parts_json.push_back({
            {"type", part.type},
            {"text", part.text}
        });
    }

    return nlohmann::ordered_json{
            {"role", msg.role},
            {"content", msg.content},
            {"content_parts", content_parts_json},
            {"tool_calls", tool_calls_json},
            {"reasoning_content", msg.reasoning_content},
            {"tool_name", msg.tool_name},
            {"tool_call_id", msg.tool_call_id}
    };
}

void hibiki_common_chat_parse(const char * input, enum HibikiCommonChatFormat format_intput, char * out_json) {
    enum common_chat_format format = (enum common_chat_format)format_intput;
    common_chat_msg msg = common_chat_parse(std::string(input), format);
    auto msg_json = to_json(msg);
    auto msg_str = to_string(msg_json);
    strcpy(out_json, msg_str.c_str());
}
