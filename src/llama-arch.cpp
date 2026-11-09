#include "llama-arch.h"

#include "llama-ext.h"
#include "llama-impl.h"

#include <map>
#include <vector>

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_QWEN35,           "qwen35"           },
    { LLM_ARCH_GRANITE,          "granite"          },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                           "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,                   "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,           "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,                      "general.alignment"                     },
    { LLM_KV_GENERAL_FILE_TYPE,                      "general.file_type"                     },
    { LLM_KV_GENERAL_SAMPLING_SEQUENCE,              "general.sampling.sequence"             },
    { LLM_KV_GENERAL_SAMPLING_TOP_K,                 "general.sampling.top_k"                },
    { LLM_KV_GENERAL_SAMPLING_TOP_P,                 "general.sampling.top_p"                },
    { LLM_KV_GENERAL_SAMPLING_MIN_P,                 "general.sampling.min_p"                },
    { LLM_KV_GENERAL_SAMPLING_XTC_PROBABILITY,       "general.sampling.xtc_probability"      },
    { LLM_KV_GENERAL_SAMPLING_XTC_THRESHOLD,         "general.sampling.xtc_threshold"        },
    { LLM_KV_GENERAL_SAMPLING_TEMP,                  "general.sampling.temp"                 },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_LAST_N,        "general.sampling.penalty_last_n"       },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_REPEAT,        "general.sampling.penalty_repeat"       },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT,              "general.sampling.mirostat"             },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_TAU,          "general.sampling.mirostat_tau"         },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_ETA,          "general.sampling.mirostat_eta"         },
    { LLM_KV_GENERAL_NAME,                           "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                         "general.author"                        },
    { LLM_KV_GENERAL_VERSION,                        "general.version"                       },
    { LLM_KV_GENERAL_URL,                            "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,                    "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,                        "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,                     "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,                 "general.source.huggingface.repository" },
    { LLM_KV_GENERAL_TENSOR_EXTRA_NAME,              "general.tensor_extra.name"             },
    { LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4,           "general.tensor_extra.prec_a4"          },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_EMBEDDING_LENGTH_OUT,              "%s.embedding_length_out"              },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_NEXTN_PREDICT_LAYERS,              "%s.nextn_predict_layers"              },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "%s.logit_scale"                       },
    { LLM_KV_RESIDUAL_SCALE,                    "%s.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "%s.embedding_scale"                   },
    { LLM_KV_FULL_ATTENTION_INTERVAL,           "%s.full_attention_interval"           },

    { LLM_KV_ATTENTION_HEAD_COUNT,                   "%s.attention.head_count"                   },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,                "%s.attention.head_count_kv"                },
    { LLM_KV_ATTENTION_KEY_LENGTH,                   "%s.attention.key_length"                   },
    { LLM_KV_ATTENTION_VALUE_LENGTH,                 "%s.attention.value_length"                 },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,            "%s.attention.layer_norm_rms_epsilon"       },
    { LLM_KV_ATTENTION_CAUSAL,                       "%s.attention.causal"                       },
    { LLM_KV_ATTENTION_SCALE,                        "%s.attention.scale"                        },

    { LLM_KV_ATTENTION_KEY_LENGTH_SWA,               "%s.attention.key_length_swa"               },
    { LLM_KV_ATTENTION_VALUE_LENGTH_SWA,             "%s.attention.value_length_swa"             },
    { LLM_KV_ATTENTION_RECURRENT_LAYERS,             "%s.attention.recurrent_layers"             },

    { LLM_KV_ROPE_DIMENSION_COUNT,           "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_COUNT_SWA,       "%s.rope.dimension_count_swa"             },
    { LLM_KV_ROPE_DIMENSION_SECTIONS,        "%s.rope.dimension_sections"              },
    { LLM_KV_ROPE_SECTION_ORDER,             "%s.rope.section_order"                   },
    { LLM_KV_ROPE_FREQ_BASE,                 "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_SCALE_LINEAR,              "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,              "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,            "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ALPHA,             "%s.rope.scaling.alpha"                   },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,       "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,      "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,         "%s.rope.scaling.finetuned"               },

    { LLM_KV_SPLIT_NO,            "split.no"            },
    { LLM_KV_SPLIT_COUNT,         "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,    "%s.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,     "%s.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,     "%s.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK, "%s.ssm.time_step_rank" },
    { LLM_KV_SSM_GROUP_COUNT,    "%s.ssm.group_count"    },

    { LLM_KV_CLASSIFIER_OUTPUT_LABELS, "%s.classifier.output_labels" },
    { LLM_KV_CLASSIFIER_POOLING_TYPE,  "%s.classifier.pooling_type"  },
    { LLM_KV_CLASSIFIER_ACTIVATION,    "%s.classifier.activation"    },


    // sentence-transformers dense modules feature dims

    { LLM_KV_TOKENIZER_MODEL,                    "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                      "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                     "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,               "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,         "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,                   "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,                   "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,                   "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,                   "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_EOT_ID,                   "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,                   "tokenizer.ggml.eom_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,                   "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,                   "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,                   "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,                   "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,                  "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,                  "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,                  "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_SEP,                  "tokenizer.ggml.add_sep_token"            },
    { LLM_KV_TOKENIZER_DNA_AUTO_TAGS,            "tokenizer.ggml.dna_auto_tags"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,               "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,          "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_CLEAN_SPACES,             "tokenizer.ggml.clean_spaces"             },
    { LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,     "tokenizer.ggml.precompiled_charsmap"     },
    { LLM_KV_TOKENIZER_HF_JSON,                  "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                     "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE,            "tokenizer.chat_template"                 },
    { LLM_KV_TOKENIZER_NORMALIZER_NFC,           "tokenizer.ggml.normalizer.nfc"           },
    { LLM_KV_TOKENIZER_NORMALIZER_LOWERCASE,     "tokenizer.ggml.normalizer.lowercase"     },
    { LLM_KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS, "tokenizer.ggml.normalizer.strip_accents" },
    { LLM_KV_TOKENIZER_METASPACE_PREPEND_SCHEME, "tokenizer.ggml.metaspace.prepend_scheme" },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,               "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,               "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,               "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,               "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,               "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,               "tokenizer.ggml.fim_sep_token_id"         },
    { LLM_KV_TOKENIZER_SUPPRESS_TOKENS,          "tokenizer.ggml.suppress_tokens"          },

    { LLM_KV_ADAPTER_TYPE,                    "adapter.type"               },
    { LLM_KV_ADAPTER_LORA_ALPHA,              "adapter.lora.alpha"         },
    { LLM_KV_ADAPTER_LORA_TASK_NAME,          "adapter.lora.task_name"     },
    { LLM_KV_ADAPTER_LORA_PROMPT_PREFIX,      "adapter.lora.prompt_prefix" },
    { LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS, "adapter.alora.invocation_tokens" },

    { LLM_KV_PRISM_HADAMARD_VERSION,              "prism.hadamard.version"              },
    { LLM_KV_PRISM_HADAMARD_TIED_OUTPUT,          "prism.hadamard.tied_output"          },
    { LLM_KV_PRISM_HADAMARD_BLOCK_SIZE,           "prism.hadamard.block_size"           },
    { LLM_KV_PRISM_HADAMARD_TRANSFORM,            "prism.hadamard.transform"            },
    { LLM_KV_PRISM_HADAMARD_AXIS,                 "prism.hadamard.axis"                 },
    { LLM_KV_PRISM_HADAMARD_SIGN_MODE,            "prism.hadamard.sign_mode"            },
    { LLM_KV_PRISM_HADAMARD_SIGN_WIDTHS,          "prism.hadamard.sign_widths"          },
    { LLM_KV_PRISM_HADAMARD_SIGN_VALUES,          "prism.hadamard.sign_values"          },
    { LLM_KV_PRISM_HADAMARD_WEIGHT_NAMES,         "prism.hadamard.weight_names"         },
    { LLM_KV_PRISM_HADAMARD_INVERSE_WEIGHT_NAMES, "prism.hadamard.inverse_weight_names" },
    { LLM_KV_PRISM_HADAMARD_GDN_V_GROUPED,        "prism.hadamard.gdn_v_grouped"        },

    // deprecated
    { LLM_KV_TOKENIZER_PREFIX_ID, "tokenizer.ggml.prefix_token_id" },
    { LLM_KV_TOKENIZER_SUFFIX_ID, "tokenizer.ggml.suffix_token_id" },
    { LLM_KV_TOKENIZER_MIDDLE_ID, "tokenizer.ggml.middle_token_id" },
};

static const std::map<llm_tensor, const char *> LLM_TENSOR_NAMES = {
    { LLM_TENSOR_TOKEN_EMBD,                             "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,                            "output_norm" },
    { LLM_TENSOR_OUTPUT,                                 "output" },
    { LLM_TENSOR_ROPE_FREQS,                             "rope_freqs" },
    { LLM_TENSOR_ROPE_FACTORS_LONG,                      "rope_factors_long" },
    { LLM_TENSOR_ROPE_FACTORS_SHORT,                     "rope_factors_short" },
    { LLM_TENSOR_ATTN_NORM,                              "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,                                 "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,                                 "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,                                 "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,                               "blk.%d.attn_output" },
    { LLM_TENSOR_FFN_NORM,                               "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,                               "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,                               "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,                                 "blk.%d.ffn_up" },
    { LLM_TENSOR_ATTN_POST_NORM,                         "blk.%d.post_attention_norm" },
    { LLM_TENSOR_ATTN_Q_NORM,                            "blk.%d.attn_q_norm" },
    { LLM_TENSOR_ATTN_K_NORM,                            "blk.%d.attn_k_norm" },
    { LLM_TENSOR_ATTN_GATE,                              "blk.%d.attn_gate" },
    { LLM_TENSOR_ATTN_QKV,                               "blk.%d.attn_qkv" },
    { LLM_TENSOR_CLS_OUT,                                "cls.output" },
    { LLM_TENSOR_SSM_A_NOSCAN,                           "blk.%d.ssm_a" },
    { LLM_TENSOR_SSM_CONV1D,                             "blk.%d.ssm_conv1d" },
    { LLM_TENSOR_SSM_DT,                                 "blk.%d.ssm_dt" },
    { LLM_TENSOR_SSM_ALPHA,                              "blk.%d.ssm_alpha" },
    { LLM_TENSOR_SSM_NORM,                               "blk.%d.ssm_norm" },
    { LLM_TENSOR_SSM_OUT,                                "blk.%d.ssm_out" },
    { LLM_TENSOR_SSM_BETA,                               "blk.%d.ssm_beta" },
    { LLM_TENSOR_NEXTN_EH_PROJ,                          "blk.%d.nextn.eh_proj" },
    { LLM_TENSOR_NEXTN_EMBED_TOKENS,                     "blk.%d.nextn.embed_tokens" },
    { LLM_TENSOR_NEXTN_ENORM,                            "blk.%d.nextn.enorm" },
    { LLM_TENSOR_NEXTN_HNORM,                            "blk.%d.nextn.hnorm" },
    { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,                 "blk.%d.nextn.shared_head_head" },
    { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,                 "blk.%d.nextn.shared_head_norm" },
};

// declare information about the model weight tensors:
// - the layer in which the tensor is going to be used. this is needed in order to assign the correct buffer type for the weight
// - the operator which is going to use the weight. this is needed to determine if the respective backend supports the operator
//
// for example, input layers are usually assigned to CPU/host buffer types
//
// a mismatch between the declared information and the actual layer/op in which the tensor is used can lead to sub-optimal
//   assignment of the buffer types and extra overhead during computation
// example: https://github.com/ggml-org/llama.cpp/pull/17548
//
static const std::map<llm_tensor, llm_tensor_info> LLM_TENSOR_INFOS = {
    {LLM_TENSOR_TOKEN_EMBD,                 {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_OUTPUT,                     {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CLS_OUT,                    {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_OUTPUT_NORM,                {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},
    {LLM_TENSOR_ROPE_FREQS,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ROPE_FACTORS_LONG,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ROPE_FACTORS_SHORT,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ATTN_Q,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_QKV,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_OUT,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_GATE,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_DT,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_OUT,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_ALPHA,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_CONV1D,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_SSM_A_NOSCAN,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}}, // a version of SSM_A used for MUL instead of SSM_SCAN
    {LLM_TENSOR_SSM_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_BETA,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_NORM,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_POST_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_Q_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    // NextN/MTP tensors are stored per-block (blk.%d.nextn.*) even though only the
    // last nextn_predict_layers blocks carry them. Classify as LAYER_REPEATING so
    // the model loader doesn't fault on the block index.
    {LLM_TENSOR_NEXTN_EH_PROJ,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_NEXTN_EMBED_TOKENS,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_NEXTN_ENORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_NEXTN_HNORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
};

LLM_KV::LLM_KV(llm_arch arch, const char * suffix) : arch(arch), suffix(suffix) {}

std::string LLM_KV::operator()(llm_kv kv) const {
    std::string name = ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));

    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

LLM_TN_IMPL::LLM_TN_IMPL(llm_arch arch, llm_tensor tensor, const char * suffix, int bid, int xid)
    : arch(arch), tensor(tensor), suffix(suffix), bid(bid), xid(xid) {}

std::string LLM_TN_IMPL::str() const {
    if (LLM_TENSOR_NAMES.find(tensor) == LLM_TENSOR_NAMES.end()) {
        GGML_ABORT("unknown tensor name for tensor id %d", static_cast<int>(tensor));
    }

    std::string name = ::format(LLM_TENSOR_NAMES.at(tensor), bid, xid);
    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

std::vector<llm_arch> llm_arch_all() {
    std::vector<llm_arch> ret;
    ret.reserve(LLM_ARCH_NAMES.size());
    for (const auto & [arch, _] : LLM_ARCH_NAMES) {
        ret.push_back(arch);
    }
    return ret;
}

const char * llm_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

const llm_tensor_info & llm_tensor_info_for(llm_tensor tensor) {
    return LLM_TENSOR_INFOS.at(tensor);
}

// qwen35 is hybrid and supports recurrent state rollback, granite is a plain transformer (KV cache only). Both support
// mixed batches and saving. Only qwen35 supports the tensor split mode: llama_meta_device_get_split_state only knows
// the layout of its tensors (e.g. the attention gate inside its Q tensors)
bool llama_arch_supports_save(llm_arch arch) {
    return arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_GRANITE;
}

std::vector<llm_arch> llama_arch_all() {
    return llm_arch_all();
}

const char * llama_arch_name(llm_arch arch) {
    return llm_arch_name(arch);
}

llm_arch llama_arch_from_string(const std::string & name) {
    return llm_arch_from_string(name);
}

std::string llama_arch_kv_name(llm_arch arch, llm_kv key) {
    return LLM_KV(arch)(key);
}

bool llama_arch_supports_sm_tensor(llm_arch arch) {
    return arch == LLM_ARCH_QWEN35;
}

bool llama_arch_supports_mixed_batch(llm_arch arch) {
    return arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_GRANITE;
}
