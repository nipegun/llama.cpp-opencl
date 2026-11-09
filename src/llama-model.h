#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;
struct llama_model;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_0_8B,
    LLM_TYPE_2B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_27B,
    LLM_TYPE_30B,
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order);

struct llama_layer_nextn {
    struct ggml_tensor * eh_proj               = nullptr;
    struct ggml_tensor * eh_proj_s             = nullptr;
    struct ggml_tensor * eh_proj_in_s          = nullptr;
    struct ggml_tensor * embed_tokens          = nullptr;
    struct ggml_tensor * enorm                 = nullptr;
    struct ggml_tensor * hnorm                 = nullptr;
    struct ggml_tensor * shared_head_head      = nullptr;
    struct ggml_tensor * shared_head_head_s    = nullptr;
    struct ggml_tensor * shared_head_head_in_s = nullptr;
    struct ggml_tensor * shared_head_norm      = nullptr;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_norm        = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    // attention bias
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wo_b      = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3

    // ff MoE (never loaded, checked by the qwen35 and granite graphs)
    struct ggml_tensor * ffn_gate_inp = nullptr;

    // linear attention (gated delta net)
    struct ggml_tensor * ssm_dt     = nullptr;
    struct ggml_tensor * ssm_out    = nullptr;
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_alpha  = nullptr;
    struct ggml_tensor * ssm_beta   = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // per-tensor weight scales (e.g. NVFP4 scale2)
    struct ggml_tensor * wq_s        = nullptr;
    struct ggml_tensor * wk_s        = nullptr;
    struct ggml_tensor * wv_s        = nullptr;
    struct ggml_tensor * wo_s        = nullptr;
    struct ggml_tensor * wqkv_s      = nullptr;
    struct ggml_tensor * wqkv_gate_s = nullptr;
    struct ggml_tensor * ffn_gate_s  = nullptr;
    struct ggml_tensor * ffn_up_s    = nullptr;
    struct ggml_tensor * ffn_down_s  = nullptr;
    struct ggml_tensor * ssm_out_s   = nullptr;
    struct ggml_tensor * ssm_alpha_s = nullptr;
    struct ggml_tensor * ssm_beta_s  = nullptr;

    // input scales
    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ssm_out_in_s       = nullptr;
    struct ggml_tensor * ssm_alpha_in_s     = nullptr;
    struct ggml_tensor * ssm_beta_in_s      = nullptr;

    struct llama_layer_nextn nextn;
};

struct llama_device {
    bool is_meta;

    ggml_backend_dev_t dev;
};

struct llama_meta_device_get_split_state_userdata {
    size_t                     n_devices;
    const struct llama_model * model;
};

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata);

struct llama_prec_policy {
    // the key is the weight tensor `res->src[0]`, stores the recommended accumulation type of the op (unused for now)
    // TODO: migrate ad-hoc ggml_prec_set_acc() calls to this container + update apply() to use it
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_acc;

    // the key is the weight tensor `res->src[0]`, stores the recommended activation precision type
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_src1;

    bool apply(ggml_tensor * res) const;

    void load(llama_model_loader & ml, const llama_model & model);
};

// transform of a folded weight, applied to the matmul input: optional sign flip, then the normalized block Hadamard rotation
struct llama_hadamard_transform {
    ggml_tensor * rot;
    ggml_tensor * signs; // nullptr for identity sign mode

    // if perm_rep > 1, permute the input from tiled head order [hd, nk, rep] to grouped order [hd, rep, nk] before signs and rotation
    int64_t perm_hd  = 0;
    int64_t perm_nk  = 0;
    int64_t perm_rep = 0;

    // ssm_out of a gated delta net gets its value heads in tiled order, but the fold used grouped order
    // record the head geometry for that permutation, return false if it does not match the input width
    bool set_gdn_v_perm(int64_t n_in, int64_t n_v, int64_t n_k) {
        if (n_k <= 0 || n_v <= 0 || n_v % n_k != 0 || n_in % n_v != 0) {
            return false;
        }
        perm_hd  = n_in / n_v;
        perm_nk  = n_k;
        perm_rep = n_v / n_k;
        return true;
    }
};
using llama_hadamard_rotations = std::unordered_map<const ggml_tensor *, llama_hadamard_transform>;

struct llama_hadamard {
    // names and sign data come from the GGUF metadata in load_hparams, the transforms are made in load_tensors
    std::unordered_map<std::string, uint32_t> weight_blocks;
    std::unordered_map<std::string, uint32_t> inverse_blocks;

    std::map<uint32_t, std::vector<int32_t>> sign_data;

    bool gdn_v_grouped = false;
    bool tied_output = false;

    llama_hadamard_rotations rot; // folded weight -> activation transform
    llama_hadamard_rotations inv; // latent lookup table -> inverse transform
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // per-tensor activation precision policy
    llama_prec_policy prec_policy;

    llama_hadamard hdmd;

    // see llama_model_set_attn_rot
    bool attn_rot = true;

    // for classifier models
    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd    = nullptr;
    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output      = nullptr;

    // NVFP4 per-tensor scale2, input_scale for LM head
    struct ggml_tensor * output_s    = nullptr;
    struct ggml_tensor * output_in_s = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;
    struct ggml_tensor * cls_norm  = nullptr;

    std::vector<llama_layer> layers;

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers   = nullptr;
    struct ggml_tensor * dense_2_out_layers_b = nullptr;
    struct ggml_tensor * dense_3_out_layers   = nullptr;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<llama_device> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // for keeping track of associated LoRA adapters
    std::unordered_set<llama_adapter_lora *> loras;

    // which tensors can be prefetched - driven by TENSOR_READ_LAZY
    std::unordered_set<const ggml_tensor *> can_prefetch;

    // statically allocated context for assigning
    struct llama_meta_device_get_split_state_userdata get_split_state_ud;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const llama_model_params & params);
    virtual ~llama_model();

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    llama_ftype ftype() const;

    size_t size() const; // file size
    size_t n_tensors() const;
    size_t n_devices() const;
    const float * tensor_split() const;

    uint32_t n_gpu_layers() const;
    llama_split_mode split_mode() const;
    bool no_alloc() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;

    ggml_cgraph * build_graph(const llm_graph_params & params) const;

    virtual void load_stats  (llama_model_loader & ml) = 0;
    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_vocab  (llama_model_loader & ml) = 0;
    virtual bool load_tensors(llama_model_loader & ml) = 0; // returns false if cancelled by progress_callback

    // model must define these
    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;

protected:
    llama_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params);
llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params);

// model must inherit from this
struct llama_model_base : public llama_model {
    friend struct llama_model;

    llama_model * model;
    llama_model_loader * ml = nullptr;
    const LLM_TN tn;

    // llama_model_loader is not yet defined at this point, so we will set it after construction
    const int TENSOR_DUPLICATED;
    const int TENSOR_NOT_REQUIRED;
    const int TENSOR_SKIP;
    const int TENSOR_SKIP_IF_VIRTUAL;
    const int TENSOR_ALLOW_RESHAPE;
    const int TENSOR_READ_LAZY;

    explicit llama_model_base(const llama_model_params & params);
    virtual ~llama_model_base() = default;

    ggml_tensor * create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // convenience overload of create_tensor that doesn't require llama_model_loader
    ggml_tensor * create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // helper: try to load merged qkv first, fall back to separate q, k, v
    void create_tensor_qkv(llama_layer & layer, int bid,
                int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
                int flags);

    // helper: tensor flags for a file that holds the trunk, the NextN layers, or both
    struct nextn_flags_t {
        int trunk; // TENSOR_NOT_REQUIRED when the file holds only the NextN layers
        int mtp;   // TENSOR_NOT_REQUIRED when the file holds only the trunk, TENSOR_SKIP when MTP is not loaded
    };
    nextn_flags_t nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe = LLM_TENSOR_ATTN_NORM) const;

    // helper: read the prism.hadamard metadata and record which weights are folded
    void load_hparams_hadamard(llama_model_loader & ml);

    // helper: make the prism.hadamard rotation and sign tensors for the folded weights
    void load_tensors_hadamard();

    void load_stats  (llama_model_loader & ml) override;
    void load_hparams(llama_model_loader & ml) override;
    void load_vocab  (llama_model_loader & ml) override;
    bool load_tensors(llama_model_loader & ml) override;

    // model must define these
    void load_arch_hparams(llama_model_loader & ml) override = 0;
    void load_arch_tensors(llama_model_loader & ml) override = 0;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override = 0;
};

const char * llm_type_name(llm_type type);

// convenience macro for loading local variables for load_tensors() in llama_model_base
// note: cast to int64_t since we will use these for the tensor dimensions
#define LLAMA_LOAD_LOCALS \
    const int     n_layer        = hparams.n_layer();        GGML_UNUSED(n_layer); \
    const int     n_layer_all    = hparams.n_layer_all;      GGML_UNUSED(n_layer_all); \
    const int     n_layer_nextn  = hparams.n_layer_nextn;    GGML_UNUSED(n_layer_nextn); \
    const int64_t n_head         = hparams.n_head();         GGML_UNUSED(n_head); \
    const int64_t n_head_kv      = hparams.n_head_kv();      GGML_UNUSED(n_head_kv); \
    const int64_t n_embd         = hparams.n_embd;           GGML_UNUSED(n_embd); \
    const int64_t n_embd_k_gqa   = hparams.n_embd_k_gqa();   GGML_UNUSED(n_embd_k_gqa); \
    const int64_t n_embd_v_gqa   = hparams.n_embd_v_gqa();   GGML_UNUSED(n_embd_v_gqa); \
    const int64_t n_embd_head_k  = hparams.n_embd_head_k();  GGML_UNUSED(n_embd_head_k); \
    const int64_t n_embd_head_v  = hparams.n_embd_head_v();  GGML_UNUSED(n_embd_head_v); \
    const int64_t n_ff           = hparams.n_ff();           GGML_UNUSED(n_ff); \
    const int64_t n_embd_gqa     = n_embd_v_gqa;             GGML_UNUSED(n_embd_gqa); \
    const int64_t n_vocab        = vocab.n_tokens();         GGML_UNUSED(n_vocab); \
    const int64_t n_token_types  = vocab.n_token_types();    GGML_UNUSED(n_token_types); \
    const int64_t n_rot          = hparams.n_rot();          GGML_UNUSED(n_rot); \
    const int64_t n_ctx_train    = hparams.n_ctx_train;      GGML_UNUSED(n_ctx_train);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
