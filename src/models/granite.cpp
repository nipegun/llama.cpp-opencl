#include "models.h"

// IBM Granite dense models (arch "granite": Granite 3.x, Granite 4.x dense, e.g. Granite 4.2 3B), ported from
// src/models/granite.cpp of upstream llama.cpp. Only the dense path of the architecture is included: upstream shares
// this graph with granitemoe (the MoE branch of build_layer_ffn) and with Granite Vision (the deepstack streams of the
// projector, injected before the decoder layers), and this fork has neither MoE nor multimodal support.

void llama_model_granite::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, false);

    // Granite uses rope_finetuned as a switch for rope, so default to true
    bool rope_finetuned = true;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned; // needed for round trip save
    std::fill(hparams.rope_pattern.begin(), hparams.rope_pattern.end(), rope_finetuned);

    switch (hparams.n_layer()) {
        case 32: type = LLM_TYPE_3B; break;
        case 40: {
            switch (hparams.n_embd) {
                case 2048: type = LLM_TYPE_2B; break;
                case 2560: type = LLM_TYPE_3B; break;
                case 4096: type = LLM_TYPE_8B; break;
                default: type = LLM_TYPE_UNKNOWN;
            }
            break;
        }
        case 64: type = LLM_TYPE_30B; break;
        // Add additional layer/vocab/etc checks here for other model sizes
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_granite::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // optional MLP bias
        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_granite::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_granite::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // the embedding scale of Granite (f_embedding_scale) is applied by build_inp_embd
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - built only if rope enabled
    ggml_tensor * inp_pos = nullptr;
    if (hparams.has_rope(0)) {
        inp_pos = build_inp_pos();
    }
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        cur = build_attention_layer(
            cur, inp_pos, inp_attn,
            n_embd_head, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // ffn
        cur = build_layer_ffn(cur, inpSA, il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    // For Granite architectures - scale logits
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_granite::graph::build_attention_layer(
          ggml_tensor             * cur,
          ggml_tensor             * inp_pos,
          llm_graph_input_attn_kv * inp_attn,
    const int64_t                 n_embd_head,
    const int                     il) {

    auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il);

    // normal RoPE (pairs of consecutive head values) with the parameters of upstream's llama_model::get_rope_params:
    // the layers without RoPE (rope_finetuned = false) keep Q and K as they are
    if (hparams.has_rope(il)) {
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                hparams.n_rot(il), rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                hparams.n_rot(il), rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_granite::graph::build_layer_ffn(
          ggml_tensor       * cur,
          ggml_tensor       * inpSA,
    const int                 il) {

    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    // feed-forward network: granite is dense (the MoE branch of upstream belongs to granitemoe, which is not included)
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_norm(ffn_inp,
            model.layers[il].ffn_norm, NULL,
            LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

    cur = build_ffn(cur,
            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
            model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
            model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "ffn_out", il);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
