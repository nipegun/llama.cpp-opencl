#include "llama-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>

bool llama_hparams::is_swa_any() const {
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        if (is_swa_impl[il]) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer_all) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer_all) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer_all) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_rot(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_rot_swa : n_rot_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_inp() const {
    return n_embd;
}

uint32_t llama_hparams::n_embd_inp_enc() const {
    return n_embd_inp();
}

uint32_t llama_hparams::n_embd_out() const {
    return n_embd_out_impl > 0 ? n_embd_out_impl : n_embd;
}

uint32_t llama_hparams::n_embd_head_k(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_head_v(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa(il) ? n_embd_head_v_swa : n_embd_head_v_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k(il) * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_v(il) * n_head_kv;
}

bool llama_hparams::is_n_embd_v_gqa_variable() const {
    const uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        if (val != n_embd_v_gqa(il)) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_embd_v_gqa_max() const {
    uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        val = std::max(val, n_embd_v_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_r() const {
    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);
}

uint32_t llama_hparams::n_embd_s() const {
    return ssm_d_state * ssm_d_inner;
}

bool llama_hparams::is_recr(uint32_t il) const {
    if (il < n_layer_all) {
        return is_recr_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

uint32_t llama_hparams::n_pos_per_embd() const {
    return (rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE)
            ? GGML_MROPE_SECTIONS : 1;
}

bool llama_hparams::is_swa(uint32_t il) const {
    if (il < n_layer_all) {
        return is_swa_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

bool llama_hparams::has_kv(uint32_t il) const {
    return kv_source(il) == il;
}

uint32_t llama_hparams::kv_source(uint32_t il) const {
    if (il < n_layer_all) {
        return kv_source_impl[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

bool llama_hparams::has_rope(uint32_t il) const {
    if (il < n_layer_all) {
        return rope_pattern[il] != 0;
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, il, n_layer_all);
}

uint32_t llama_hparams::n_layer() const {
    return n_layer_all - n_layer_nextn;
}

