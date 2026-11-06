// libmtmd stub
//
// llamacppOpenCL11 only supports the qwen35 text model, so the multimodal library (vision, audio input and
// audio generation) is not included. This stub keeps the public API of mtmd.h and mtmd-helper.h, so that the
// server, the CLI and the tests build unchanged:
// - the functions that create a multimodal context or media data fail cleanly and log why
// - the functions that only return defaults (parameters, markers, text-only capabilities) behave as upstream
// - every other function is a safe no-op that returns an empty, false or null value

#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdint>
#include <cstdio>

#define MTMD_STUB_MSG "multimodal support is not included in llamacppOpenCL11"

static ggml_log_callback g_log_callback  = nullptr;
static void *            g_log_user_data = nullptr;

static void mtmd_stub_log_error(const char * func) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: " MTMD_STUB_MSG "\n", func);
    if (g_log_callback) {
        g_log_callback(GGML_LOG_LEVEL_ERROR, buf, g_log_user_data);
    } else {
        fputs(buf, stderr);
        fflush(stderr);
    }
}

// a list of chunks that is always empty, so that callers can create and free it
struct mtmd_input_chunks {
};

//
// mtmd.h
//

const char * mtmd_default_marker() {
    return "<__media__>";
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* device            */ nullptr,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* batch_max_tokens  */ 1024,
        /* progress_callback */ nullptr,
        /* progress_callback_user_data */ nullptr,
    };
    return params;
}

mtmd_context * mtmd_init_from_file(const char *, const struct llama_model *, const struct mtmd_context_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_free(mtmd_context *) {
}

bool mtmd_decode_use_non_causal(const mtmd_context *, const mtmd_input_chunk *) {
    return false;
}

bool mtmd_decode_use_mrope(const mtmd_context *) {
    return false;
}

bool mtmd_support_vision(const mtmd_context *) {
    return false;
}

bool mtmd_support_audio(const mtmd_context *) {
    return false;
}

int mtmd_get_audio_sample_rate(const mtmd_context *) {
    return -1; // audio is not supported
}

struct mtmd_video_config mtmd_get_video_config(const mtmd_context *, int32_t, int32_t, int32_t, float) {
    return {};
}

const char * mtmd_get_marker(const mtmd_context *) {
    return mtmd_default_marker();
}

mtmd_bitmap * mtmd_bitmap_init(uint32_t, uint32_t, const unsigned char *) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t, const float *) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap *) {
    return 0;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap *) {
    return 0;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap *) {
    return nullptr;
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap *) {
    return 0;
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap *) {
    return false;
}

bool mtmd_bitmap_is_voice(const mtmd_bitmap *) {
    return false;
}

void mtmd_bitmap_free(mtmd_bitmap *) {
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap *) {
    return "";
}

void mtmd_bitmap_set_id(mtmd_bitmap *, const char *) {
}

void mtmd_bitmap_set_mergeable(mtmd_bitmap *, bool) {
}

void mtmd_bitmap_set_is_video(mtmd_bitmap *, bool) {
}

mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context *, const char *, void *, mtmd_bitmap_lazy_callback) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks *) {
    return 0;
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks *, size_t) {
    return nullptr;
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    delete chunks;
}

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk *) {
    return MTMD_INPUT_CHUNK_TYPE_TEXT;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk *, size_t * n_tokens_output) {
    if (n_tokens_output) {
        *n_tokens_output = 0;
    }
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk *) {
    return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk *) {
    return 0;
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk *) {
    return "";
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk *) {
    return 0;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk *) {
    return nullptr;
}

void mtmd_input_chunk_free(mtmd_input_chunk *) {
}

mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk *) {
    return nullptr;
}

int32_t mtmd_input_chunk_save(const mtmd_input_chunk *, char *, size_t, size_t * expected_out_len) {
    if (expected_out_len) {
        *expected_out_len = 0;
    }
    return 1;
}

mtmd_input_chunk * mtmd_input_chunk_load(const char *, size_t) {
    return nullptr;
}

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens *) {
    return 0;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens *) {
    return "";
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens *) {
    return 0;
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens *) {
    return 0;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens *) {
    return 0;
}

struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens *, llama_pos, size_t) {
    return {};
}

int32_t mtmd_tokenize(const mtmd_context *, mtmd_input_chunks *, const mtmd_input_text *, const mtmd_bitmap * const *, size_t) {
    mtmd_stub_log_error(__func__);
    return 2; // media preprocessing error
}

int32_t mtmd_tokenize_from_parts(const mtmd_context *, mtmd_input_chunks *, const mtmd_input_part * const *, size_t, bool) {
    mtmd_stub_log_error(__func__);
    return 2; // media preprocessing error
}

int32_t mtmd_encode(mtmd_context *, const mtmd_image_tokens *) {
    return 1;
}

int32_t mtmd_encode_chunk(mtmd_context *, const mtmd_input_chunk *) {
    return 1;
}

float * mtmd_get_output_embd(mtmd_context *) {
    return nullptr;
}

mtmd_batch * mtmd_batch_init(mtmd_context *) {
    return nullptr;
}

void mtmd_batch_free(mtmd_batch *) {
}

int32_t mtmd_batch_add_chunk(mtmd_batch *, const mtmd_input_chunk *) {
    return 1;
}

int32_t mtmd_batch_encode(mtmd_batch *) {
    return 1;
}

float * mtmd_batch_get_output_embd(mtmd_batch *, const mtmd_input_chunk *) {
    return nullptr;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_log_callback  = log_callback;
    g_log_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char *) {
    mtmd_stub_log_error(__func__);
    return {};
}

bool mtmd_support_audio_stream(const mtmd_context *) {
    return false;
}

struct mtmd_audio_stream_params mtmd_audio_stream_params_default() {
    mtmd_audio_stream_params params;
    params.step_ms  = 320;
    params.prompt   = nullptr;
    params.n_prompt = 0;
    return params;
}

mtmd_audio_stream * mtmd_audio_stream_init(mtmd_context *, struct mtmd_audio_stream_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_audio_stream_free(mtmd_audio_stream *) {
}

int32_t mtmd_audio_stream_push(mtmd_audio_stream *, const float *, size_t) {
    return 1; // the stream has ended
}

int32_t mtmd_audio_stream_flush(mtmd_audio_stream *, bool) {
    return 1; // the stream has ended
}

int32_t mtmd_audio_stream_step(mtmd_audio_stream *, llama_token) {
    return -2; // error
}

const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream *) {
    return nullptr;
}

bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream *) {
    return false;
}

int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream *) {
    return 0;
}

int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream *) {
    return 0;
}

struct mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context *) {
    mtmd_gen_audio_info info {};
    info.type = MTMD_GEN_AUDIO_TYPE_NONE;
    return info;
}

size_t mtmd_gen_audio_read_tensor(mtmd_context *, const char *, float *, size_t) {
    return 0;
}

const char * mtmd_gen_audio_voice_name(const mtmd_context *, int32_t) {
    return nullptr;
}

mtmd_bitmap * mtmd_gen_audio_get_voice(const mtmd_context *, const char *) {
    return nullptr;
}

struct mtmd_gen_state_params mtmd_gen_state_params_default() {
    mtmd_gen_state_params params {};
    params.seed = UINT32_MAX;
    return params;
}

mtmd_gen_state * mtmd_gen_state_init(mtmd_context *, struct mtmd_gen_state_params) {
    return nullptr;
}

void mtmd_gen_state_free(mtmd_gen_state *) {
}

void mtmd_gen_state_reset(mtmd_gen_state *) {
}

int32_t mtmd_gen_state_set_ref(mtmd_context *, mtmd_gen_state *) {
    return 1;
}

struct mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context *) {
    mtmd_gen_inp inp {};
    inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    return inp;
}

int32_t mtmd_gen_audio_process(mtmd_context *, mtmd_gen_state *, const struct mtmd_gen_inp *, struct mtmd_gen_out *) {
    return 1;
}

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    return mtmd_input_chunks_init();
}

struct mtmd_memory_usage mtmd_get_memory_usage(const char *, struct mtmd_context_params) {
    mtmd_stub_log_error(__func__);
    return {};
}

//
// mtmd-helper.h
//

struct mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
    return {
        /* fps_target             */ 4.0f,
        /* ffmpeg_bin_dir         */ nullptr,
        /* timestamp_interval_ms  */ 5000,
        /* no_soundtrack          */ false,
    };
}

struct mtmd_helper_init_opt mtmd_helper_init_opt_default() {
    return {
        /* video_params */ mtmd_helper_video_init_params_default(),
    };
}

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    mtmd_log_set(log_callback, user_data);
}

struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(const mtmd_context *, const char *, bool, struct mtmd_helper_init_opt) {
    mtmd_stub_log_error(__func__);
    return { nullptr, nullptr };
}

struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(const mtmd_context *, const unsigned char *, size_t, bool, struct mtmd_helper_init_opt) {
    mtmd_stub_log_error(__func__);
    return { nullptr, nullptr };
}

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks *) {
    return 0;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks *) {
    return 0;
}

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens *, llama_pos, struct mtmd_decoder_pos *) {
}

int32_t mtmd_helper_eval_chunks(mtmd_context *, struct llama_context *, const mtmd_input_chunks *, llama_pos,
        llama_seq_id, int32_t, bool, llama_pos *) {
    mtmd_stub_log_error(__func__);
    return 1;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context *, struct llama_context *, const mtmd_input_chunk *, llama_pos,
        llama_seq_id, int32_t, bool, llama_pos *) {
    mtmd_stub_log_error(__func__);
    return 1;
}

int32_t mtmd_helper_decode_image_chunk(mtmd_context *, struct llama_context *, const mtmd_input_chunk *, float *,
        llama_pos, llama_seq_id, int32_t, llama_pos *, mtmd_helper_post_decode_callback, void *) {
    return -1; // not a valid image chunk
}

mtmd_helper_video * mtmd_helper_video_init(const struct mtmd_context *, const char *, struct mtmd_helper_video_init_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(const struct mtmd_context *, const unsigned char *, size_t,
        struct mtmd_helper_video_init_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_helper_video_free(mtmd_helper_video *) {
}

struct mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video *) {
    mtmd_helper_video_info info {};
    info.n_frames = -1;
    return info;
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video *, mtmd_bitmap ** out_bitmap, char ** out_text) {
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (out_text) {
        *out_text = nullptr;
    }
    return -2; // error
}

// without a multimodal context, the model is a text model that can chat
struct mtmd_helper_cap mtmd_helper_get_capabilities(const struct llama_model *, const struct mtmd_context *) {
    mtmd_helper_cap cap = {};
    cap.chat     = true;
    cap.text_gen = true;
    return cap;
}

bool mtmd_helper_model_can_chat(const struct llama_context *, const struct mtmd_context *) {
    return true;
}

bool mtmd_helper_support_video(const struct mtmd_context *) {
    return false;
}

struct mtmd_helper_gen_audio_inp mtmd_helper_gen_audio_inp_default() {
    mtmd_helper_gen_audio_inp inp {};
    inp.top_k    = 50;
    inp.top_p    = 1.0f;
    inp.seed     = UINT32_MAX; // random
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    return inp;
}

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context *, struct mtmd_context *) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio *) {
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio *) {
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio *, const struct mtmd_helper_gen_audio_inp *) {
    return 1;
}

int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio *, int32_t) {
    return -1;
}

int32_t mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio *, llama_token, const float *, const float ** h_state_out, bool * out_stop) {
    if (h_state_out) {
        *h_state_out = nullptr;
    }
    if (out_stop) {
        *out_stop = true;
    }
    return -1;
}

bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio *) {
    return false;
}

const llama_logit_bias * mtmd_helper_gen_audio_get_logit_bias(mtmd_helper_gen_audio *, size_t * n_bias) {
    if (n_bias) {
        *n_bias = 0;
    }
    return nullptr;
}

int32_t mtmd_helper_audio_stream_get_logit_bias(const mtmd_context *, const struct llama_model *, llama_logit_bias *, int32_t) {
    return 0;
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio *, int32_t * out_sample_rate, const char ** out_data,
        size_t * out_data_len, int64_t * out_n_samples) {
    if (out_sample_rate) {
        *out_sample_rate = 0;
    }
    if (out_data) {
        *out_data = nullptr;
    }
    if (out_data_len) {
        *out_data_len = 0;
    }
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    return 1;
}

int32_t mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio *) {
    return 1;
}

//
// test API of mtmd-helper.h (internals of mtmd_helper_video)
//

mtmd_helper_video_seq::~mtmd_helper_video_seq() {
}

void mtmd_helper_video_seq::configure(const mtmd_video_config &, const mtmd_helper_video_src &, float) {
}

int32_t mtmd_helper_video_seq::ts_per_n_frames() const {
    return 0;
}

mtmd_bitmap * mtmd_helper_video_seq::soundtrack_slice(int32_t) const {
    return nullptr;
}

mtmd_bitmap * mtmd_helper_video_seq::read_next_frame() {
    return nullptr;
}

int32_t mtmd_helper_video_seq::read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (out_text) {
        *out_text = nullptr;
    }
    return -2; // error
}
