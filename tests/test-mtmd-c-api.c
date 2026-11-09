#undef NDEBUG

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mtmd.h"
#include "mtmd-helper.h"

// llamacppOpenCL11 builds libmtmd as a stub without multimodal support:
// the C API must still compile and link, initialization must fail cleanly and the rest must be safe no-ops
int main(void) {
    printf("\n\nTesting the libmtmd stub C API...\n");
    printf("--------\n\n");

    struct mtmd_context_params params = mtmd_context_params_default();
    printf("Default media marker: %s\n", params.media_marker);
    assert(params.media_marker != NULL);
    assert(strcmp(params.media_marker, mtmd_default_marker()) == 0);

    // there is no multimodal support, so a context can never be created
    mtmd_context * ctx = mtmd_init_from_file("mmproj.gguf", NULL, params);
    assert(ctx == NULL);

    struct mtmd_caps caps = mtmd_get_cap_from_file("mmproj.gguf");
    assert(!caps.inp_vision && !caps.inp_audio && !caps.gen_audio);

    assert(!mtmd_support_vision(NULL));
    assert(!mtmd_support_audio(NULL));
    assert(mtmd_get_audio_sample_rate(NULL) == -1);

    // a text model without a multimodal context can chat
    struct mtmd_helper_cap cap = mtmd_helper_get_capabilities(NULL, NULL);
    assert(cap.chat && cap.text_gen);
    assert(!cap.vision && !cap.audio && !cap.video && !cap.audio_gen);

    // media cannot be created
    mtmd_bitmap * bmp = mtmd_bitmap_init(4, 4, NULL);
    assert(bmp == NULL);
    mtmd_bitmap_free(bmp);

    // the chunk containers can be created and freed, but they stay empty
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    assert(chunks != NULL);
    assert(mtmd_input_chunks_size(chunks) == 0);
    assert(mtmd_input_chunks_get(chunks, 0) == NULL);
    assert(mtmd_helper_get_n_tokens(chunks) == 0);

    struct mtmd_input_text txt = { "hello", 5, false, false };
    const int32_t rc = mtmd_tokenize(NULL, chunks, &txt, NULL, 0);
    printf("tokenize rc = %d (expect non-zero)\n", rc);
    assert(rc != 0);
    assert(mtmd_input_chunks_size(chunks) == 0);

    mtmd_input_chunks_free(chunks);

    printf("\n\nDONE: test the libmtmd stub C API...\n");

    return 0;
}
