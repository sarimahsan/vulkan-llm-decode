#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

void exit(int code);
int _fltused = 0x9875;

static const char *ggml_type_name(uint32_t t) {
  switch (t) {
  case GGML_TYPE_F32:  return "F32";
  case GGML_TYPE_F16:  return "F16";
  case GGML_TYPE_Q4_0: return "Q4_0";
  case GGML_TYPE_Q4_1: return "Q4_1";
  case GGML_TYPE_Q5_0: return "Q5_0";
  case GGML_TYPE_Q5_1: return "Q5_1";
  case GGML_TYPE_Q8_0: return "Q8_0";
  case GGML_TYPE_Q4_K: return "Q4_K";
  case GGML_TYPE_Q6_K: return "Q6_K";
  default: return "OTHER";
  }
}

int mainCRTStartup() {
  const char *path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
  GgufFile gf;
  if (!gguf_load_file(path, &gf)) exit(1);

  printf("GGUF v%u | tensors=%llu | kv=%llu\n", gf.version,
         (unsigned long long)gf.tensor_count,
         (unsigned long long)gf.metadata_kv_count);
  printf("hidden=%u inter=%u heads=%u kv_heads=%u head_dim=%u blocks=%u ctx=%u\n",
         gf.config.hidden_size, gf.config.intermediate_size,
         gf.config.num_heads, gf.config.num_kv_heads, gf.config.head_dim,
         gf.config.block_count, gf.config.context_length);
  printf("vocab=%u bos=%u eos=%u eps=%g rope_theta=%g\n",
         gf.config.vocab_size, gf.config.bos_token_id, gf.config.eos_token_id,
         (double)gf.config.rms_norm_eps, (double)gf.config.rope_theta);

  printf("\nKey tensors:\n");
  for (uint64_t i = 0; i < gf.tensor_count; ++i) {
    const GgufTensorInfo *t = &gf.tensors[i];
    // Print non-blk tensors and blk.0.* tensors
    if (!(t->name[0] == 'b' && t->name[1] == 'l' && t->name[2] == 'k' && t->name[3] == '.') ||
        (t->name[4] == '0' && t->name[5] == '.')) {
      printf("  %-32s type=%-5s (%u) dims=[%llu, %llu, %llu, %llu]\n",
             t->name, ggml_type_name(t->type), t->type,
             (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
             (unsigned long long)t->dims[2], (unsigned long long)t->dims[3]);
    }
  }

  gguf_free(&gf);
  exit(0);
  return 0;
}

