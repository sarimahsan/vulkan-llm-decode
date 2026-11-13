#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>

// External CRT functions (from libs/msvcrt.lib)
void *malloc(size_t size);
void free(void *ptr);
void *memcpy(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
int printf(const char *fmt, ...);
void *fopen(const char *filename, const char *mode);
int fclose(void *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, void *stream);
int fseek(void *stream, long offset, int whence);
long ftell(void *stream);

#define GGUF_MAGIC 0x46554747u // "GGUF" little-endian
#define GGUF_DEFAULT_ALIGNMENT 32

typedef enum {
  GGUF_TYPE_UINT8   = 0,
  GGUF_TYPE_INT8    = 1,
  GGUF_TYPE_UINT16  = 2,
  GGUF_TYPE_INT16   = 3,
  GGUF_TYPE_UINT32  = 4,
  GGUF_TYPE_INT32   = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL    = 7,
  GGUF_TYPE_STRING  = 8,
  GGUF_TYPE_ARRAY   = 9,
  GGUF_TYPE_UINT64  = 10,
  GGUF_TYPE_INT64   = 11,
  GGUF_TYPE_FLOAT64 = 12
} GgufValueType;

typedef enum {
  GGML_TYPE_F32  = 0,
  GGML_TYPE_F16  = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14
} GgmlType;

typedef struct {
  char name[128];
  uint32_t n_dims;
  uint64_t dims[4];
  uint32_t type;
  uint64_t offset; // Offset relative to tensor_data_start
} GgufTensorInfo;

typedef struct {
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t block_count;
  uint32_t context_length;
  uint32_t vocab_size;
  uint32_t bos_token_id;
  uint32_t eos_token_id;
  float rms_norm_eps;
  float rope_theta;
  uint32_t alignment;
} GgufModelConfig;

typedef struct {
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
  GgufModelConfig config;
  GgufTensorInfo *tensors;
  char **vocab_tokens;
  uint32_t vocab_count;
  uint8_t *file_buf;
  size_t file_size;
  const uint8_t *tensor_data;
} GgufFile;

static inline int gguf_str_eq(const char *a, const char *b) {
  while (*a && *b) {
    if (*a != *b) return 0;
    a++; b++;
  }
  return (*a == *b);
}

static inline int gguf_str_ends_with(const char *s, const char *suffix) {
  size_t slen = 0, suflen = 0;
  while (s[slen]) slen++;
  while (suffix[suflen]) suflen++;
  if (suflen > slen) return 0;
  return gguf_str_eq(s + (slen - suflen), suffix);
}

static inline uint32_t gguf_read_u32(const uint8_t **p) {
  uint32_t v;
  memcpy(&v, *p, 4);
  *p += 4;
  return v;
}

static inline uint64_t gguf_read_u64(const uint8_t **p) {
  uint64_t v;
  memcpy(&v, *p, 8);
  *p += 8;
  return v;
}

static inline float gguf_read_f32(const uint8_t **p) {
  float v;
  memcpy(&v, *p, 4);
  *p += 4;
  return v;
}

static void gguf_skip_value(const uint8_t **p, uint32_t type) {
  switch (type) {
  case GGUF_TYPE_UINT8:
  case GGUF_TYPE_INT8:
  case GGUF_TYPE_BOOL:
    *p += 1;
    break;
  case GGUF_TYPE_UINT16:
  case GGUF_TYPE_INT16:
    *p += 2;
    break;
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
  case GGUF_TYPE_FLOAT32:
    *p += 4;
    break;
  case GGUF_TYPE_UINT64:
  case GGUF_TYPE_INT64:
  case GGUF_TYPE_FLOAT64:
    *p += 8;
    break;
  case GGUF_TYPE_STRING: {
    uint64_t len = gguf_read_u64(p);
    *p += len;
    break;
  }
  case GGUF_TYPE_ARRAY: {
    uint32_t elem_type = gguf_read_u32(p);
    uint64_t count = gguf_read_u64(p);
    for (uint64_t i = 0; i < count; ++i) {
      gguf_skip_value(p, elem_type);
    }
    break;
  }
  default:
    break;
  }
}

static int gguf_load_file(const char *path, GgufFile *out) {
  memset(out, 0, sizeof(GgufFile));
  out->config.alignment = GGUF_DEFAULT_ALIGNMENT;
  out->config.rms_norm_eps = 1e-6f;
  out->config.rope_theta = 10000.0f;

  void *fp = fopen(path, "rb");
  if (!fp) {
    printf("Error: Could not open GGUF file: %s\n", path);
    return 0;
  }
  fseek(fp, 0, 2); // SEEK_END
  long size = ftell(fp);
  fseek(fp, 0, 0); // SEEK_SET

  uint8_t *buf = (uint8_t *)malloc((size_t)size);
  if (!buf) {
    fclose(fp);
    return 0;
  }
  size_t read_bytes = fread(buf, 1, (size_t)size, fp);
  fclose(fp);

  out->file_buf = buf;
  out->file_size = read_bytes;

  const uint8_t *p = buf;
  uint32_t magic = gguf_read_u32(&p);
  if (magic != GGUF_MAGIC) {
    printf("Error: Invalid GGUF magic: 0x%08X\n", magic);
    return 0;
  }
  out->version = gguf_read_u32(&p);
  out->tensor_count = gguf_read_u64(&p);
  out->metadata_kv_count = gguf_read_u64(&p);

  // Parse metadata KV pairs
  for (uint64_t i = 0; i < out->metadata_kv_count; ++i) {
    uint64_t klen = gguf_read_u64(&p);
    char key[128];
    uint64_t copy_len = (klen < 127) ? klen : 127;
    memcpy(key, p, (size_t)copy_len);
    key[copy_len] = '\0';
    p += klen;

    uint32_t vtype = gguf_read_u32(&p);
    if (vtype == GGUF_TYPE_UINT32) {
      uint32_t val = gguf_read_u32(&p);
      if (gguf_str_eq(key, "general.alignment")) out->config.alignment = val;
      else if (gguf_str_ends_with(key, ".embedding_length")) out->config.hidden_size = val;
      else if (gguf_str_ends_with(key, ".feed_forward_length")) out->config.intermediate_size = val;
      else if (gguf_str_ends_with(key, ".attention.head_count")) out->config.num_heads = val;
      else if (gguf_str_ends_with(key, ".attention.head_count_kv")) out->config.num_kv_heads = val;
      else if (gguf_str_ends_with(key, ".block_count")) out->config.block_count = val;
      else if (gguf_str_ends_with(key, ".context_length")) out->config.context_length = val;
      else if (gguf_str_eq(key, "tokenizer.ggml.bos_token_id")) out->config.bos_token_id = val;
      else if (gguf_str_eq(key, "tokenizer.ggml.eos_token_id")) out->config.eos_token_id = val;
    } else if (vtype == GGUF_TYPE_FLOAT32) {
      float val = gguf_read_f32(&p);
      if (gguf_str_ends_with(key, ".attention.layer_norm_rms_epsilon")) out->config.rms_norm_eps = val;
      else if (gguf_str_ends_with(key, ".rope.freq_base")) out->config.rope_theta = val;
    } else if (vtype == GGUF_TYPE_ARRAY && gguf_str_eq(key, "tokenizer.ggml.tokens")) {
      uint32_t elem_type = gguf_read_u32(&p);
      uint64_t count = gguf_read_u64(&p);
      if (elem_type == GGUF_TYPE_STRING && count > 0) {
        out->vocab_count = (uint32_t)count;
        out->config.vocab_size = (uint32_t)count;
        out->vocab_tokens = (char **)malloc(sizeof(char *) * (size_t)count);
        for (uint64_t t = 0; t < count; ++t) {
          uint64_t slen = gguf_read_u64(&p);
          char *tok = (char *)malloc((size_t)slen + 1);
          if (slen > 0) memcpy(tok, p, (size_t)slen);
          tok[slen] = '\0';
          out->vocab_tokens[t] = tok;
          p += slen;
        }
      } else {
        for (uint64_t t = 0; t < count; ++t) gguf_skip_value(&p, elem_type);
      }
    } else {
      gguf_skip_value(&p, vtype);
    }
  }

  if (out->config.num_heads > 0 && out->config.hidden_size > 0) {
    out->config.head_dim = out->config.hidden_size / out->config.num_heads;
  }

  // Parse Tensor Info table
  out->tensors = (GgufTensorInfo *)malloc(sizeof(GgufTensorInfo) * (size_t)out->tensor_count);
  for (uint64_t i = 0; i < out->tensor_count; ++i) {
    GgufTensorInfo *t = &out->tensors[i];
    memset(t, 0, sizeof(GgufTensorInfo));

    uint64_t nlen = gguf_read_u64(&p);
    uint64_t copy_len = (nlen < 127) ? nlen : 127;
    memcpy(t->name, p, (size_t)copy_len);
    t->name[copy_len] = '\0';
    p += nlen;

    t->n_dims = gguf_read_u32(&p);
    for (uint32_t d = 0; d < t->n_dims && d < 4; ++d) {
      t->dims[d] = gguf_read_u64(&p);
    }
    t->type = gguf_read_u32(&p);
    t->offset = gguf_read_u64(&p);
  }

  // Align to general.alignment for start of tensor data blob
  size_t header_bytes = (size_t)(p - buf);
  size_t align = (size_t)out->config.alignment;
  size_t data_offset = (header_bytes + align - 1) & ~(align - 1);
  out->tensor_data = buf + data_offset;

  return 1;
}

static const GgufTensorInfo *gguf_find_tensor(const GgufFile *gf, const char *name) {
  for (uint64_t i = 0; i < gf->tensor_count; ++i) {
    if (gguf_str_eq(gf->tensors[i].name, name)) {
      return &gf->tensors[i];
    }
  }
  return NULL;
}

static void gguf_free(GgufFile *gf) {
  if (gf->vocab_tokens) {
    for (uint32_t i = 0; i < gf->vocab_count; ++i) {
      if (gf->vocab_tokens[i]) free(gf->vocab_tokens[i]);
    }
    free(gf->vocab_tokens);
  }
  if (gf->tensors) free(gf->tensors);
  if (gf->file_buf) free(gf->file_buf);
  memset(gf, 0, sizeof(GgufFile));
}

#endif // GGUF_H

