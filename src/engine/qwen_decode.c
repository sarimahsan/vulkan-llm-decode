#include "gguf.h"
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

int puts(const char *str);
int putchar(int c);
void exit(int code);

int _fltused = 0x9875;

#define MAX_SEQ 512
#define MAX_LAYERS 64

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = (x);                                                        \
    if (err != VK_SUCCESS) {                                                   \
      printf("Vulkan error at line %d: code %d\n", __LINE__, (int)err);        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Kernel32 QPC for wall-clock tok/s timing
typedef union {
  int64_t QuadPart;
} LARGE_INTEGER_QPC;
int __stdcall QueryPerformanceCounter(LARGE_INTEGER_QPC *lpPerformanceCount);
int __stdcall QueryPerformanceFrequency(LARGE_INTEGER_QPC *lpFrequency);

static int str_has(const char *hay, const char *needle) {
  for (int i = 0; hay[i]; ++i) {
    int j = 0;
    while (needle[j] && hay[i + j] == needle[j])
      ++j;
    if (!needle[j])
      return 1;
  }
  return 0;
}

static uint32_t parse_flag_uint(const char *cmd, const char *flag,
                                uint32_t def) {
  for (int i = 0; cmd[i]; ++i) {
    int j = 0;
    while (flag[j] && cmd[i + j] == flag[j])
      ++j;
    if (!flag[j] && (cmd[i + j] == ' ' || cmd[i + j] == '=')) {
      const char *p = cmd + i + j + 1;
      while (*p == ' ')
        ++p;
      uint32_t v = 0;
      if (*p < '0' || *p > '9')
        return def;
      while (*p >= '0' && *p <= '9') {
        v = v * 10u + (uint32_t)(*p - '0');
        ++p;
      }
      return v;
    }
  }
  return def;
}

static double get_time_sec(void) {
  LARGE_INTEGER_QPC t, f;
  QueryPerformanceCounter(&t);
  QueryPerformanceFrequency(&f);
  return (double)t.QuadPart / (double)f.QuadPart;
}

// Software IEEE-754 FP32 -> FP16
static inline uint16_t float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x007FFFFF;
  if (exp <= 0) {
    if (exp < -10)
      return (uint16_t)sign;
    mant = (mant | 0x00800000) >> (1 - exp);
    return (uint16_t)(sign | ((mant + 0x00000FFF + ((mant >> 13) & 1)) >> 13));
  } else if (exp == 0xFF - 127 + 15) {
    if (mant == 0)
      return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | 0x7C00 | (mant >> 13));
  } else {
    if (exp > 30)
      return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) |
                      ((mant + 0x00000FFF + ((mant >> 13) & 1)) >> 13));
  }
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  int32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x03FF;
  uint32_t val;
  if (exp == 0) {
    if (mant == 0)
      val = sign;
    else {
      exp = 1;
      while ((mant & 0x0400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FF;
      val = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    val = sign | 0x7F800000 | (mant << 13);
  } else {
    val = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
  }
  float res;
  memcpy(&res, &val, 4);
  return res;
}

static inline float sqrt_f(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float y = x;
  for (int i = 0; i < 12; ++i)
    y = 0.5f * (y + x / y);
  return y;
}

static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

static size_t ggml_tensor_bytes(const GgufTensorInfo *t) {
  uint64_t n = 1;
  for (uint32_t d = 0; d < t->n_dims; ++d)
    n *= t->dims[d];
  if (t->type == GGML_TYPE_F32)
    return (size_t)n * 4;
  if (t->type == GGML_TYPE_F16)
    return (size_t)n * 2;
  if (t->type == GGML_TYPE_Q4_0)
    return (size_t)(n / 32) * 18;
  if (t->type == GGML_TYPE_Q8_0)
    return (size_t)(n / 32) * 34;
  return 0;
}

static void dequant_q4_0_row_to_fp16(const uint8_t *blocks, uint32_t K,
                                     uint16_t *out_fp16) {
  uint32_t nb = K / 32;
  for (uint32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = blocks + b * 18;
    uint16_t d_u16;
    memcpy(&d_u16, blk, 2);
    float d = half_to_float(d_u16);
    const uint8_t *qs = blk + 2;
    for (uint32_t i = 0; i < 16; ++i) {
      int q0 = (int)(qs[i] & 0x0F) - 8;
      int q1 = (int)(qs[i] >> 4) - 8;
      out_fp16[b * 32 + i] = float_to_half((float)q0 * d);
      out_fp16[b * 32 + i + 16] = float_to_half((float)q1 * d);
    }
  }
}

static void repack_q4_0_coalesced(const uint8_t *src, size_t n_blocks,
                                  uint32_t *dst_w, uint16_t *dst_s) {
  for (size_t b = 0; b < n_blocks; ++b) {
    const uint8_t *blk = src + b * 18;
    memcpy(&dst_s[b], blk, 2);
    const uint8_t *qs = blk + 2;
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    for (uint32_t i = 0; i < 8; ++i) {
      w0 |= (uint32_t)(qs[i] & 0x0F) << (i * 4);
      w1 |= (uint32_t)(qs[i + 8] & 0x0F) << (i * 4);
      w2 |= (uint32_t)(qs[i] >> 4) << (i * 4);
      w3 |= (uint32_t)(qs[i + 8] >> 4) << (i * 4);
    }
    dst_w[b * 4 + 0] = w0;
    dst_w[b * 4 + 1] = w1;
    dst_w[b * 4 + 2] = w2;
    dst_w[b * 4 + 3] = w3;
  }
}

static void repack_q4_0_to_g128_coalesced(const uint8_t *src, size_t n_blocks,
                                          uint32_t *dst_w, uint16_t *dst_s) {
  size_t n_groups = n_blocks / 4;
  for (size_t g = 0; g < n_groups; ++g) {
    float vals[128];
    float amax = 0.0f;
    for (size_t sub = 0; sub < 4; ++sub) {
      const uint8_t *blk = src + (g * 4 + sub) * 18;
      uint16_t d_u16;
      memcpy(&d_u16, blk, 2);
      float d = half_to_float(d_u16);
      const uint8_t *qs = blk + 2;
      for (uint32_t i = 0; i < 16; ++i) {
        int q0 = (int)(qs[i] & 0x0F) - 8;
        int q1 = (int)(qs[i] >> 4) - 8;
        float v0 = (float)q0 * d;
        float v1 = (float)q1 * d;
        vals[sub * 32 + i] = v0;
        vals[sub * 32 + i + 16] = v1;
        float av0 = (v0 < 0.0f) ? -v0 : v0;
        float av1 = (v1 < 0.0f) ? -v1 : v1;
        if (av0 > amax)
          amax = av0;
        if (av1 > amax)
          amax = av1;
      }
    }
    float d128 = amax / 7.0f;
    float id128 = (d128 != 0.0f) ? (1.0f / d128) : 0.0f;
    dst_s[g] = float_to_half(d128);
    for (size_t sub = 0; sub < 4; ++sub) {
      size_t b = g * 4 + sub;
      uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
      for (uint32_t i = 0; i < 8; ++i) {
        int q0 = (int)(vals[sub * 32 + i] * id128 + 8.5f);
        int q1 = (int)(vals[sub * 32 + i + 8] * id128 + 8.5f);
        int q2 = (int)(vals[sub * 32 + i + 16] * id128 + 8.5f);
        int q3 = (int)(vals[sub * 32 + i + 24] * id128 + 8.5f);
        if (q0 < 0)
          q0 = 0;
        if (q0 > 15)
          q0 = 15;
        if (q1 < 0)
          q1 = 0;
        if (q1 > 15)
          q1 = 15;
        if (q2 < 0)
          q2 = 0;
        if (q2 > 15)
          q2 = 15;
        if (q3 < 0)
          q3 = 0;
        if (q3 > 15)
          q3 = 15;
        w0 |= (uint32_t)q0 << (i * 4);
        w1 |= (uint32_t)q1 << (i * 4);
        w2 |= (uint32_t)q2 << (i * 4);
        w3 |= (uint32_t)q3 << (i * 4);
      }
      dst_w[b * 4 + 0] = w0;
      dst_w[b * 4 + 1] = w1;
      dst_w[b * 4 + 2] = w2;
      dst_w[b * 4 + 3] = w3;
    }
  }
}

static void requant_q8_0_to_q4_g128_coalesced(const uint8_t *src,
                                              size_t n_blocks, uint32_t *dst_w,
                                              uint16_t *dst_s) {
  size_t n_groups = n_blocks / 4;
  for (size_t g = 0; g < n_groups; ++g) {
    float vals[128];
    float amax = 0.0f;
    for (size_t sub = 0; sub < 4; ++sub) {
      const uint8_t *blk = src + (g * 4 + sub) * 34;
      uint16_t d8_u16;
      memcpy(&d8_u16, blk, 2);
      float d8 = half_to_float(d8_u16);
      const int8_t *qs = (const int8_t *)(blk + 2);
      for (int i = 0; i < 32; ++i) {
        float v = (float)qs[i] * d8;
        vals[sub * 32 + i] = v;
        float av = (v < 0.0f) ? -v : v;
        if (av > amax)
          amax = av;
      }
    }
    float d128 = amax / 7.0f;
    float id128 = (d128 != 0.0f) ? (1.0f / d128) : 0.0f;
    dst_s[g] = float_to_half(d128);
    for (size_t sub = 0; sub < 4; ++sub) {
      size_t b = g * 4 + sub;
      for (uint32_t c = 0; c < 4; ++c) {
        uint32_t packed = 0;
        for (uint32_t j = 0; j < 8; ++j) {
          int q = (int)(vals[sub * 32 + c * 8 + j] * id128 + 8.5f);
          if (q < 0)
            q = 0;
          if (q > 15)
            q = 15;
          packed |= ((uint32_t)q) << (j * 4);
        }
        dst_w[b * 4 + c] = packed;
      }
    }
  }
}

static void repack_q4_0_to_q3_g128(const uint8_t *src, size_t n_blocks,
                                   uint32_t *dst_w, uint16_t *dst_s) {
  size_t n_groups = n_blocks / 4;
  for (size_t g = 0; g < n_groups; ++g) {
    float vals[128];
    float amax = 0.0f;
    for (size_t sub = 0; sub < 4; ++sub) {
      const uint8_t *blk = src + (g * 4 + sub) * 18;
      uint16_t d_u16;
      memcpy(&d_u16, blk, 2);
      float d = half_to_float(d_u16);
      const uint8_t *qs = blk + 2;
      for (uint32_t i = 0; i < 16; ++i) {
        int q0 = (int)(qs[i] & 0x0F) - 8;
        int q1 = (int)(qs[i] >> 4) - 8;
        float v0 = (float)q0 * d;
        float v1 = (float)q1 * d;
        vals[sub * 32 + i] = v0;
        vals[sub * 32 + i + 16] = v1;
        float av0 = (v0 < 0.0f) ? -v0 : v0;
        float av1 = (v1 < 0.0f) ? -v1 : v1;
        if (av0 > amax)
          amax = av0;
        if (av1 > amax)
          amax = av1;
      }
    }
    float d3 = (amax * 0.72f) / 3.5f;
    float id3 = (d3 != 0.0f) ? (1.0f / d3) : 0.0f;
    dst_s[g] = float_to_half(d3);
    for (size_t sub = 0; sub < 4; ++sub) {
      size_t b = g * 4 + sub;
      uint32_t q[32];
      for (uint32_t i = 0; i < 32; ++i) {
        int qi = (int)(vals[sub * 32 + i] * id3 + 4.5f);
        if (qi < 0)
          qi = 0;
        if (qi > 7)
          qi = 7;
        q[i] = (uint32_t)qi;
      }
      uint32_t u0 = 0, u1 = 0, u2 = 0;
      for (uint32_t i = 0; i < 10; ++i) {
        u0 |= q[i] << (i * 3);
        u1 |= q[10 + i] << (i * 3);
        u2 |= q[20 + i] << (i * 3);
      }
      u0 |= (q[30] & 3u) << 30;
      u1 |= (q[31] & 3u) << 30;
      u2 |= ((q[30] >> 2) & 1u) << 30;
      u2 |= ((q[31] >> 2) & 1u) << 31;
      dst_w[b * 3 + 0] = u0;
      dst_w[b * 3 + 1] = u1;
      dst_w[b * 3 + 2] = u2;
    }
  }
}

static void requant_q8_0_to_q4_coalesced(const uint8_t *src, size_t n_blocks,
                                         uint32_t *dst_w, uint16_t *dst_s) {
  for (size_t b = 0; b < n_blocks; ++b) {
    const uint8_t *blk = src + b * 34;
    uint16_t d8_u16;
    memcpy(&d8_u16, blk, 2);
    float d8 = half_to_float(d8_u16);
    const int8_t *qs = (const int8_t *)(blk + 2);

    float amax = 0.0f;
    float max_v = 0.0f;
    for (int i = 0; i < 32; ++i) {
      float v = (float)qs[i] * d8;
      float av = (v < 0.0f) ? -v : v;
      if (av > amax) {
        amax = av;
        max_v = v;
      }
    }
    float d4 = max_v / -8.0f;
    float id4 = (d4 != 0.0f) ? (1.0f / d4) : 0.0f;
    dst_s[b] = float_to_half(d4);

    for (uint32_t c = 0; c < 4; ++c) {
      uint32_t packed = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        float v = (float)qs[c * 8 + j] * d8;
        int q = (int)(v * id4 + 8.5f);
        if (q < 0)
          q = 0;
        if (q > 15)
          q = 15;
        packed |= ((uint32_t)q) << (j * 4);
      }
      dst_w[b * 4 + c] = packed;
    }
  }
}

// String helper without libc
static size_t cstr_len(const char *s) {
  size_t n = 0;
  while (s[n])
    n++;
  return n;
}

static void fmt_layer_name(char *buf, uint32_t layer, const char *suffix) {
  // "blk." + digits + "." + suffix
  buf[0] = 'b';
  buf[1] = 'l';
  buf[2] = 'k';
  buf[3] = '.';
  size_t pos = 4;
  if (layer >= 10) {
    buf[pos++] = (char)('0' + (layer / 10));
    buf[pos++] = (char)('0' + (layer % 10));
  } else {
    buf[pos++] = (char)('0' + layer);
  }
  buf[pos++] = '.';
  size_t slen = cstr_len(suffix);
  memcpy(buf + pos, suffix, slen + 1);
}

// Tokenizer helpers
static int32_t vocab_find_exact(const GgufFile *gf, const char *str) {
  for (uint32_t i = 0; i < gf->vocab_count; ++i) {
    if (gguf_str_eq(gf->vocab_tokens[i], str))
      return (int32_t)i;
  }
  return -1;
}

// Convert ASCII text into GPT-2/Qwen BPE UTF-8 representation (' ' -> 0xC4
// 0xA0, '\n' -> 0xC4 0x8A)
static size_t text_to_bpe_utf8(const char *in, char *out, size_t max_out) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 3 < max_out; ++i) {
    unsigned char c = (unsigned char)in[i];
    if (c == ' ') {
      out[j++] = (char)0xC4;
      out[j++] = (char)0xA0;
    } else if (c == '\n') {
      out[j++] = (char)0xC4;
      out[j++] = (char)0x8A;
    } else {
      out[j++] = (char)c;
    }
  }
  out[j] = '\0';
  return j;
}

// Greedy longest-match tokenizer over GGUF vocab
static uint32_t encode_bpe_greedy(const GgufFile *gf, const char *ascii_text,
                                  uint32_t *out_ids, uint32_t max_ids) {
  char bpe_buf[2048];
  size_t len = text_to_bpe_utf8(ascii_text, bpe_buf, sizeof(bpe_buf));
  uint32_t count = 0;
  size_t pos = 0;
  while (pos < len && count < max_ids) {
    int32_t best_id = -1;
    size_t best_len = 0;
    for (uint32_t v = 0; v < gf->vocab_count; ++v) {
      const char *tok = gf->vocab_tokens[v];
      size_t tlen = cstr_len(tok);
      if (tlen == 0 || tlen <= best_len || pos + tlen > len)
        continue;
      int match = 1;
      for (size_t k = 0; k < tlen; ++k) {
        if (tok[k] != bpe_buf[pos + k]) {
          match = 0;
          break;
        }
      }
      if (match) {
        best_id = (int32_t)v;
        best_len = tlen;
      }
    }
    if (best_id >= 0) {
      out_ids[count++] = (uint32_t)best_id;
      pos += best_len;
    } else {
      // Fallback single byte
      pos++;
    }
  }
  return count;
}

static void print_decoded_token(const char *tok) {
  char buf[256];
  size_t k = 0;
  for (size_t i = 0; tok[i] && k + 2 < sizeof(buf);) {
    unsigned char c0 = (unsigned char)tok[i];
    unsigned char c1 = (unsigned char)tok[i + 1];
    if (c0 == 0xC4 && c1 == 0xA0) {
      buf[k++] = ' ';
      i += 2;
    } else if (c0 == 0xC4 && c1 == 0x8A) {
      buf[k++] = '\n';
      i += 2;
    } else {
      buf[k++] = (char)c0;
      i += 1;
    }
  }
  if (k > 0) {
    buf[k] = '\0';
    printf("%s", buf);
  }
}

// Prompt-lookup / N-gram self-drafter: searches history backwards for longest
// matching 3-gram, 2-gram, or 1-gram suffix and proposes up to max_draft
// continuation tokens.
static uint32_t draft_ngram(const uint32_t *hist, uint32_t hist_len,
                            uint32_t *out_draft, uint32_t max_draft,
                            uint32_t eos_id, uint32_t im_end_id) {
  for (uint32_t n = 3; n >= 2; --n) {
    if (hist_len <= n)
      continue;
    for (int32_t i = (int32_t)(hist_len - n - 1); i >= 0; --i) {
      int match = 1;
      for (uint32_t j = 0; j < n; ++j) {
        if (hist[(uint32_t)i + j] != hist[hist_len - n + j]) {
          match = 0;
          break;
        }
      }
      if (match) {
        uint32_t count = 0;
        for (uint32_t k = 0; k < max_draft && (uint32_t)i + n + k < hist_len;
             ++k) {
          uint32_t tok = hist[(uint32_t)i + n + k];
          if (tok == eos_id || tok == im_end_id)
            break;
          out_draft[count++] = tok;
        }
        if (count > 0)
          return count;
      }
    }
  }
  return 0;
}

// Vulkan helpers
static uint32_t match_ngram_buf(const uint32_t *buf, uint32_t buf_len,
                                const uint32_t *suffix, uint32_t n,
                                uint32_t *out_draft, uint32_t max_draft,
                                uint32_t eos_id, uint32_t im_end_id) {
  if (buf_len <= n)
    return 0;
  for (int32_t i = (int32_t)(buf_len - n - 1); i >= 0; --i) {
    int match = 1;
    for (uint32_t j = 0; j < n; ++j) {
      if (buf[(uint32_t)i + j] != suffix[j]) {
        match = 0;
        break;
      }
    }
    if (match) {
      uint32_t count = 0;
      for (uint32_t k = 0; k < max_draft && (uint32_t)i + n + k < buf_len;
           ++k) {
        uint32_t tok = buf[(uint32_t)i + n + k];
        if (tok == eos_id || tok == im_end_id)
          break;
        out_draft[count++] = tok;
      }
      if (count > 0)
        return count;
    }
  }
  return 0;
}

static uint32_t draft_ngram_smart(const uint32_t *hist, uint32_t hist_len,
                                  const uint32_t *corpus, uint32_t corpus_len,
                                  uint32_t *out_draft, uint32_t max_draft,
                                  uint32_t eos_id, uint32_t im_end_id) {
  for (uint32_t n = 4; n >= 2; --n) {
    if (hist_len > n) {
      uint32_t c = match_ngram_buf(hist, hist_len, hist + hist_len - n, n,
                                   out_draft, max_draft, eos_id, im_end_id);
      if (c > 0)
        return c;
    }
  }
  if (corpus_len > 0) {
    for (uint32_t n = 3; n >= 2; --n) {
      if (hist_len >= n) {
        uint32_t c = match_ngram_buf(corpus, corpus_len, hist + hist_len - n, n,
                                     out_draft, max_draft, eos_id, im_end_id);
        if (c > 0)
          return c;
      }
    }
  }
  if (hist_len > 1) {
    uint32_t c = match_ngram_buf(hist, hist_len, hist + hist_len - 1, 1,
                                 out_draft, max_draft, eos_id, im_end_id);
    if (c > 0)
      return c;
  }
  if (corpus_len > 1 && hist_len >= 1) {
    uint32_t c = match_ngram_buf(corpus, corpus_len, hist + hist_len - 1, 1,
                                 out_draft, max_draft, eos_id, im_end_id);
    if (c > 0)
      return c;
  }
  return 0;
}
static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeFilter,
                                 VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((typeFilter & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  return 0;
}

static void create_buffer(VkDevice device, VkPhysicalDevice phys,
                          VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags props, VkBuffer *buf,
                          VkDeviceMemory *mem) {
  VkBufferCreateInfo bInfo = {0};
  bInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bInfo.size = size;
  bInfo.usage = usage;
  bInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bInfo, NULL, buf));

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, *buf, &memReqs);

  VkMemoryAllocateInfo aInfo = {0};
  aInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  aInfo.allocationSize = memReqs.size;
  aInfo.memoryTypeIndex = find_memory_type(phys, memReqs.memoryTypeBits, props);
  VK_CHECK(vkAllocateMemory(device, &aInfo, NULL, mem));
  VK_CHECK(vkBindBufferMemory(device, *buf, *mem, 0));
}

static VkShaderModule load_shader(VkDevice device, const char *path) {
  void *fp = fopen(path, "rb");
  if (!fp) {
    printf("Failed to open shader: %s\n", path);
    exit(1);
  }
  fseek(fp, 0, 2);
  long size = ftell(fp);
  fseek(fp, 0, 0);
  uint8_t *code = (uint8_t *)malloc((size_t)size);
  fread(code, 1, (size_t)size, fp);
  fclose(fp);

  VkShaderModuleCreateInfo info = {0};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = (size_t)size;
  info.pCode = (const uint32_t *)code;
  VkShaderModule mod;
  VK_CHECK(vkCreateShaderModule(device, &info, NULL, &mod));
  free(code);
  return mod;
}

static void create_compute_pipeline_spec(
      VkDevice device, VkShaderModule mod, uint32_t numBindings,
      uint32_t pushConstSize, const VkSpecializationInfo *specInfo,
      VkDescriptorSetLayout *dsl, VkPipelineLayout *pl, VkPipeline *pipe) {
    VkDescriptorSetLayoutBinding bindings[8];
    for (uint32_t i = 0; i < numBindings; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dslInfo = {0};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = numBindings;
    dslInfo.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, dsl));

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = pushConstSize;

    VkPipelineLayoutCreateInfo plInfo = {0};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = dsl;
    plInfo.pushConstantRangeCount = (pushConstSize > 0) ? 1 : 0;
    plInfo.pPushConstantRanges = (pushConstSize > 0) ? &pcr : NULL;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, pl));

    VkComputePipelineCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.stage.pSpecializationInfo = specInfo;
    cpInfo.layout = *pl;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL,
                                      pipe));
  }

  static void create_compute_pipeline(
      VkDevice device, VkShaderModule mod, uint32_t numBindings,
      uint32_t pushConstSize, VkDescriptorSetLayout *dsl, VkPipelineLayout *pl,
      VkPipeline *pipe) {
    create_compute_pipeline_spec(device, mod, numBindings, pushConstSize, NULL,
                                 dsl, pl, pipe);
  }

  static VkDescriptorSet alloc_ds(VkDevice device, VkDescriptorPool pool,
                                  VkDescriptorSetLayout dsl) {
    VkDescriptorSetAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &ai, &ds));
    return ds;
  }

  static void update_ds(VkDevice device, VkDescriptorSet ds, uint32_t n,
                        const VkBuffer *bufs, const VkDeviceSize *offsets,
                        const VkDeviceSize *ranges) {
    VkDescriptorBufferInfo bi[8];
    VkWriteDescriptorSet w[8];
    for (uint32_t i = 0; i < n; ++i) {
      bi[i].buffer = bufs[i];
      bi[i].offset = offsets[i];
      bi[i].range = ranges[i];
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].pNext = NULL;
      w[i].dstSet = ds;
      w[i].dstBinding = i;
      w[i].dstArrayElement = 0;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pImageInfo = NULL;
      w[i].pBufferInfo = &bi[i];
      w[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets(device, n, w, 0, NULL);
  }

  static void cmd_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
  }

  typedef struct {
    VkDescriptorSet ds_attn_norm; // Used for Layer 0 only
    VkDescriptorSet ds_qkv;
    VkDescriptorSet ds_rope;
    VkDescriptorSet ds_kv_append;
    VkDescriptorSet ds_attn;
    VkDescriptorSet ds_o;
    VkDescriptorSet ds_res1_ffn_norm;
    VkDescriptorSet ds_ffn_fused;
    VkDescriptorSet ds_down;
    VkDescriptorSet ds_res2_next_norm;
    VkDeviceSize kv_k_off;
    VkDeviceSize kv_v_off;
  } LayerDescSets;

  int mainCRTStartup() {
    puts("====================================================================="
         "=="
         "=========");
    puts("   Vulkan LLM Decode: Qwen2.5-0.5B-Instruct (Q4_0) Real GGUF "
         "Inference "
         "Engine   ");
    puts("====================================================================="
         "=="
         "=========");

    const char *modelPath = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    GgufFile gf;
    double t_load0 = get_time_sec();
    if (!gguf_load_file(modelPath, &gf))
      exit(1);
    double t_load1 = get_time_sec();

    uint32_t H = gf.config.hidden_size;
    uint32_t I = gf.config.intermediate_size;
    uint32_t Q_HEADS = gf.config.num_heads;
    uint32_t KV_HEADS = gf.config.num_kv_heads;
    uint32_t HEAD_DIM = gf.config.head_dim;
    uint32_t KV_DIM = KV_HEADS * HEAD_DIM;
    uint32_t LAYERS = gf.config.block_count;
    uint32_t VOCAB = gf.config.vocab_size;

    char cmdline[256];
    cmdline[0] = 0;
    {
      void *cf = fopen("qwen_cfg.txt", "rb");
      if (cf) {
        size_t n = fread(cmdline, 1, 255, cf);
        cmdline[n] = 0;
        fclose(cf);
      }
    }
    int use_spec = !str_has(cmdline, "--no-spec");
    int smart_draft = str_has(cmdline, "--smart-draft");
    int use_g128 = str_has(cmdline, "--g128");
    int use_q3_ffn = str_has(cmdline, "--q3-ffn");
    int mode_json = str_has(cmdline, "--json");
    int mode_code = str_has(cmdline, "--code");
    uint32_t prof_layer = str_has(cmdline, "--l12") ? 12u : 0u;
    uint32_t ACTIVE_VOCAB = parse_flag_uint(cmdline, "--vocab", VOCAB);
    if (ACTIVE_VOCAB == 0 || ACTIVE_VOCAB > VOCAB)
      ACTIVE_VOCAB = VOCAB;
    ACTIVE_VOCAB = (ACTIVE_VOCAB / 8u) * 8u;

    printf("Loaded GGUF:       %s (%.2f MB in %.2f s)\n", modelPath,
           (double)gf.file_size / (1024.0 * 1024.0), t_load1 - t_load0);
    printf("Architecture:      layers=%u, hidden=%u, inter=%u, q_heads=%u, "
           "kv_heads=%u, head_dim=%u\n",
           LAYERS, H, I, Q_HEADS, KV_HEADS, HEAD_DIM);
    printf("Config / Flags:    vocab=%u/%u, spec=%s%s, g128=%s, q3_ffn=%s, "
           "prof_layer=%u\n",
           ACTIVE_VOCAB, VOCAB, use_spec ? "ON" : "OFF",
           smart_draft ? "(smart)" : "", use_g128 ? "ON" : "OFF",
           use_q3_ffn ? "ON" : "OFF", prof_layer);

    if (LAYERS == 0 || LAYERS > MAX_LAYERS) {
      printf("Unsupported layer count: %u\n", LAYERS);
      exit(1);
    }

    // Keep token_embd.weight in its own CPU buffer so we can free the 408 MB
    // file_buf after GPU upload
    const GgufTensorInfo *t_tok_embd =
        gguf_find_tensor(&gf, "token_embd.weight");
    const GgufTensorInfo *t_out_norm =
        gguf_find_tensor(&gf, "output_norm.weight");
    const GgufTensorInfo *t_out_w = gguf_find_tensor(&gf, "output.weight");
    if (!t_tok_embd || !t_out_norm || !t_out_w) {
      puts("Error: Missing token_embd.weight, output_norm.weight, or "
           "output.weight!");
      exit(1);
    }

    size_t embd_bytes = ggml_tensor_bytes(t_tok_embd);
    uint8_t *cpu_tok_embd = (uint8_t *)malloc(embd_bytes);
    memcpy(cpu_tok_embd, gf.tensor_data + t_tok_embd->offset, embd_bytes);

    // 1. Initialize Vulkan
    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Qwen2.5VulkanDecode";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instInfo = {0};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, NULL);
    VkPhysicalDevice gpus[8];
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus);
    VkPhysicalDevice phys = gpus[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("GPU Device:        %s\n", props.deviceName);

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, NULL);
    VkQueueFamilyProperties qfs[16];
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfs);
    uint32_t qfIdx = 0;
    for (uint32_t i = 0; i < qfCount; ++i) {
      if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        qfIdx = i;
        break;
      }
    }

    float qPrio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {0};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = qfIdx;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &qPrio;

    VkPhysicalDeviceVulkan11Features f11 = {0};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.storageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12 = {0};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &f11;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderSubgroupExtendedTypes = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;

    VkPhysicalDeviceFeatures feats = {0};
    feats.shaderInt16 = VK_TRUE;

    VkDeviceCreateInfo devInfo = {0};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.pNext = &f12;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.pEnabledFeatures = &feats;

    VkDevice device;
    VK_CHECK(vkCreateDevice(phys, &devInfo, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, qfIdx, 0, &queue);

    // 2. Compute packed weights buffer size and upload all weights
    size_t align = 256;
    size_t total_weight_bytes = 0;
    for (uint32_t l = 0; l < LAYERS; ++l) {
      char nbuf[128];
      const char *q4_names[] = {"attn_q.weight",   "attn_k.weight",
                                "attn_v.weight",   "attn_output.weight",
                                "ffn_gate.weight", "ffn_up.weight",
                                "ffn_down.weight"};
      for (int k = 0; k < 7; ++k) {
        fmt_layer_name(nbuf, l, q4_names[k]);
        const GgufTensorInfo *t = gguf_find_tensor(&gf, nbuf);
        size_t nb = ggml_tensor_bytes(t) / 18;
        total_weight_bytes = align_up(total_weight_bytes, align) + nb * 16;
        total_weight_bytes = align_up(total_weight_bytes, align) + nb * 2;
        int is_ffn = (k >= 4);
        size_t wb = (is_ffn && use_q3_ffn) ? (nb * 12) : (nb * 16);
        size_t sb =
            ((is_ffn && use_q3_ffn) || use_g128) ? ((nb / 4) * 2) : (nb * 2);
        total_weight_bytes = align_up(total_weight_bytes, align) + wb;
        total_weight_bytes = align_up(total_weight_bytes, align) + sb;
      }
      // F16 norms & biases
      total_weight_bytes =
          align_up(total_weight_bytes, align) + H * 2; // attn_norm
      total_weight_bytes =
          align_up(total_weight_bytes, align) + H * 2; // ffn_norm
      total_weight_bytes =
          align_up(total_weight_bytes, align) + H * 2; // attn_q.bias
      total_weight_bytes =
          align_up(total_weight_bytes, align) + KV_DIM * 2; // attn_k.bias
      total_weight_bytes =
          align_up(total_weight_bytes, align) + KV_DIM * 2; // attn_v.bias
    }
    total_weight_bytes =
        align_up(total_weight_bytes, align) + H * 2; // output_norm
    {
      size_t nb_out = (size_t)ACTIVE_VOCAB * (H / 32);
      total_weight_bytes = align_up(total_weight_bytes, align) + nb_out * 16;
      total_weight_bytes = align_up(total_weight_bytes, align) + nb_out * 2;
      total_weight_bytes = align_up(total_weight_bytes, align) +
                           (use_g128 ? ((nb_out / 4) * 2) : (nb_out * 2));
    }

    VkBuffer weightsBuf;
    VkDeviceMemory weightsMem;
    create_buffer(device, phys, total_weight_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &weightsBuf, &weightsMem);

    uint8_t *wPtr = NULL;
    VK_CHECK(vkMapMemory(device, weightsMem, 0, total_weight_bytes, 0,
                         (void **)&wPtr));

    // Track offsets per layer (coalesced W + scales; Q/K/V packed contiguously)
    typedef struct {
      VkDeviceSize off_wqkv, sz_wqkv, off_sqkv, sz_sqkv, off_bqkv;
      VkDeviceSize off_wo, sz_wo, off_so, sz_so;
      VkDeviceSize off_wgate, sz_wgate, off_sgate, sz_sgate;
      VkDeviceSize off_wup, sz_wup, off_sup, sz_sup;
      VkDeviceSize off_wdown, sz_wdown, off_sdown, sz_sdown;
      VkDeviceSize off_attn_norm;
      VkDeviceSize off_ffn_norm;
    } LayerWeightOffs;

    LayerWeightOffs *lwo =
        (LayerWeightOffs *)malloc(sizeof(LayerWeightOffs) * LAYERS);
    size_t cur = 0;

    for (uint32_t l = 0; l < LAYERS; ++l) {
      char nbuf[128];
      const GgufTensorInfo *t;

      // Pack Q, K, V weights contiguously, then scales contiguously, so 1 GEMV
      // dispatch computes all 3
      {
        fmt_layer_name(nbuf, l, "attn_q.weight");
        const GgufTensorInfo *tq = gguf_find_tensor(&gf, nbuf);
        fmt_layer_name(nbuf, l, "attn_k.weight");
        const GgufTensorInfo *tk = gguf_find_tensor(&gf, nbuf);
        fmt_layer_name(nbuf, l, "attn_v.weight");
        const GgufTensorInfo *tv = gguf_find_tensor(&gf, nbuf);
        size_t nbq = ggml_tensor_bytes(tq) / 18;
        size_t nbk = ggml_tensor_bytes(tk) / 18;
        size_t nbv = ggml_tensor_bytes(tv) / 18;

        cur = align_up(cur, align);
        lwo[l].off_wqkv = cur;
        lwo[l].sz_wqkv = (nbq + nbk + nbv) * 16;
        cur += lwo[l].sz_wqkv;

        cur = align_up(cur, align);
        lwo[l].off_sqkv = cur;
        lwo[l].sz_sqkv = (nbq + nbk + nbv) * 2;
        lwo[l].sz_sqkv =
            use_g128 ? (((nbq + nbk + nbv) / 4) * 2) : ((nbq + nbk + nbv) * 2);
        cur += lwo[l].sz_sqkv;

        uint32_t *wq_dst = (uint32_t *)(wPtr + lwo[l].off_wqkv);
        uint16_t *sq_dst = (uint16_t *)(wPtr + lwo[l].off_sqkv);
        repack_q4_0_coalesced(gf.tensor_data + tq->offset, nbq, wq_dst, sq_dst);
        repack_q4_0_coalesced(gf.tensor_data + tk->offset, nbk,
                              wq_dst + nbq * 4, sq_dst + nbq);
        repack_q4_0_coalesced(gf.tensor_data + tv->offset, nbv,
                              wq_dst + (nbq + nbk) * 4, sq_dst + nbq + nbk);
        if (use_g128) {
          repack_q4_0_to_g128_coalesced(gf.tensor_data + tq->offset, nbq,
                                        wq_dst, sq_dst);
          repack_q4_0_to_g128_coalesced(gf.tensor_data + tk->offset, nbk,
                                        wq_dst + nbq * 4, sq_dst + nbq / 4);
          repack_q4_0_to_g128_coalesced(gf.tensor_data + tv->offset, nbv,
                                        wq_dst + (nbq + nbk) * 4,
                                        sq_dst + (nbq + nbk) / 4);
        } else {
          repack_q4_0_coalesced(gf.tensor_data + tq->offset, nbq, wq_dst,
                                sq_dst);
          repack_q4_0_coalesced(gf.tensor_data + tk->offset, nbk,
                                wq_dst + nbq * 4, sq_dst + nbq);
          repack_q4_0_coalesced(gf.tensor_data + tv->offset, nbv,
                                wq_dst + (nbq + nbk) * 4, sq_dst + nbq + nbk);
        }
      }

#define REPACK_Q4_TENSOR(suffix, off_w, sz_w, off_s, sz_s, is_ffn)             \
  fmt_layer_name(nbuf, l, suffix);                                             \
  t = gguf_find_tensor(&gf, nbuf);                                             \
  {                                                                            \
    size_t nb = ggml_tensor_bytes(t) / 18;                                     \
    int q3 = (is_ffn) && use_q3_ffn;                                           \
    cur = align_up(cur, align);                                                \
    lwo[l].off_w = cur;                                                        \
    lwo[l].sz_w = q3 ? (nb * 12) : (nb * 16);                                  \
    cur += lwo[l].sz_w;                                                        \
    cur = align_up(cur, align);                                                \
    lwo[l].off_s = cur;                                                        \
    lwo[l].sz_s = (q3 || use_g128) ? ((nb / 4) * 2) : (nb * 2);                \
    cur += lwo[l].sz_s;                                                        \
    if (q3) {                                                                  \
      repack_q4_0_to_q3_g128(gf.tensor_data + t->offset, nb,                   \
                             (uint32_t *)(wPtr + lwo[l].off_w),                \
                             (uint16_t *)(wPtr + lwo[l].off_s));               \
    } else if (use_g128) {                                                     \
      repack_q4_0_to_g128_coalesced(gf.tensor_data + t->offset, nb,            \
                                    (uint32_t *)(wPtr + lwo[l].off_w),         \
                                    (uint16_t *)(wPtr + lwo[l].off_s));        \
    } else {                                                                   \
      repack_q4_0_coalesced(gf.tensor_data + t->offset, nb,                    \
                            (uint32_t *)(wPtr + lwo[l].off_w),                 \
                            (uint16_t *)(wPtr + lwo[l].off_s));                \
    }                                                                          \
  }

    REPACK_Q4_TENSOR("attn_output.weight", off_wo, sz_wo, off_so, sz_so, 0);
    REPACK_Q4_TENSOR("ffn_gate.weight", off_wgate, sz_wgate, off_sgate,
                     sz_sgate, 1);
    REPACK_Q4_TENSOR("ffn_up.weight", off_wup, sz_wup, off_sup, sz_sup, 1);
    REPACK_Q4_TENSOR("ffn_down.weight", off_wdown, sz_wdown, off_sdown,
                     sz_sdown, 1);
#undef REPACK_Q4_TENSOR

#define COPY_F32_TO_F16(suffix, off_field, count)                              \
  fmt_layer_name(nbuf, l, suffix);                                             \
  t = gguf_find_tensor(&gf, nbuf);                                             \
  cur = align_up(cur, align);                                                  \
  lwo[l].off_field = cur;                                                      \
  {                                                                            \
    const float *src_f32 = (const float *)(gf.tensor_data + t->offset);        \
    uint16_t *dst_f16 = (uint16_t *)(wPtr + cur);                              \
    for (uint32_t idx = 0; idx < (count); ++idx)                               \
                     dst_f16[idx] = float_to_half(src_f32[idx]);               \
  }                                                                            \
  cur += (size_t)(count) * 2;

                     COPY_F32_TO_F16("attn_norm.weight", off_attn_norm, H);
                     COPY_F32_TO_F16("ffn_norm.weight", off_ffn_norm, H);
#undef COPY_F32_TO_F16

                     // Pack Q, K, V biases contiguously
                     cur = align_up(cur, align);
                     lwo[l].off_bqkv = cur;
                     {
                       const char *bnames[3] = {"attn_q.bias", "attn_k.bias",
                                                "attn_v.bias"};
                       uint32_t bcounts[3] = {H, KV_DIM, KV_DIM};
                       uint16_t *dst_f16 = (uint16_t *)(wPtr + cur);
                       size_t boff = 0;
                       for (int bi = 0; bi < 3; ++bi) {
                         fmt_layer_name(nbuf, l, bnames[bi]);
                         t = gguf_find_tensor(&gf, nbuf);
                         const float *src_f32 =
                             (const float *)(gf.tensor_data + t->offset);
                         for (uint32_t idx = 0; idx < bcounts[bi]; ++idx)
                           dst_f16[boff++] = float_to_half(src_f32[idx]);
                       }
                       cur += (size_t)(H + 2 * KV_DIM) * 2;
                     }
    }

    cur = align_up(cur, align);
    VkDeviceSize off_out_norm = cur;
    {
      const float *src_f32 =
          (const float *)(gf.tensor_data + t_out_norm->offset);
      uint16_t *dst_f16 = (uint16_t *)(wPtr + cur);
      for (uint32_t idx = 0; idx < H; ++idx)
        dst_f16[idx] = float_to_half(src_f32[idx]);
    }
    cur += (size_t)H * 2;

    size_t nb_out = (size_t)ACTIVE_VOCAB * (H / 32);
    cur = align_up(cur, align);
    VkDeviceSize off_out_w = cur;
    VkDeviceSize sz_out_w = nb_out * 16;
    cur += (size_t)sz_out_w;
    cur = align_up(cur, align);
    VkDeviceSize off_out_s = cur;
    VkDeviceSize sz_out_s = use_g128 ? ((nb_out / 4) * 2) : (nb_out * 2);
    cur += (size_t)sz_out_s;
    if (use_g128) {
      requant_q8_0_to_q4_g128_coalesced(gf.tensor_data + t_out_w->offset,
                                        nb_out, (uint32_t *)(wPtr + off_out_w),
                                        (uint16_t *)(wPtr + off_out_s));
    } else {
      requant_q8_0_to_q4_coalesced(gf.tensor_data + t_out_w->offset, nb_out,
                                   (uint32_t *)(wPtr + off_out_w),
                                   (uint16_t *)(wPtr + off_out_s));
    }

    vkUnmapMemory(device, weightsMem);

    // Free the 408 MB file buffer now that GPU weights & CPU token_embd are
    // copied
    if (gf.file_buf) {
      free(gf.file_buf);
      gf.file_buf = NULL;
    }

    // 3. KV Cache Buffer (all 24 layers)
    VkDeviceSize layer_kv_bytes = align_up((size_t)MAX_SEQ * KV_DIM * 2, align);
    VkDeviceSize total_kv_bytes = (VkDeviceSize)LAYERS * 2 * layer_kv_bytes;
    VkBuffer kvBuf;
    VkDeviceMemory kvMem;
    create_buffer(device, phys, total_kv_bytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &kvBuf, &kvMem);

    // 4. Scratch Buffer (persistent mapped)
    VkDeviceSize s_cur = 0;
#define ALLOC_SCRATCH(var_off, var_sz, bytes)                                  \
  VkDeviceSize var_off = s_cur;                                                \
  VkDeviceSize var_sz = (bytes);                                               \
  s_cur = align_up(s_cur + var_sz, align);

#define MAX_BATCH 4u
#define ARGMAX_WG 128u
    ALLOC_SCRATCH(off_x, sz_x, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_x_norm, sz_x_norm, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_qkv, sz_qkv, MAX_BATCH * (H + 2 * KV_DIM) * 2);
    VkDeviceSize off_q = off_qkv;
    VkDeviceSize off_k = off_qkv + H * 2;
    VkDeviceSize off_v = off_qkv + (H + KV_DIM) * 2;
    ALLOC_SCRATCH(off_attn_out, sz_attn_out, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_o_out, sz_o_out, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_x_mid, sz_x_mid, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_swiglu, sz_swiglu, MAX_BATCH * I * 2);
    ALLOC_SCRATCH(off_down_out, sz_down_out, MAX_BATCH * H * 2);
    ALLOC_SCRATCH(off_logits, sz_logits, MAX_BATCH * VOCAB * 2);
    ALLOC_SCRATCH(off_argmax_vals, sz_argmax_vals, MAX_BATCH * ARGMAX_WG * 4);
    ALLOC_SCRATCH(off_argmax_idxs, sz_argmax_idxs, MAX_BATCH * ARGMAX_WG * 4);
    ALLOC_SCRATCH(off_out_tok, sz_out_tok, MAX_BATCH * 4);
    ALLOC_SCRATCH(off_pos, sz_pos, 4);
#undef ALLOC_SCRATCH

    VkBuffer scratchBuf;
    VkDeviceMemory scratchMem;
    create_buffer(device, phys, s_cur,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &scratchBuf, &scratchMem);

    uint8_t *scratchPtr = NULL;
    VK_CHECK(
        vkMapMemory(device, scratchMem, 0, s_cur, 0, (void **)&scratchPtr));

    // 5. Load shaders & create compute pipelines
    VkShaderModule modRms = load_shader(device, "spirv/rmsnorm.spv");
    VkShaderModule modResRms =
        load_shader(device, "spirv/residual_rmsnorm.spv");
    VkShaderModule modQ4 =
        load_shader(device, "spirv/gemv_int4_group32_slm.spv");
    VkShaderModule modQ4K896 =
        load_shader(device, "spirv/gemv_int4_k896_slm.spv");
    VkShaderModule modQ4Bias =
        load_shader(device, "spirv/gemv_int4_group32_bias.spv");
    VkShaderModule modRope = load_shader(device, "spirv/rope_dyn.spv");
    VkShaderModule modKvAppend = load_shader(device, "spirv/kv_append_dyn.spv");
    VkShaderModule modAttn =
        load_shader(device, "spirv/attention_decode_dyn.spv");
    VkShaderModule modFfnFused =
        load_shader(device, use_q3_ffn ? "spirv/gemv_q3_swiglu_slm.spv"
                                       : "spirv/gemv_q4_swiglu_slm.spv");
    VkShaderModule modDown =
        use_q3_ffn ? load_shader(device, "spirv/gemv_int3_down_slm.spv")
                   : VK_NULL_HANDLE;
    VkShaderModule modArgmax1 = load_shader(device, "spirv/argmax_pass1.spv");
    VkShaderModule modArgmax2 = load_shader(device, "spirv/argmax_pass2.spv");

    VkShaderModule modResRmsB =
        load_shader(device, "spirv/residual_rmsnorm_batch.spv");
    VkShaderModule modQ4B =
        load_shader(device, "spirv/gemm_int4_group32_batch.spv");
    VkShaderModule modQ4K896B =
        load_shader(device, "spirv/gemm_int4_k896_batch.spv");
    VkShaderModule modQ4BiasB =
        load_shader(device, "spirv/gemm_int4_group32_bias_batch.spv");
    VkShaderModule modRopeB = load_shader(device, "spirv/rope_dyn_batch.spv");
    VkShaderModule modKvAppendB =
        load_shader(device, "spirv/kv_append_dyn_batch.spv");
    VkShaderModule modAttnB =
        load_shader(device, "spirv/attention_decode_dyn_batch.spv");
    VkShaderModule modFfnFusedB =
        load_shader(device, use_q3_ffn ? "spirv/gemm_q3_swiglu_batch.spv"
                                       : "spirv/gemm_q4_swiglu_batch.spv");
    VkShaderModule modDownB =
        use_q3_ffn ? load_shader(device, "spirv/gemm_int3_down_batch.spv")
                   : VK_NULL_HANDLE;
    VkShaderModule modArgmax1B =
        load_shader(device, "spirv/argmax_pass1_batch.spv");
    VkShaderModule modArgmax2B =
        load_shader(device, "spirv/argmax_pass2_batch.spv");

    VkDescriptorSetLayout dslRms, dslResRms, dslQ4, dslQ4K896, dslQ4Bias,
        dslRope, dslKvAppend, dslAttn, dslFfnFused, dslDown, dslArgmax1,
        dslArgmax2;
    VkPipelineLayout plRms, plResRms, plQ4, plQ4K896, plQ4Bias, plRope,
        plKvAppend, plAttn, plFfnFused, plDown, plArgmax1, plArgmax2;
    VkPipeline pipeRms, pipeResRms, pipeQ4, pipeQ4K896, pipeQ4Bias, pipeRope,
        pipeKvAppend, pipeAttn, pipeFfnFused, pipeDown, pipeArgmax1,
        pipeArgmax2;

    VkDescriptorSetLayout dslResRmsB, dslQ4B, dslQ4K896B, dslQ4BiasB, dslRopeB,
        dslKvAppendB, dslAttnB, dslFfnFusedB, dslDownB, dslArgmax1B,
        dslArgmax2B;
    VkPipelineLayout plResRmsB, plQ4B, plQ4K896B, plQ4BiasB, plRopeB,
        plKvAppendB, plAttnB, plFfnFusedB, plDownB, plArgmax1B, plArgmax2B;
    VkPipeline pipeResRmsB, pipeQ4B, pipeQ4K896B, pipeQ4BiasB, pipeRopeB,
        pipeKvAppendB, pipeAttnB, pipeFfnFusedB, pipeDownB, pipeArgmax1B,
        pipeArgmax2B;

    uint32_t scale_shift_val = use_g128 ? 2u : 0u;
    VkSpecializationMapEntry scaleMapEntry = {0, 0, sizeof(uint32_t)};
    VkSpecializationInfo scaleSpecInfo = {1, &scaleMapEntry, sizeof(uint32_t),
                                          &scale_shift_val};

    create_compute_pipeline(device, modRms, 3, 12, &dslRms, &plRms, &pipeRms);
    create_compute_pipeline(device, modResRms, 5, 12, &dslResRms, &plResRms,
                            &pipeResRms);
    create_compute_pipeline(device, modQ4, 4, 8, &dslQ4, &plQ4, &pipeQ4);
    create_compute_pipeline(device, modQ4K896, 4, 8, &dslQ4K896, &plQ4K896,
                            &pipeQ4K896);
    create_compute_pipeline(device, modQ4Bias, 5, 8, &dslQ4Bias, &plQ4Bias,
                            &pipeQ4Bias);
    create_compute_pipeline_spec(device, modQ4, 4, 8, &scaleSpecInfo, &dslQ4,
                                 &plQ4, &pipeQ4);
    create_compute_pipeline_spec(device, modQ4K896, 4, 8, &scaleSpecInfo,
                                 &dslQ4K896, &plQ4K896, &pipeQ4K896);
    create_compute_pipeline_spec(device, modQ4Bias, 5, 8, &scaleSpecInfo,
                                 &dslQ4Bias, &plQ4Bias, &pipeQ4Bias);
    create_compute_pipeline(device, modRope, 3, 16, &dslRope, &plRope,
                            &pipeRope);
    create_compute_pipeline(device, modKvAppend, 5, 4, &dslKvAppend,
                            &plKvAppend, &pipeKvAppend);
    create_compute_pipeline(device, modAttn, 5, 16, &dslAttn, &plAttn,
                            &pipeAttn);
    create_compute_pipeline(device, modFfnFused, 6, 8, &dslFfnFused,
                            &plFfnFused, &pipeFfnFused);
    create_compute_pipeline_spec(device, modFfnFused, 6, 8,
                                 use_q3_ffn ? NULL : &scaleSpecInfo,
                                 &dslFfnFused, &plFfnFused, &pipeFfnFused);
    if (use_q3_ffn) {
      create_compute_pipeline(device, modDown, 4, 8, &dslDown, &plDown,
                              &pipeDown);
    } else {
      dslDown = dslQ4;
      plDown = plQ4;
      pipeDown = pipeQ4;
    }
    create_compute_pipeline(device, modArgmax1, 3, 4, &dslArgmax1, &plArgmax1,
                            &pipeArgmax1);
    create_compute_pipeline(device, modArgmax2, 3, 4, &dslArgmax2, &plArgmax2,
                            &pipeArgmax2);

    create_compute_pipeline(device, modResRmsB, 5, 12, &dslResRmsB, &plResRmsB,
                            &pipeResRmsB);
    create_compute_pipeline(device, modQ4B, 4, 12, &dslQ4B, &plQ4B, &pipeQ4B);
    create_compute_pipeline(device, modQ4K896B, 4, 12, &dslQ4K896B, &plQ4K896B,
                            &pipeQ4K896B);
    create_compute_pipeline(device, modQ4BiasB, 5, 12, &dslQ4BiasB, &plQ4BiasB,
                            &pipeQ4BiasB);
    create_compute_pipeline_spec(device, modQ4B, 4, 12, &scaleSpecInfo, &dslQ4B,
                                 &plQ4B, &pipeQ4B);
    create_compute_pipeline_spec(device, modQ4K896B, 4, 12, &scaleSpecInfo,
                                 &dslQ4K896B, &plQ4K896B, &pipeQ4K896B);
    create_compute_pipeline_spec(device, modQ4BiasB, 5, 12, &scaleSpecInfo,
                                 &dslQ4BiasB, &plQ4BiasB, &pipeQ4BiasB);
    create_compute_pipeline(device, modRopeB, 3, 16, &dslRopeB, &plRopeB,
                            &pipeRopeB);
    create_compute_pipeline(device, modKvAppendB, 5, 8, &dslKvAppendB,
                            &plKvAppendB, &pipeKvAppendB);
    create_compute_pipeline(device, modAttnB, 5, 16, &dslAttnB, &plAttnB,
                            &pipeAttnB);
    create_compute_pipeline(device, modFfnFusedB, 6, 12, &dslFfnFusedB,
                            &plFfnFusedB, &pipeFfnFusedB);
    create_compute_pipeline_spec(device, modFfnFusedB, 6, 12,
                                 use_q3_ffn ? NULL : &scaleSpecInfo,
                                 &dslFfnFusedB, &plFfnFusedB, &pipeFfnFusedB);
    if (use_q3_ffn) {
      create_compute_pipeline(device, modDownB, 4, 12, &dslDownB, &plDownB,
                              &pipeDownB);
    } else {
      dslDownB = dslQ4B;
      plDownB = plQ4B;
      pipeDownB = pipeQ4B;
    }
    create_compute_pipeline(device, modArgmax1B, 3, 4, &dslArgmax1B,
                            &plArgmax1B, &pipeArgmax1B);
    create_compute_pipeline(device, modArgmax2B, 3, 4, &dslArgmax2B,
                            &plArgmax2B, &pipeArgmax2B);

    // 6. Descriptor Pool & pre-populated descriptor sets for all 24 layers +
    // head
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048};
    VkDescriptorPoolCreateInfo dpInfo = {0};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 512;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool));

    LayerDescSets *lds =
        (LayerDescSets *)malloc(sizeof(LayerDescSets) * LAYERS);
    for (uint32_t l = 0; l < LAYERS; ++l) {
      lds[l].kv_k_off = (VkDeviceSize)(l * 2 + 0) * layer_kv_bytes;
      lds[l].kv_v_off = (VkDeviceSize)(l * 2 + 1) * layer_kv_bytes;

      if (l == 0) {
        lds[l].ds_attn_norm = alloc_ds(device, descPool, dslRms);
        update_ds(device, lds[l].ds_attn_norm, 3,
                  (VkBuffer[]){scratchBuf, weightsBuf, scratchBuf},
                  (VkDeviceSize[]){off_x, lwo[l].off_attn_norm, off_x_norm},
                  (VkDeviceSize[]){sz_x, H * 2, sz_x_norm});
      } else {
        lds[l].ds_attn_norm = VK_NULL_HANDLE;
      }

      lds[l].ds_qkv = alloc_ds(device, descPool, dslQ4Bias);
      update_ds(device, lds[l].ds_qkv, 5,
                (VkBuffer[]){scratchBuf, weightsBuf, weightsBuf, weightsBuf,
                             scratchBuf},
                (VkDeviceSize[]){off_x_norm, lwo[l].off_wqkv, lwo[l].off_sqkv,
                                 lwo[l].off_bqkv, off_qkv},
                (VkDeviceSize[]){sz_x_norm, lwo[l].sz_wqkv, lwo[l].sz_sqkv,
                                 (H + 2 * KV_DIM) * 2, sz_qkv});

      lds[l].ds_rope = alloc_ds(device, descPool, dslRope);
      update_ds(device, lds[l].ds_rope, 3,
                (VkBuffer[]){scratchBuf, scratchBuf, scratchBuf},
                (VkDeviceSize[]){off_q, off_k, off_pos},
                (VkDeviceSize[]){sz_qkv, sz_qkv - H * 2, 4});

      lds[l].ds_kv_append = alloc_ds(device, descPool, dslKvAppend);
      update_ds(device, lds[l].ds_kv_append, 5,
                (VkBuffer[]){scratchBuf, scratchBuf, kvBuf, kvBuf, scratchBuf},
                (VkDeviceSize[]){off_k, off_v, lds[l].kv_k_off, lds[l].kv_v_off,
                                 off_pos},
                (VkDeviceSize[]){sz_qkv - H * 2, sz_qkv - (H + KV_DIM) * 2,
                                 layer_kv_bytes, layer_kv_bytes, 4});

      lds[l].ds_attn = alloc_ds(device, descPool, dslAttn);
      update_ds(device, lds[l].ds_attn, 5,
                (VkBuffer[]){scratchBuf, kvBuf, kvBuf, scratchBuf, scratchBuf},
                (VkDeviceSize[]){off_q, lds[l].kv_k_off, lds[l].kv_v_off,
                                 off_attn_out, off_pos},
                (VkDeviceSize[]){sz_qkv, layer_kv_bytes, layer_kv_bytes,
                                 sz_attn_out, 4});

      lds[l].ds_o = alloc_ds(device, descPool, dslQ4);
      update_ds(
          device, lds[l].ds_o, 4,
          (VkBuffer[]){scratchBuf, weightsBuf, weightsBuf, scratchBuf},
          (VkDeviceSize[]){off_attn_out, lwo[l].off_wo, lwo[l].off_so,
                           off_o_out},
          (VkDeviceSize[]){sz_attn_out, lwo[l].sz_wo, lwo[l].sz_so, sz_o_out});

      lds[l].ds_res1_ffn_norm = alloc_ds(device, descPool, dslResRms);
      update_ds(device, lds[l].ds_res1_ffn_norm, 5,
                (VkBuffer[]){scratchBuf, scratchBuf, weightsBuf, scratchBuf,
                             scratchBuf},
                (VkDeviceSize[]){off_x, off_o_out, lwo[l].off_ffn_norm,
                                 off_x_mid, off_x_norm},
                (VkDeviceSize[]){sz_x, sz_o_out, H * 2, sz_x_mid, sz_x_norm});

      lds[l].ds_ffn_fused = alloc_ds(device, descPool, dslFfnFused);
      update_ds(device, lds[l].ds_ffn_fused, 6,
                (VkBuffer[]){scratchBuf, weightsBuf, weightsBuf, weightsBuf,
                             weightsBuf, scratchBuf},
                (VkDeviceSize[]){off_x_norm, lwo[l].off_wgate, lwo[l].off_sgate,
                                 lwo[l].off_wup, lwo[l].off_sup, off_swiglu},
                (VkDeviceSize[]){sz_x_norm, lwo[l].sz_wgate, lwo[l].sz_sgate,
                                 lwo[l].sz_wup, lwo[l].sz_sup, sz_swiglu});

      lds[l].ds_down = alloc_ds(device, descPool, dslQ4);
      lds[l].ds_down = alloc_ds(device, descPool, dslDown);
      update_ds(device, lds[l].ds_down, 4,
                (VkBuffer[]){scratchBuf, weightsBuf, weightsBuf, scratchBuf},
                (VkDeviceSize[]){off_swiglu, lwo[l].off_wdown, lwo[l].off_sdown,
                                 off_down_out},
                (VkDeviceSize[]){sz_swiglu, lwo[l].sz_wdown, lwo[l].sz_sdown,
                                 sz_down_out});

      VkDeviceSize next_norm_off =
          (l + 1 < LAYERS) ? lwo[l + 1].off_attn_norm : off_out_norm;
      lds[l].ds_res2_next_norm = alloc_ds(device, descPool, dslResRms);
      update_ds(
          device, lds[l].ds_res2_next_norm, 5,
          (VkBuffer[]){scratchBuf, scratchBuf, weightsBuf, scratchBuf,
                       scratchBuf},
          (VkDeviceSize[]){off_x_mid, off_down_out, next_norm_off, off_x,
                           off_x_norm},
          (VkDeviceSize[]){sz_x_mid, sz_down_out, H * 2, sz_x, sz_x_norm});
    }

    VkDescriptorSet ds_lm_head = alloc_ds(device, descPool, dslQ4);
    update_ds(device, ds_lm_head, 4,
              (VkBuffer[]){scratchBuf, weightsBuf, weightsBuf, scratchBuf},
              (VkDeviceSize[]){off_x_norm, off_out_w, off_out_s, off_logits},
              (VkDeviceSize[]){sz_x_norm, sz_out_w, sz_out_s, sz_logits});

    VkDescriptorSet ds_argmax1 = alloc_ds(device, descPool, dslArgmax1);
    update_ds(device, ds_argmax1, 3,
              (VkBuffer[]){scratchBuf, scratchBuf, scratchBuf},
              (VkDeviceSize[]){off_logits, off_argmax_vals, off_argmax_idxs},
              (VkDeviceSize[]){sz_logits, sz_argmax_vals, sz_argmax_idxs});

    VkDescriptorSet ds_argmax2 = alloc_ds(device, descPool, dslArgmax2);
    update_ds(device, ds_argmax2, 3,
              (VkBuffer[]){scratchBuf, scratchBuf, scratchBuf},
              (VkDeviceSize[]){off_argmax_vals, off_argmax_idxs, off_out_tok},
              (VkDeviceSize[]){sz_argmax_vals, sz_argmax_idxs, MAX_BATCH * 4});

    // 7. Command Pool & Command Buffer
    VkCommandPoolCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.queueFamilyIndex = qfIdx;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &cpInfo, NULL, &cmdPool));

    VkCommandBufferAllocateInfo cbInfo = {0};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = cmdPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 5;
    VkCommandBuffer cbs[5];
    VK_CHECK(vkAllocateCommandBuffers(device, &cbInfo, cbs));
    VkCommandBuffer cb_prefill = cbs[0];
    VkCommandBuffer cb_decode = cbs[1];
    // cbs[2..4] = speculative verify CBs for M = 2, 3, 4

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, NULL, &fence));

    VkQueryPoolCreateInfo qpInfo = {0};
    qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = 14;
    VkQueryPool queryPool;
    VK_CHECK(vkCreateQueryPool(device, &qpInfo, NULL, &queryPool));
    uint64_t prof_ts[14] = {0};
    int prof_captured = 0;
    double host_record_ms = 0.0;

    // Push constant structs (100% static once recorded!)
    struct {
      uint32_t K;
      uint32_t K_vec4;
      float eps;
    } pcRms = {H, H / 4, gf.config.rms_norm_eps};
    struct {
      uint32_t K_blocks;
      uint32_t N;
    } pcQ = {H / 32, H};
    struct {
      uint32_t K_blocks;
      uint32_t N;
    } pcQKV = {H / 32, H + 2 * KV_DIM};
    struct {
      uint32_t K_blocks;
      uint32_t N;
    } pcGateUp = {H / 32, I};
    struct {
      uint32_t K_blocks;
      uint32_t N;
    } pcDown = {I / 32, H};
    struct {
      uint32_t K_blocks;
      uint32_t N;
    } pcLM = {H / 32, ACTIVE_VOCAB};
    struct {
      uint32_t qh;
      uint32_t kvh;
      uint32_t hd;
      float theta;
    } pcRope = {Q_HEADS, KV_HEADS, HEAD_DIM, gf.config.rope_theta};
    struct {
      uint32_t qh;
      uint32_t kvh;
      uint32_t hd;
      float scale;
    } pcAttn = {Q_HEADS, KV_HEADS, HEAD_DIM, 1.0f / sqrt_f((float)HEAD_DIM)};
    uint32_t pcKvAppend = KV_DIM / 4;
    struct {
      uint32_t kv_dim_v4;
      uint32_t in_stride_v4;
    } pcKvAppendB = {KV_DIM / 4, (H + 2 * KV_DIM) / 4};
    uint32_t pcVocab = ACTIVE_VOCAB;
    uint32_t pcArgmaxWg = ARGMAX_WG;
    uint32_t ropePairs = (Q_HEADS + KV_HEADS) * (HEAD_DIM / 2);

    // Pre-record cb_prefill (mode 0), cb_decode (mode 1), and cb_verify[M=2..4]
    // (mode 2..4) ONCE at startup
    double t_rec0 = get_time_sec();
    for (int mode = 0; mode < 5; ++mode) {
      VkCommandBuffer cb = cbs[mode];
      int do_head = (mode >= 1);
      uint32_t M = (mode <= 1) ? 1u : (uint32_t)mode;
      int batched = (M > 1u);

      struct {
        uint32_t K_blocks;
        uint32_t N;
        uint32_t M;
      } pcQB = {H / 32, H, M};
      struct {
        uint32_t K_blocks;
        uint32_t N;
        uint32_t M;
      } pcQKVB = {H / 32, H + 2 * KV_DIM, M};
      struct {
        uint32_t K_blocks;
        uint32_t N;
        uint32_t M;
      } pcGateUpB = {H / 32, I, M};
      struct {
        uint32_t K_blocks;
        uint32_t N;
        uint32_t M;
      } pcDownB = {I / 32, H, M};
      struct {
        uint32_t K_blocks;
        uint32_t N;
        uint32_t M;
      } pcLMB = {H / 32, ACTIVE_VOCAB, M};

      VkCommandBufferBeginInfo bi = {0};
      bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      bi.flags = 0; // Reusable static command buffer
      VK_CHECK(vkBeginCommandBuffer(cb, &bi));
      if (mode >= 1)
        vkCmdResetQueryPool(cb, queryPool, 0, 14);

#define TS(idx)                                                                \
  if (mode == 1 && l == prof_layer)                                            \
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool,     \
                      (idx))

      for (uint32_t l = 0; l < LAYERS; ++l) {
        if (mode >= 1 && l == 0)
          vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              queryPool, 0);
        if (l == 0) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRms);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plRms, 0,
                                  1, &lds[l].ds_attn_norm, 0, NULL);
          vkCmdPushConstants(cb, plRms, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcRms);
          vkCmdDispatch(cb, M, 1, 1);
          cmd_barrier(cb);
        }
        TS(1);

        // Fused Q+K+V GEMV/GEMM + Bias
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4Bias);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4Bias,
                                  0, 1, &lds[l].ds_qkv, 0, NULL);
          vkCmdPushConstants(cb, plQ4Bias, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcQKV);
          vkCmdDispatch(cb, (H + 2 * KV_DIM + 7) / 8, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4BiasB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4BiasB,
                                  0, 1, &lds[l].ds_qkv, 0, NULL);
          vkCmdPushConstants(cb, plQ4BiasB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcQKVB);
          vkCmdDispatch(cb, (H + 2 * KV_DIM + 7) / 8, 1, 1);
        }
        cmd_barrier(cb);
        TS(2);

        // Dynamic RoPE on Q and K
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRope);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plRope, 0,
                                  1, &lds[l].ds_rope, 0, NULL);
          vkCmdPushConstants(cb, plRope, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                             &pcRope);
          vkCmdDispatch(cb, (ropePairs + 63) / 64, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRopeB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plRopeB,
                                  0, 1, &lds[l].ds_rope, 0, NULL);
          vkCmdPushConstants(cb, plRopeB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                             &pcRope);
          vkCmdDispatch(cb, (ropePairs + 63) / 64, M, 1);
        }
        cmd_barrier(cb);
        TS(3);

        // Dynamic KV Append into cache[pos..pos+M-1]
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeKvAppend);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plKvAppend, 0, 1, &lds[l].ds_kv_append, 0,
                                  NULL);
          vkCmdPushConstants(cb, plKvAppend, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                             &pcKvAppend);
          vkCmdDispatch(cb, (pcKvAppend + 63) / 64, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeKvAppendB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plKvAppendB, 0, 1, &lds[l].ds_kv_append, 0,
                                  NULL);
          vkCmdPushConstants(cb, plKvAppendB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcKvAppendB);
          vkCmdDispatch(cb, (pcKvAppend + 63) / 64, M, 1);
        }
        cmd_barrier(cb);
        TS(4);

        // Dynamic Causal FlashDecode GQA Attention
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeAttn);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plAttn, 0,
                                  1, &lds[l].ds_attn, 0, NULL);
          vkCmdPushConstants(cb, plAttn, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                             &pcAttn);
          vkCmdDispatch(cb, Q_HEADS, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeAttnB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plAttnB,
                                  0, 1, &lds[l].ds_attn, 0, NULL);
          vkCmdPushConstants(cb, plAttnB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                             &pcAttn);
          vkCmdDispatch(cb, Q_HEADS, M, 1);
        }
        cmd_barrier(cb);
        TS(5);

        // O-proj
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4K896);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4K896,
                                  0, 1, &lds[l].ds_o, 0, NULL);
          vkCmdPushConstants(cb, plQ4K896, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcQ);
          vkCmdDispatch(cb, (H + 7) / 8, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4K896B);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4K896B,
                                  0, 1, &lds[l].ds_o, 0, NULL);
          vkCmdPushConstants(cb, plQ4K896B, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcQB);
          vkCmdDispatch(cb, (H + 7) / 8, 1, 1);
        }
        cmd_barrier(cb);
        TS(6);

        // Fused Residual 1 + FFN RMSNorm
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeResRms);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plResRms,
                                  0, 1, &lds[l].ds_res1_ffn_norm, 0, NULL);
          vkCmdPushConstants(cb, plResRms, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcRms);
          vkCmdDispatch(cb, 1, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeResRmsB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plResRmsB,
                                  0, 1, &lds[l].ds_res1_ffn_norm, 0, NULL);
          vkCmdPushConstants(cb, plResRmsB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcRms);
          vkCmdDispatch(cb, M, 1, 1);
        }
        cmd_barrier(cb);
        TS(7);

        // Fused Gate + Up + SwiGLU
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFfnFused);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plFfnFused, 0, 1, &lds[l].ds_ffn_fused, 0,
                                  NULL);
          vkCmdPushConstants(cb, plFfnFused, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcGateUp);
          vkCmdDispatch(cb, (I + 7) / 8, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFfnFusedB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plFfnFusedB, 0, 1, &lds[l].ds_ffn_fused, 0,
                                  NULL);
          vkCmdPushConstants(cb, plFfnFusedB, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             12, &pcGateUpB);
          vkCmdDispatch(cb, (I + 7) / 8, 1, 1);
        }
        cmd_barrier(cb);
        TS(8);

        // Down GEMV / GEMM
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeDown);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plDown, 0,
                                  1, &lds[l].ds_down, 0, NULL);
          vkCmdPushConstants(cb, plDown, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcDown);
          vkCmdDispatch(cb, (H + 7) / 8, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeDownB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plDownB,
                                  0, 1, &lds[l].ds_down, 0, NULL);
          vkCmdPushConstants(cb, plDownB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcDownB);
          vkCmdDispatch(cb, (H + 7) / 8, 1, 1);
        }
        cmd_barrier(cb);
        TS(9);

        // Fused Residual 2 + Next Layer Attn RMSNorm (or Output RMSNorm on last
        // layer)
        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeResRms);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plResRms,
                                  0, 1, &lds[l].ds_res2_next_norm, 0, NULL);
          vkCmdPushConstants(cb, plResRms, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcRms);
          vkCmdDispatch(cb, 1, 1, 1);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeResRmsB);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plResRmsB,
                                  0, 1, &lds[l].ds_res2_next_norm, 0, NULL);
          vkCmdPushConstants(cb, plResRmsB, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcRms);
          vkCmdDispatch(cb, M, 1, 1);
        }
        cmd_barrier(cb);
        TS(10);
      }
#undef TS

      if (do_head) {
        if (mode == 1)
          vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              queryPool, 11);

        if (!batched) {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4K896);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4K896,
                                  0, 1, &ds_lm_head, 0, NULL);
          vkCmdPushConstants(cb, plQ4K896, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                             &pcLM);
          vkCmdDispatch(cb, (ACTIVE_VOCAB + 7) / 8, 1, 1);
          cmd_barrier(cb);
          if (mode == 1)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                queryPool, 12);

          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeArgmax1);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plArgmax1,
                                  0, 1, &ds_argmax1, 0, NULL);
          vkCmdPushConstants(cb, plArgmax1, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                             &pcVocab);
          vkCmdDispatch(cb, ARGMAX_WG, 1, 1);
          cmd_barrier(cb);

          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeArgmax2);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plArgmax2,
                                  0, 1, &ds_argmax2, 0, NULL);
          vkCmdPushConstants(cb, plArgmax2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                             &pcArgmaxWg);
          vkCmdDispatch(cb, 1, 1, 1);
          if (mode == 1)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                queryPool, 13);
        } else {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeQ4K896B);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plQ4K896B,
                                  0, 1, &ds_lm_head, 0, NULL);
          vkCmdPushConstants(cb, plQ4K896B, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12,
                             &pcLMB);
          vkCmdDispatch(cb, (ACTIVE_VOCAB + 7) / 8, 1, 1);
          cmd_barrier(cb);

          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeArgmax1B);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plArgmax1B, 0, 1, &ds_argmax1, 0, NULL);
          vkCmdPushConstants(cb, plArgmax1B, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                             &pcVocab);
          vkCmdDispatch(cb, ARGMAX_WG, M, 1);
          cmd_barrier(cb);

          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeArgmax2B);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  plArgmax2B, 0, 1, &ds_argmax2, 0, NULL);
          vkCmdPushConstants(cb, plArgmax2B, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                             &pcArgmaxWg);
          vkCmdDispatch(cb, 1, M, 1);
          vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              queryPool, 13);
        }
      }

      VK_CHECK(vkEndCommandBuffer(cb));
    }
    double t_rec1 = get_time_sec();
    host_record_ms = (t_rec1 - t_rec0) * 1000.0;

    // 8. Build ChatML prompt tokens
    int32_t tok_im_start = vocab_find_exact(&gf, "<|im_start|>");
    int32_t tok_im_end = vocab_find_exact(&gf, "<|im_end|>");
    if (tok_im_start < 0 || tok_im_end < 0) {
      puts("Error: Could not find <|im_start|> / <|im_end|> in vocab!");
      exit(1);
    }

    const char *user_prompt =
        "Explain how a GPU works in detail, covering "
        "cores, memory bandwidth, and parallel processing.";
    if (mode_json) {
      user_prompt =
          "Output a JSON array of 10 identical template user objects with keys "
          "id, name, role, active, and team, incrementing id from 1 to 10.";
    } else if (mode_code) {
      user_prompt =
          "Write a repetitive C switch statement mapping error codes 0 "
          "through 20 to ERR_0 through ERR_20 string literals.";
    }
    uint32_t prompt_tokens[512];
    uint32_t n_prompt = 0;
    prompt_tokens[n_prompt++] = (uint32_t)tok_im_start;
    n_prompt += encode_bpe_greedy(&gf, "user\n", &prompt_tokens[n_prompt], 16);
    n_prompt +=
        encode_bpe_greedy(&gf, user_prompt, &prompt_tokens[n_prompt], 400);
    prompt_tokens[n_prompt++] = (uint32_t)tok_im_end;
    n_prompt += encode_bpe_greedy(&gf, "\n", &prompt_tokens[n_prompt], 16);
    prompt_tokens[n_prompt++] = (uint32_t)tok_im_start;
    n_prompt +=
        encode_bpe_greedy(&gf, "assistant\n", &prompt_tokens[n_prompt], 16);

    printf("Prompt tokens:     %u (static CBs recorded once in %.2f ms)\n",
           n_prompt, host_record_ms);
    puts("---------------------------------------------------------------------"
         "--"
         "---------");
    printf("Prompt: %s\n", user_prompt);
    printf("Assistant: ");

    uint32_t token_history[MAX_SEQ];
    uint32_t hist_len = 0;

    size_t embd_row_bytes = (size_t)(H / 32) * 18;
    uint32_t max_gen = 160;
    uint32_t cur_token = prompt_tokens[0];
    uint32_t gen_count = 0;
    uint32_t spec_steps = 0;
    uint32_t spec_drafted = 0;
    uint32_t spec_accepted = 0;
    uint64_t total_gpu_ticks = 0;
    double t_decode_start = 0.0;

    // 1. Prefill prompt tokens 0 .. n_prompt - 2 (M=1, no head)
    uint32_t pos = 0;
    for (; pos + 1 < n_prompt && pos < MAX_SEQ; ++pos) {
      token_history[hist_len++] = cur_token;
      dequant_q4_0_row_to_fp16(cpu_tok_embd +
                                   (size_t)cur_token * embd_row_bytes,
                               H, (uint16_t *)(scratchPtr + off_x));
      *(uint32_t *)(scratchPtr + off_pos) = pos;

      VkSubmitInfo si = {0};
      si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cb_prefill;
      VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
      VK_CHECK(
          vkWaitForFences(device, 1, &fence, VK_TRUE, 0xFFFFFFFFFFFFFFFFULL));
      VK_CHECK(vkResetFences(device, 1, &fence));

      cur_token = prompt_tokens[pos + 1];
    }

    // Feed last prompt token (pos == n_prompt - 1) to get first generated token
    if (pos < MAX_SEQ) {
      token_history[hist_len++] = cur_token;
      dequant_q4_0_row_to_fp16(cpu_tok_embd +
                                   (size_t)cur_token * embd_row_bytes,
                               H, (uint16_t *)(scratchPtr + off_x));
      *(uint32_t *)(scratchPtr + off_pos) = pos;

      VkSubmitInfo si = {0};
      si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cb_decode;
      VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
      VK_CHECK(
          vkWaitForFences(device, 1, &fence, VK_TRUE, 0xFFFFFFFFFFFFFFFFULL));
      VK_CHECK(vkResetFences(device, 1, &fence));

      uint32_t first_tok = *(uint32_t *)(scratchPtr + off_out_tok);
      pos++;
      if (first_tok != gf.config.eos_token_id &&
          first_tok != (uint32_t)tok_im_end) {
        print_decoded_token(gf.vocab_tokens[first_tok]);
        token_history[hist_len++] = first_tok;
        cur_token = first_tok;
        gen_count++;
      } else {
        cur_token = first_tok;
      }
    }

    uint32_t corpus_toks[1024];
    uint32_t corpus_len = 0;
    if (smart_draft) {
      static const char *phrases[] = {" is a ",
                                      " is the ",
                                      " in the ",
                                      " of the ",
                                      " to the ",
                                      " for the ",
                                      " on the ",
                                      " and the ",
                                      " that the ",
                                      " with the ",
                                      " as a ",
                                      " from the ",
                                      " by the ",
                                      " it is ",
                                      " there is ",
                                      " there are ",
                                      " can be ",
                                      " will be ",
                                      " has been ",
                                      " have been ",
                                      " such as ",
                                      " for example, ",
                                      " in order to ",
                                      " as well as ",
                                      " on the other hand, ",
                                      " one of the ",
                                      " part of the ",
                                      " based on the ",
                                      " according to ",
                                      " in addition, ",
                                      " however, ",
                                      " therefore, ",
                                      " furthermore, ",
                                      " large language models",
                                      " machine learning",
                                      " artificial intelligence",
                                      " neural network",
                                      " graphics processing unit",
                                      " memory bandwidth",
                                      " compute shader",
                                      " speculative decoding",
                                      " key-value cache",
                                      " floating point",
                                      " quantized weights",
                                      " matrix multiplication",
                                      " attention mechanism",
                                      " transformer architecture"};
      uint32_t n_phr = (uint32_t)(sizeof(phrases) / sizeof(phrases[0]));
      for (uint32_t p = 0; p < n_phr && corpus_len + 32 < 1024; ++p) {
        corpus_len += encode_bpe_greedy(
            &gf, phrases[p], corpus_toks + corpus_len, 1024 - corpus_len);
        if (corpus_len < 1024)
          corpus_toks[corpus_len++] = gf.config.eos_token_id;
      }
    }

    // 2. Speculative Decode Loop (M = 1..4 with lossless n-gram self-drafting)
    t_decode_start = get_time_sec();
    int done = (cur_token == gf.config.eos_token_id ||
                cur_token == (uint32_t)tok_im_end);
    while (!done && gen_count < max_gen && pos < MAX_SEQ) {
      uint32_t draft_toks[MAX_BATCH - 1];
      uint32_t max_d = MAX_BATCH - 1;
      if (pos + 1 + max_d > MAX_SEQ)
        max_d = (MAX_SEQ > pos + 1) ? (MAX_SEQ - pos - 1) : 0;
      (use_spec && max_d > 0)
          ? draft_ngram(token_history, hist_len, draft_toks, max_d,
                        gf.config.eos_token_id, (uint32_t)tok_im_end)
          : 0;
      uint32_t n_draft = 0;
      if (use_spec && max_d > 0) {
        n_draft =
            smart_draft
                ? draft_ngram_smart(token_history, hist_len, corpus_toks,
                                    corpus_len, draft_toks, max_d,
                                    gf.config.eos_token_id,
                                    (uint32_t)tok_im_end)
                : draft_ngram(token_history, hist_len, draft_toks, max_d,
                              gf.config.eos_token_id, (uint32_t)tok_im_end);
      }
      uint32_t M = 1u + n_draft;

      dequant_q4_0_row_to_fp16(cpu_tok_embd +
                                   (size_t)cur_token * embd_row_bytes,
                               H, (uint16_t *)(scratchPtr + off_x));
      for (uint32_t d = 0; d < n_draft; ++d) {
        dequant_q4_0_row_to_fp16(
            cpu_tok_embd + (size_t)draft_toks[d] * embd_row_bytes, H,
            (uint16_t *)(scratchPtr + off_x + (size_t)(d + 1u) * H * 2));
      }
      *(uint32_t *)(scratchPtr + off_pos) = pos;

      VkCommandBuffer active_cb =
          cbs[M]; // cbs[1]=cb_decode, cbs[2..4]=cb_verify[M]
      VkSubmitInfo si = {0};
      si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &active_cb;
      VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
      VK_CHECK(
          vkWaitForFences(device, 1, &fence, VK_TRUE, 0xFFFFFFFFFFFFFFFFULL));
      VK_CHECK(vkResetFences(device, 1, &fence));

      if (M == 1u && !prof_captured) {
        VK_CHECK(vkGetQueryPoolResults(
            device, queryPool, 0, 14, sizeof(prof_ts), prof_ts,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        prof_captured = 1;
        total_gpu_ticks += (prof_ts[13] - prof_ts[0]);
      } else {
        uint64_t ts_start = 0, ts_end = 0;
        VK_CHECK(vkGetQueryPoolResults(
            device, queryPool, 0, 1, sizeof(uint64_t), &ts_start,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        VK_CHECK(vkGetQueryPoolResults(
            device, queryPool, 13, 1, sizeof(uint64_t), &ts_end,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        total_gpu_ticks += (ts_end - ts_start);
      }

      spec_steps++;
      spec_drafted += n_draft;

      uint32_t *preds = (uint32_t *)(scratchPtr + off_out_tok);
      for (uint32_t m = 0; m < M; ++m) {
        uint32_t tok = preds[m];
        if (tok == gf.config.eos_token_id || tok == (uint32_t)tok_im_end) {
          done = 1;
          break;
        }
        print_decoded_token(gf.vocab_tokens[tok]);
        token_history[hist_len++] = tok;
        gen_count++;
        pos++;
        if (gen_count >= max_gen || pos >= MAX_SEQ) {
          done = 1;
          break;
        }
        if (m < n_draft && tok == draft_toks[m]) {
          spec_accepted++;
        } else {
          cur_token = tok;
          break;
        }
      }
    }

    double t_decode_end = get_time_sec();
    double decode_sec = t_decode_end - t_decode_start;
    double gpu_only_sec =
        (double)total_gpu_ticks * (double)props.limits.timestampPeriod * 1e-9;
    puts("\n-------------------------------------------------------------------"
         "--"
         "-----------");
    printf("Wall-Clock Decode: %u tokens in %.3f s (%.2f tokens/sec, %.2f "
           "ms/token)\n",
           gen_count, decode_sec,
           (decode_sec > 0.0) ? ((double)gen_count / decode_sec) : 0.0,
           (gen_count > 0) ? (decode_sec * 1000.0 / (double)gen_count) : 0.0);
    printf("GPU-Only Decode:   %u tokens in %.3f s (%.2f tokens/sec, %.2f "
           "ms/token)\n",
           gen_count, gpu_only_sec,
           (gpu_only_sec > 0.0) ? ((double)gen_count / gpu_only_sec) : 0.0,
           (gen_count > 0) ? (gpu_only_sec * 1000.0 / (double)gen_count) : 0.0);
    printf("Speculative Decode: %u steps | drafted=%u, accepted=%u (%.1f%% "
           "accept, %.2f tok/step)\n",
           spec_steps, spec_drafted, spec_accepted,
           (spec_drafted > 0)
               ? (100.0 * (double)spec_accepted / (double)spec_drafted)
               : 0.0,
           (spec_steps > 0) ? ((double)(gen_count - 1) / (double)spec_steps)
                            : 0.0);
    if (prof_captured) {
      double tp_us = (double)props.limits.timestampPeriod / 1000.0;
      const char *names[13] = {"L0 Attn RMSNorm",
                               "L0 Q/K/V+Bias GEMV",
                               "L0 RoPE",
                               "L0 KV Append",
                               "L0 FlashDecode Attn",
                               "L0 O-proj GEMV",
                               "L0 Res1+FFN RMSNorm",
                               "L0 Fused Gate+Up+SwiGLU",
                               "L0 Down GEMV",
                               "L0 Res2+Next RMSNorm",
                               "Layers 1..23",
                               "LM Head (Q4_0 SLM)",
                               "Argmax (2-Pass)"};
      puts("-------------------------------------------------------------------"
           "--"
           "-----------");
      printf("GPU Timestamp Profile (pos=%u, pre-recorded CB):\n",
             n_prompt + 2);
      for (int i = 0; i < 13; ++i) {
        double us = (double)(prof_ts[i + 1] - prof_ts[i]) * tp_us;
        printf("  %-24s %8.2f us (%6.2f ms)\n", names[i], us, us / 1000.0);
      }
      double l0_ms = (double)(prof_ts[10] - prof_ts[0]) * tp_us / 1000.0;
      double all_layers_ms =
          (double)(prof_ts[11] - prof_ts[0]) * tp_us / 1000.0;
      double head_ms = (double)(prof_ts[13] - prof_ts[11]) * tp_us / 1000.0;
      double total_gpu_ms = (double)(prof_ts[13] - prof_ts[0]) * tp_us / 1000.0;
      printf("  Summary: L0=%.3f ms | 24 Layers=%.3f ms | Head=%.3f ms | Total "
             "GPU=%.3f ms\n",
             l0_ms, all_layers_ms, head_ms, total_gpu_ms);
    }
    puts("====================================================================="
         "=="
         "=========");

    free(lwo);
    free(lds);
    free(cpu_tok_embd);
    gguf_free(&gf);
    exit(0);
    return 0;
  }
