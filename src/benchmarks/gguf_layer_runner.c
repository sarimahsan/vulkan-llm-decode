#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "gguf.h"

int puts(const char *str);
void exit(int code);

int _fltused = 0x9875;

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = (x);                                                        \
    if (err != VK_SUCCESS) {                                                   \
      printf("Vulkan error at line %d: code %d\n", __LINE__, (int)err);        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Software IEEE-754 FP16 <-> FP32 conversion
static inline uint16_t float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x007FFFFF;
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant = (mant | 0x00800000) >> (1 - exp);
    return (uint16_t)(sign | ((mant + 0x00000FFF + ((mant >> 13) & 1)) >> 13));
  } else if (exp == 0xFF - 127 + 15) {
    if (mant == 0) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | 0x7C00 | (mant >> 13));
  } else {
    if (exp > 30) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | ((mant + 0x00000FFF + ((mant >> 13) & 1)) >> 13));
  }
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  int32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x03FF;
  uint32_t val;
  if (exp == 0) {
    if (mant == 0) val = sign;
    else {
      while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
      exp++; mant &= ~0x0400;
      val = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    val = sign | 0x7F800000 | (mant << 13);
  } else {
    val = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &val, 4);
  return f;
}

static inline float abs_f(float x) { return x < 0.0f ? -x : x; }

static inline float sqrt_f(float x) {
  if (x <= 0.0f) return 0.0f;
  float y = x;
  uint32_t i;
  memcpy(&i, &y, 4);
  i = 0x1FBD1DF5 + (i >> 1);
  memcpy(&y, &i, 4);
  for (int k = 0; k < 4; ++k) y = 0.5f * (y + x / y);
  return y;
}

static inline float exp_f(float x) {
  if (x < -80.0f) return 0.0f;
  if (x > 80.0f) return 1e34f;
  const float inv_ln2 = 1.4426950408889634f;
  const float ln2 = 0.6931471805599453f;
  int k = (int)(x * inv_ln2 + (x >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * ln2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.04166667f + r * 0.00833333f))));
  uint32_t bits;
  memcpy(&bits, &p, 4);
  bits += (uint32_t)(k << 23);
  memcpy(&p, &bits, 4);
  return p;
}

static inline float log_f(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  int e = (int)((bits >> 23) & 0xFF) - 127;
  bits = (bits & 0x007FFFFF) | 0x3F800000;
  float m;
  memcpy(&m, &bits, 4);
  float u = (m - 1.0f) / (m + 1.0f);
  float u2 = u * u;
  float ln_m = 2.0f * u * (1.0f + u2 * (0.33333333f + u2 * (0.2f + u2 * 0.14285714f)));
  return ln_m + (float)e * 0.6931471805599453f;
}

static inline void sincos_f(float x, float *out_s, float *out_c) {
  const float two_pi = 6.2831853071795865f;
  const float pi = 3.1415926535897932f;
  const float half_pi = 1.5707963267948966f;
  int k = (int)(x / two_pi + (x >= 0.0f ? 0.5f : -0.5f));
  x -= (float)k * two_pi;
  int sign_c = 1;
  if (x > half_pi) { x = pi - x; sign_c = -1; }
  else if (x < -half_pi) { x = -pi - x; sign_c = -1; }
  float x2 = x * x;
  *out_s = x * (1.0f - x2 * (0.16666667f - x2 * (0.00833333f - x2 * 0.00019841f)));
  float c = 1.0f - x2 * (0.5f - x2 * (0.04166667f - x2 * 0.00138889f));
  *out_c = (float)sign_c * c;
}

// Vulkan helpers
static uint32_t *readSpirvFile(const char *filename, size_t *outSize) {
  void *fp = fopen(filename, "rb");
  if (!fp) { printf("Error opening SPIR-V: %s\n", filename); exit(1); }
  fseek(fp, 0, 2);
  long size = ftell(fp);
  fseek(fp, 0, 0);
  uint32_t *buffer = (uint32_t *)malloc((size_t)size);
  *outSize = fread(buffer, 1, (size_t)size, fp);
  fclose(fp);
  return buffer;
}

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeBits, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) return i;
  }
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if (typeBits & (1 << i)) return i;
  }
  exit(1);
}

typedef struct {
  VkBuffer buf;
  VkDeviceMemory mem;
  VkDeviceSize size;
} GpuBuffer;

static GpuBuffer createGpuBuffer(VkDevice device, VkPhysicalDevice physDev, VkDeviceSize size, const void *initData) {
  GpuBuffer gb;
  gb.size = size;
  VkBufferCreateInfo bufInfo = {0};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, &gb.buf));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device, gb.buf, &memReq);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex = findMemoryType(physDev, memReq.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &gb.mem));
  VK_CHECK(vkBindBufferMemory(device, gb.buf, gb.mem, 0));

  if (initData) {
    void *ptr;
    VK_CHECK(vkMapMemory(device, gb.mem, 0, size, 0, &ptr));
    memcpy(ptr, initData, (size_t)size);
    vkUnmapMemory(device, gb.mem);
  }
  return gb;
}

static void destroyGpuBuffer(VkDevice device, GpuBuffer *gb) {
  vkDestroyBuffer(device, gb->buf, NULL);
  vkFreeMemory(device, gb->mem, NULL);
}

// Push constant structs matching shaders
typedef struct { uint32_t K; uint32_t K_vec4; float eps; } PcRmsNorm;
typedef struct { uint32_t K_blocks; uint32_t N; } PcQ4Gemv;
typedef struct { uint32_t num_q_heads; uint32_t num_kv_heads; uint32_t head_dim; uint32_t pos; float rope_theta; } PcRope;
typedef struct { uint32_t num_q_heads; uint32_t num_kv_heads; uint32_t head_dim; uint32_t seq_len; float scale; } PcAttn;
typedef struct { uint32_t N_vec4; } PcVec4;

#pragma pack(push, 1)
typedef struct { uint16_t d; uint8_t qs[16]; } BlockQ4_0;
#pragma pack(pop)

// CPU Reference Helpers for full 15-step Transformer Layer verification
static void cpu_rmsnorm(const float *x, const uint16_t *gamma, float *out, uint32_t K, float eps) {
  float ss = 0.0f;
  for (uint32_t i = 0; i < K; ++i) ss += x[i] * x[i];
  float inv = 1.0f / sqrt_f(ss / (float)K + eps);
  for (uint32_t i = 0; i < K; ++i) out[i] = x[i] * inv * half_to_float(gamma[i]);
}

static void cpu_q4_gemv(const float *x, const BlockQ4_0 *W, float *out, uint32_t N, uint32_t K) {
  uint32_t blocks_per_row = K / 32;
  for (uint32_t r = 0; r < N; ++r) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
      const BlockQ4_0 *blk = &W[r * blocks_per_row + b];
      float d = half_to_float(blk->d);
      float gsum = 0.0f;
      uint32_t xbase = b * 32;
      for (uint32_t j = 0; j < 16; ++j) {
        int q0 = (int)(blk->qs[j] & 0x0F) - 8;
        int q1 = (int)(blk->qs[j] >> 4) - 8;
        gsum += (float)q0 * x[xbase + j] + (float)q1 * x[xbase + j + 16];
      }
      acc += gsum * d;
    }
    out[r] = acc;
  }
}

static void cpu_rope(float *vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta) {
  uint32_t half = head_dim / 2;
  for (uint32_t h = 0; h < n_heads; ++h) {
    float *v = vec + h * head_dim;
    for (uint32_t j = 0; j < half; ++j) {
      float freq = (float)pos * exp_f(-2.0f * (float)j / (float)head_dim * log_f(theta));
      float s, c;
      sincos_f(freq, &s, &c);
      float x0 = v[j], x1 = v[j + half];
      v[j] = x0 * c - x1 * s;
      v[j + half] = x0 * s + x1 * c;
    }
  }
}

static void cmdBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier mb = {0};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 1, &mb, 0, NULL, 0, NULL);
}

static void updateDescSet3(VkDevice device, VkDescriptorSet ds, VkBuffer b0, VkDeviceSize s0,
                           VkBuffer b1, VkDeviceSize s1, VkBuffer b2, VkDeviceSize s2) {
  VkDescriptorBufferInfo dbi[3] = {{b0, 0, s0}, {b1, 0, s1}, {b2, 0, s2}};
  VkWriteDescriptorSet w[3] = {0};
  for (int i = 0; i < 3; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = ds;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(device, 3, w, 0, NULL);
}

int mainCRTStartup() {
  puts("================================================================================");
  puts("   Vulkan LLM Decode: GGUF v3 Zero-Copy Full Transformer Decoder Layer Runner   ");
  puts("================================================================================");

  // 1. Load GGUF v3 file from disk
  const char *ggufPath = "bin/test_layer.gguf";
  GgufFile gf;
  if (!gguf_load_file(ggufPath, &gf)) exit(1);

  uint32_t H = gf.config.hidden_size;
  uint32_t I = gf.config.intermediate_size;
  uint32_t Q_HEADS = gf.config.num_heads;
  uint32_t KV_HEADS = gf.config.num_kv_heads;
  uint32_t HEAD_DIM = gf.config.head_dim;
  uint32_t KV_DIM = KV_HEADS * HEAD_DIM;

  printf("Loaded GGUF v%u:    %s (%.2f MB)\n", gf.version, ggufPath, (double)gf.file_size / (1024.0 * 1024.0));
  printf("Model Config:      hidden=%u, inter=%u, q_heads=%u, kv_heads=%u, head_dim=%u\n",
         H, I, Q_HEADS, KV_HEADS, HEAD_DIM);
  printf("Tensors Parsed:    %llu\n", (unsigned long long)gf.tensor_count);

  const GgufTensorInfo *t_attn_norm = gguf_find_tensor(&gf, "blk.0.attn_norm.weight");
  const GgufTensorInfo *t_wq        = gguf_find_tensor(&gf, "blk.0.attn_q.weight");
  const GgufTensorInfo *t_wk        = gguf_find_tensor(&gf, "blk.0.attn_k.weight");
  const GgufTensorInfo *t_wv        = gguf_find_tensor(&gf, "blk.0.attn_v.weight");
  const GgufTensorInfo *t_wo        = gguf_find_tensor(&gf, "blk.0.attn_output.weight");
  const GgufTensorInfo *t_ffn_norm  = gguf_find_tensor(&gf, "blk.0.ffn_norm.weight");
  const GgufTensorInfo *t_wgate     = gguf_find_tensor(&gf, "blk.0.ffn_gate.weight");
  const GgufTensorInfo *t_wup       = gguf_find_tensor(&gf, "blk.0.ffn_up.weight");
  const GgufTensorInfo *t_wdown     = gguf_find_tensor(&gf, "blk.0.ffn_down.weight");

  if (!t_attn_norm || !t_wq || !t_wk || !t_wv || !t_wo || !t_ffn_norm || !t_wgate || !t_wup || !t_wdown) {
    puts("Error: Missing required blk.0.* tensors in GGUF file!");
    exit(1);
  }

  // 2. Initialize Vulkan
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "GgufLayerRunner";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instInfo = {0};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;
  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, NULL);
  VkPhysicalDevice *physDevices = (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, physDevices);
  VkPhysicalDevice physDev = physDevices[0];
  free(physDevices);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);
  printf("GPU Device:        %s\n", props.deviceName);

  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
  VkQueueFamilyProperties *qfProps = (VkQueueFamilyProperties *)malloc(sizeof(VkQueueFamilyProperties) * qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);
  uint32_t qIdx = 0;
  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qfProps[i].timestampValidBits > 0)) { qIdx = i; break; }
  }
  free(qfProps);

  VkPhysicalDeviceVulkan12Features f12 = {0};
  f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  f12.shaderFloat16 = VK_TRUE;
  f12.shaderSubgroupExtendedTypes = VK_TRUE;
  f12.storageBuffer8BitAccess = VK_TRUE;

  VkPhysicalDeviceVulkan11Features f11 = {0};
  f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  f11.pNext = &f12;
  f11.storageBuffer16BitAccess = VK_TRUE;

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qInfo = {0};
  qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qInfo.queueFamilyIndex = qIdx;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &prio;

  VkDeviceCreateInfo devInfo = {0};
  devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devInfo.pNext = &f11;
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &qInfo;

  VkDevice device;
  VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, qIdx, 0, &queue);

  // 3. Create Pipelines (RMSNorm, Q4_0 GEMV, RoPE, Attention, SwiGLU, ResidualAdd)
  VkDescriptorSetLayout dsl2, dsl3, dsl4;
  {
    VkDescriptorSetLayoutBinding b[4] = {0};
    for (int i = 0; i < 4; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pBindings = b;
    info.bindingCount = 2; VK_CHECK(vkCreateDescriptorSetLayout(device, &info, NULL, &dsl2));
    info.bindingCount = 3; VK_CHECK(vkCreateDescriptorSetLayout(device, &info, NULL, &dsl3));
    info.bindingCount = 4; VK_CHECK(vkCreateDescriptorSetLayout(device, &info, NULL, &dsl4));
  }

  const char *spvPaths[6] = {
      "spirv/rmsnorm.spv", "spirv/gemv_q4_0.spv", "spirv/rope.spv",
      "spirv/attention_decode.spv", "spirv/swiglu.spv", "spirv/residual_add.spv"};
  uint32_t pcSizes[6] = {sizeof(PcRmsNorm), sizeof(PcQ4Gemv), sizeof(PcRope),
                         sizeof(PcAttn), sizeof(PcVec4), sizeof(PcVec4)};
  VkDescriptorSetLayout dsls[6] = {dsl3, dsl3, dsl2, dsl4, dsl3, dsl3};

  VkShaderModule sms[6];
  VkPipelineLayout pls[6];
  VkPipeline pipes[6];
  for (int i = 0; i < 6; ++i) {
    size_t sz;
    uint32_t *code = readSpirvFile(spvPaths[i], &sz);
    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sz; smi.pCode = code;
    VK_CHECK(vkCreateShaderModule(device, &smi, NULL, &sms[i]));
    free(code);

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSizes[i]};
    VkPipelineLayoutCreateInfo pli = {0};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsls[i];
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, NULL, &pls[i]));

    VkComputePipelineCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sms[i];
    cpi.stage.pName = "main";
    cpi.layout = pls[i];
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipes[i]));
  }

  // 4. Upload GGUF Weights directly to Vulkan Storage Buffers (zero conversion!)
  GpuBuffer b_attn_norm = createGpuBuffer(device, physDev, H * 2, gf.tensor_data + t_attn_norm->offset);
  GpuBuffer b_wq        = createGpuBuffer(device, physDev, (H * H / 32) * 18, gf.tensor_data + t_wq->offset);
  GpuBuffer b_wk        = createGpuBuffer(device, physDev, (KV_DIM * H / 32) * 18, gf.tensor_data + t_wk->offset);
  GpuBuffer b_wv        = createGpuBuffer(device, physDev, (KV_DIM * H / 32) * 18, gf.tensor_data + t_wv->offset);
  GpuBuffer b_wo        = createGpuBuffer(device, physDev, (H * H / 32) * 18, gf.tensor_data + t_wo->offset);
  GpuBuffer b_ffn_norm  = createGpuBuffer(device, physDev, H * 2, gf.tensor_data + t_ffn_norm->offset);
  GpuBuffer b_wgate     = createGpuBuffer(device, physDev, (I * H / 32) * 18, gf.tensor_data + t_wgate->offset);
  GpuBuffer b_wup       = createGpuBuffer(device, physDev, (I * H / 32) * 18, gf.tensor_data + t_wup->offset);
  GpuBuffer b_wdown     = createGpuBuffer(device, physDev, (H * I / 32) * 18, gf.tensor_data + t_wdown->offset);

  // Activation & KV Cache Buffers
  const uint32_t MAX_SEQ = 256;
  const uint32_t TEST_POS = 15; // 16 tokens in KV cache (0..15)
  GpuBuffer b_x        = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_xnorm    = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_q        = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_k        = createGpuBuffer(device, physDev, KV_DIM * 2, NULL);
  GpuBuffer b_v        = createGpuBuffer(device, physDev, KV_DIM * 2, NULL);
  GpuBuffer b_kcache   = createGpuBuffer(device, physDev, MAX_SEQ * KV_DIM * 2, NULL);
  GpuBuffer b_vcache   = createGpuBuffer(device, physDev, MAX_SEQ * KV_DIM * 2, NULL);
  GpuBuffer b_attn_out = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_oproj    = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_h        = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_gate     = createGpuBuffer(device, physDev, I * 2, NULL);
  GpuBuffer b_up       = createGpuBuffer(device, physDev, I * 2, NULL);
  GpuBuffer b_mlp_act  = createGpuBuffer(device, physDev, I * 2, NULL);
  GpuBuffer b_mlp_out  = createGpuBuffer(device, physDev, H * 2, NULL);
  GpuBuffer b_out      = createGpuBuffer(device, physDev, H * 2, NULL);

  // Initialize input x and past KV cache [0 .. TEST_POS-1]
  float *cpu_x = (float *)malloc(sizeof(float) * H);
  {
    uint16_t *px;
    VK_CHECK(vkMapMemory(device, b_x.mem, 0, H * 2, 0, (void **)&px));
    for (uint32_t i = 0; i < H; ++i) {
      px[i] = float_to_half(((float)((i % 9) - 4)) * 0.25f);
      cpu_x[i] = half_to_float(px[i]);
    }
    vkUnmapMemory(device, b_x.mem);

    uint16_t *pkc, *pvc;
    VK_CHECK(vkMapMemory(device, b_kcache.mem, 0, TEST_POS * KV_DIM * 2, 0, (void **)&pkc));
    VK_CHECK(vkMapMemory(device, b_vcache.mem, 0, TEST_POS * KV_DIM * 2, 0, (void **)&pvc));
    for (uint32_t i = 0; i < TEST_POS * KV_DIM; ++i) {
      pkc[i] = float_to_half(((float)((i % 7) - 3)) * 0.1f);
      pvc[i] = float_to_half(((float)((i % 5) - 2)) * 0.15f);
    }
    vkUnmapMemory(device, b_kcache.mem);
    vkUnmapMemory(device, b_vcache.mem);
  }

  // 5. Allocate Descriptor Sets (13 total operations)
  VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 48};
  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 16;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descPool));

  VkDescriptorSet ds_rms1, ds_wq, ds_wk, ds_wv, ds_rope, ds_attn, ds_wo, ds_res1, ds_rms2, ds_wgate, ds_wup, ds_swiglu, ds_wdown, ds_res2;
  VkDescriptorSetLayout allocLayouts[14] = {dsl3, dsl3, dsl3, dsl3, dsl2, dsl4, dsl3, dsl3, dsl3, dsl3, dsl3, dsl3, dsl3, dsl3};
  VkDescriptorSet *dsPtrs[14] = {&ds_rms1, &ds_wq, &ds_wk, &ds_wv, &ds_rope, &ds_attn, &ds_wo, &ds_res1, &ds_rms2, &ds_wgate, &ds_wup, &ds_swiglu, &ds_wdown, &ds_res2};
  for (int i = 0; i < 14; ++i) {
    VkDescriptorSetAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &allocLayouts[i];
    VK_CHECK(vkAllocateDescriptorSets(device, &ai, dsPtrs[i]));
  }

  updateDescSet3(device, ds_rms1,   b_x.buf, H*2,        b_attn_norm.buf, H*2,      b_xnorm.buf, H*2);
  updateDescSet3(device, ds_wq,     b_xnorm.buf, H*2,    b_wq.buf, b_wq.size,       b_q.buf, H*2);
  updateDescSet3(device, ds_wk,     b_xnorm.buf, H*2,    b_wk.buf, b_wk.size,       b_k.buf, KV_DIM*2);
  updateDescSet3(device, ds_wv,     b_xnorm.buf, H*2,    b_wv.buf, b_wv.size,       b_v.buf, KV_DIM*2);
  updateDescSet3(device, ds_wo,     b_attn_out.buf, H*2, b_wo.buf, b_wo.size,       b_oproj.buf, H*2);
  updateDescSet3(device, ds_res1,   b_x.buf, H*2,        b_oproj.buf, H*2,          b_h.buf, H*2);
  updateDescSet3(device, ds_rms2,   b_h.buf, H*2,        b_ffn_norm.buf, H*2,       b_xnorm.buf, H*2);
  updateDescSet3(device, ds_wgate,  b_xnorm.buf, H*2,    b_wgate.buf, b_wgate.size, b_gate.buf, I*2);
  updateDescSet3(device, ds_wup,    b_xnorm.buf, H*2,    b_wup.buf, b_wup.size,     b_up.buf, I*2);
  updateDescSet3(device, ds_swiglu, b_gate.buf, I*2,     b_up.buf, I*2,             b_mlp_act.buf, I*2);
  updateDescSet3(device, ds_wdown,  b_mlp_act.buf, I*2,  b_wdown.buf, b_wdown.size, b_mlp_out.buf, H*2);
  updateDescSet3(device, ds_res2,   b_h.buf, H*2,        b_mlp_out.buf, H*2,        b_out.buf, H*2);

  // RoPE (2 bindings)
  {
    VkDescriptorBufferInfo dbi[2] = {{b_q.buf, 0, H*2}, {b_k.buf, 0, KV_DIM*2}};
    VkWriteDescriptorSet w[2] = {0};
    for (int i = 0; i < 2; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds_rope;
      w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(device, 2, w, 0, NULL);
  }
  // Attention (4 bindings)
  {
    VkDescriptorBufferInfo dbi[4] = {{b_q.buf, 0, H*2}, {b_kcache.buf, 0, b_kcache.size}, {b_vcache.buf, 0, b_vcache.size}, {b_attn_out.buf, 0, H*2}};
    VkWriteDescriptorSet w[4] = {0};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds_attn;
      w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(device, 4, w, 0, NULL);
  }

  // 6. Record Entire Transformer Decoder Layer into ONE Command Buffer
  VkQueryPoolCreateInfo qpInfo = {0};
  qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP; qpInfo.queryCount = 2;
  VkQueryPool queryPool;
  VK_CHECK(vkCreateQueryPool(device, &qpInfo, NULL, &queryPool));

  VkCommandPoolCreateInfo cpInfo = {0};
  cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpInfo.queueFamilyIndex = qIdx; cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cmdPool;
  VK_CHECK(vkCreateCommandPool(device, &cpInfo, NULL, &cmdPool));

  VkCommandBufferAllocateInfo cbAlloc = {0};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = cmdPool; cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbAlloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAlloc, &cmd));

  VkCommandBufferBeginInfo cbBegin = {0};
  cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
  vkCmdResetQueryPool(cmd, queryPool, 0, 2);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

  // 1. Attention RMSNorm
  PcRmsNorm pcRms = {H, H / 4, gf.config.rms_norm_eps};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[0], 0, 1, &ds_rms1, 0, NULL);
  vkCmdPushConstants(cmd, pls[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcRms), &pcRms);
  vkCmdDispatch(cmd, 1, 1, 1);
  cmdBarrier(cmd);

  // 2-4. Q, K, V Projections (parallelizable, barrier after)
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1]);
  PcQ4Gemv pcQ = {H / 32, H};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wq, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcQ), &pcQ);
  vkCmdDispatch(cmd, (H + 3) / 4, 1, 1);

  PcQ4Gemv pcKV = {H / 32, KV_DIM};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wk, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcKV), &pcKV);
  vkCmdDispatch(cmd, (KV_DIM + 3) / 4, 1, 1);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wv, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcKV), &pcKV);
  vkCmdDispatch(cmd, (KV_DIM + 3) / 4, 1, 1);
  cmdBarrier(cmd);

  // 5. RoPE on Q and K
  PcRope pcRope = {Q_HEADS, KV_HEADS, HEAD_DIM, TEST_POS, gf.config.rope_theta};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[2]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[2], 0, 1, &ds_rope, 0, NULL);
  vkCmdPushConstants(cmd, pls[2], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcRope), &pcRope);
  uint32_t totalPairs = (Q_HEADS + KV_HEADS) * (HEAD_DIM / 2);
  vkCmdDispatch(cmd, (totalPairs + 63) / 64, 1, 1);
  cmdBarrier(cmd);

  // 6. Append K, V to KV-Cache[TEST_POS]
  VkBufferCopy kvCopy = {0, (VkDeviceSize)TEST_POS * KV_DIM * 2, (VkDeviceSize)KV_DIM * 2};
  vkCmdCopyBuffer(cmd, b_k.buf, b_kcache.buf, 1, &kvCopy);
  vkCmdCopyBuffer(cmd, b_v.buf, b_vcache.buf, 1, &kvCopy);
  cmdBarrier(cmd);

  // 7. FlashDecode GQA Attention
  PcAttn pcAttn = {Q_HEADS, KV_HEADS, HEAD_DIM, TEST_POS + 1, 1.0f / sqrt_f((float)HEAD_DIM)};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[3]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[3], 0, 1, &ds_attn, 0, NULL);
  vkCmdPushConstants(cmd, pls[3], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcAttn), &pcAttn);
  vkCmdDispatch(cmd, Q_HEADS, 1, 1);
  cmdBarrier(cmd);

  // 8. Output Projection (W_o)
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wo, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcQ), &pcQ);
  vkCmdDispatch(cmd, (H + 3) / 4, 1, 1);
  cmdBarrier(cmd);

  // 9. Residual Add 1 (h = x + o_proj)
  PcVec4 pcH = {H / 4};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[5]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[5], 0, 1, &ds_res1, 0, NULL);
  vkCmdPushConstants(cmd, pls[5], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcH), &pcH);
  vkCmdDispatch(cmd, (pcH.N_vec4 + 127) / 128, 1, 1);
  cmdBarrier(cmd);

  // 10. FFN RMSNorm
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[0], 0, 1, &ds_rms2, 0, NULL);
  vkCmdPushConstants(cmd, pls[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcRms), &pcRms);
  vkCmdDispatch(cmd, 1, 1, 1);
  cmdBarrier(cmd);

  // 11-12. Gate & Up Projections
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1]);
  PcQ4Gemv pcGate = {H / 32, I};
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wgate, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcGate), &pcGate);
  vkCmdDispatch(cmd, (I + 3) / 4, 1, 1);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wup, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcGate), &pcGate);
  vkCmdDispatch(cmd, (I + 3) / 4, 1, 1);
  cmdBarrier(cmd);

  // 13. SwiGLU
  PcVec4 pcI = {I / 4};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[4]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[4], 0, 1, &ds_swiglu, 0, NULL);
  vkCmdPushConstants(cmd, pls[4], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcI), &pcI);
  vkCmdDispatch(cmd, (pcI.N_vec4 + 127) / 128, 1, 1);
  cmdBarrier(cmd);

  // 14. Down Projection (W_down)
  PcQ4Gemv pcDown = {I / 32, H};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[1], 0, 1, &ds_wdown, 0, NULL);
  vkCmdPushConstants(cmd, pls[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcDown), &pcDown);
  vkCmdDispatch(cmd, (H + 3) / 4, 1, 1);
  cmdBarrier(cmd);

  // 15. Residual Add 2 (out = h + mlp_out)
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[5]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pls[5], 0, 1, &ds_res2, 0, NULL);
  vkCmdPushConstants(cmd, pls[5], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcH), &pcH);
  vkCmdDispatch(cmd, (pcH.N_vec4 + 127) / 128, 1, 1);

  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  // Execute Warmup & Benchmark
  VkSubmitInfo sub = {0};
  sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  sub.commandBufferCount = 1; sub.pCommandBuffers = &cmd;

  for (int w = 0; w < 3; ++w) {
    VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
  }

  double bestMs = 1e9, totalMs = 0.0;
  uint64_t ts[2] = {0};
  const int iters = 15;
  for (int it = 0; it < iters; ++it) {
    VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
    double ms = ((double)(ts[1] - ts[0]) * (double)props.limits.timestampPeriod) / 1.0e6;
    if (ms < bestMs) bestMs = ms;
    totalMs += ms;
  }

  // 7. Full CPU FP32 Reference Verification of all 15 steps
  float *cpu_norm = (float *)malloc(sizeof(float) * H);
  float *cpu_q    = (float *)malloc(sizeof(float) * H);
  float *cpu_k    = (float *)malloc(sizeof(float) * KV_DIM);
  float *cpu_v    = (float *)malloc(sizeof(float) * KV_DIM);
  float *cpu_attn = (float *)malloc(sizeof(float) * H);
  float *cpu_op   = (float *)malloc(sizeof(float) * H);
  float *cpu_h    = (float *)malloc(sizeof(float) * H);
  float *cpu_gate = (float *)malloc(sizeof(float) * I);
  float *cpu_up   = (float *)malloc(sizeof(float) * I);
  float *cpu_act  = (float *)malloc(sizeof(float) * I);
  float *cpu_mlp  = (float *)malloc(sizeof(float) * H);
  float *cpu_out  = (float *)malloc(sizeof(float) * H);

  cpu_rmsnorm(cpu_x, (const uint16_t *)(gf.tensor_data + t_attn_norm->offset), cpu_norm, H, gf.config.rms_norm_eps);
  cpu_q4_gemv(cpu_norm, (const BlockQ4_0 *)(gf.tensor_data + t_wq->offset), cpu_q, H, H);
  cpu_q4_gemv(cpu_norm, (const BlockQ4_0 *)(gf.tensor_data + t_wk->offset), cpu_k, KV_DIM, H);
  cpu_q4_gemv(cpu_norm, (const BlockQ4_0 *)(gf.tensor_data + t_wv->offset), cpu_v, KV_DIM, H);
  cpu_rope(cpu_q, Q_HEADS, HEAD_DIM, TEST_POS, gf.config.rope_theta);
  cpu_rope(cpu_k, KV_HEADS, HEAD_DIM, TEST_POS, gf.config.rope_theta);

  // CPU Attention
  {
    uint16_t *pkc, *pvc;
    VK_CHECK(vkMapMemory(device, b_kcache.mem, 0, (TEST_POS + 1) * KV_DIM * 2, 0, (void **)&pkc));
    VK_CHECK(vkMapMemory(device, b_vcache.mem, 0, (TEST_POS + 1) * KV_DIM * 2, 0, (void **)&pvc));
    float scale = 1.0f / sqrt_f((float)HEAD_DIM);
    uint32_t gqa = Q_HEADS / KV_HEADS;
    for (uint32_t qh = 0; qh < Q_HEADS; ++qh) {
      uint32_t kvh = qh / gqa;
      float m_prev = -1e30f, l_prev = 0.0f;
      float o_acc[128] = {0};
      for (uint32_t t = 0; t <= TEST_POS; ++t) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
          float kv = (t == TEST_POS) ? cpu_k[kvh * HEAD_DIM + d] : half_to_float(pkc[t * KV_DIM + kvh * HEAD_DIM + d]);
          dot += cpu_q[qh * HEAD_DIM + d] * kv;
        }
        float sc = dot * scale;
        float m_new = (sc > m_prev) ? sc : m_prev;
        float a = exp_f(m_prev - m_new), b = exp_f(sc - m_new);
        l_prev = l_prev * a + b; m_prev = m_new;
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
          float vv = (t == TEST_POS) ? cpu_v[kvh * HEAD_DIM + d] : half_to_float(pvc[t * KV_DIM + kvh * HEAD_DIM + d]);
          o_acc[d] = o_acc[d] * a + b * vv;
        }
      }
      for (uint32_t d = 0; d < HEAD_DIM; ++d) cpu_attn[qh * HEAD_DIM + d] = o_acc[d] / l_prev;
    }
    vkUnmapMemory(device, b_kcache.mem);
    vkUnmapMemory(device, b_vcache.mem);
  }

  cpu_q4_gemv(cpu_attn, (const BlockQ4_0 *)(gf.tensor_data + t_wo->offset), cpu_op, H, H);
  for (uint32_t i = 0; i < H; ++i) cpu_h[i] = cpu_x[i] + cpu_op[i];
  cpu_rmsnorm(cpu_h, (const uint16_t *)(gf.tensor_data + t_ffn_norm->offset), cpu_norm, H, gf.config.rms_norm_eps);
  cpu_q4_gemv(cpu_norm, (const BlockQ4_0 *)(gf.tensor_data + t_wgate->offset), cpu_gate, I, H);
  cpu_q4_gemv(cpu_norm, (const BlockQ4_0 *)(gf.tensor_data + t_wup->offset), cpu_up, I, H);
  for (uint32_t i = 0; i < I; ++i) {
    float g = cpu_gate[i];
    cpu_act[i] = (g / (1.0f + exp_f(-g))) * cpu_up[i];
  }
  cpu_q4_gemv(cpu_act, (const BlockQ4_0 *)(gf.tensor_data + t_wdown->offset), cpu_mlp, H, I);
  for (uint32_t i = 0; i < H; ++i) cpu_out[i] = cpu_h[i] + cpu_mlp[i];

  // Compare GPU final output against CPU
  uint16_t *pOut;
  VK_CHECK(vkMapMemory(device, b_out.mem, 0, H * 2, 0, (void **)&pOut));
  int passed = 1;
  float maxDiff = 0.0f;
  for (uint32_t i = 0; i < H; ++i) {
    float g = half_to_float(pOut[i]);
    float diff = abs_f(g - cpu_out[i]);
    if (diff > maxDiff) maxDiff = diff;
    float rel = diff / (abs_f(cpu_out[i]) + 1e-3f);
    if (rel > 0.05f && diff > 0.05f) passed = 0;
  }
  vkUnmapMemory(device, b_out.mem);

  puts("--------------------------------------------------------------------------------");
  printf("  End-to-End 15-Step Layer Verification: %s (max diff: %.5f over %u dims)\n",
         passed ? "PASSED" : "FAILED", maxDiff, H);
  printf("  Single Layer Latency (Best / Avg):     %.3f ms / %.3f ms\n", bestMs, totalMs / iters);
  printf("  Extrapolated 24-Layer 0.5B Decode:     %.1f tokens/sec\n", 1000.0 / (bestMs * 24.0));
  puts("================================================================================");

  // Cleanup
  free(cpu_x); free(cpu_norm); free(cpu_q); free(cpu_k); free(cpu_v);
  free(cpu_attn); free(cpu_op); free(cpu_h); free(cpu_gate); free(cpu_up);
  free(cpu_act); free(cpu_mlp); free(cpu_out);
  gguf_free(&gf);

  destroyGpuBuffer(device, &b_attn_norm); destroyGpuBuffer(device, &b_wq);
  destroyGpuBuffer(device, &b_wk); destroyGpuBuffer(device, &b_wv);
  destroyGpuBuffer(device, &b_wo); destroyGpuBuffer(device, &b_ffn_norm);
  destroyGpuBuffer(device, &b_wgate); destroyGpuBuffer(device, &b_wup);
  destroyGpuBuffer(device, &b_wdown); destroyGpuBuffer(device, &b_x);
  destroyGpuBuffer(device, &b_xnorm); destroyGpuBuffer(device, &b_q);
  destroyGpuBuffer(device, &b_k); destroyGpuBuffer(device, &b_v);
  destroyGpuBuffer(device, &b_kcache); destroyGpuBuffer(device, &b_vcache);
  destroyGpuBuffer(device, &b_attn_out); destroyGpuBuffer(device, &b_oproj);
  destroyGpuBuffer(device, &b_h); destroyGpuBuffer(device, &b_gate);
  destroyGpuBuffer(device, &b_up); destroyGpuBuffer(device, &b_mlp_act);
  destroyGpuBuffer(device, &b_mlp_out); destroyGpuBuffer(device, &b_out);

  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);
  vkDestroyDescriptorPool(device, descPool, NULL);
  for (int i = 0; i < 6; ++i) {
    vkDestroyPipeline(device, pipes[i], NULL);
    vkDestroyPipelineLayout(device, pls[i], NULL);
    vkDestroyShaderModule(device, sms[i], NULL);
  }
  vkDestroyDescriptorSetLayout(device, dsl2, NULL);
  vkDestroyDescriptorSetLayout(device, dsl3, NULL);
  vkDestroyDescriptorSetLayout(device, dsl4, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);
  exit(0);
}

