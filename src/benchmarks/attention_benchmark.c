#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

// External CRT & Windows functions (linked via msvcrt.lib / kernel32.lib)
int printf(const char *fmt, ...);
int puts(const char *str);
void *malloc(size_t size);
void free(void *ptr);
void *memcpy(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
void exit(int code);
void *fopen(const char *filename, const char *mode);
int fclose(void *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, void *stream);
int fseek(void *stream, long offset, int whence);
long ftell(void *stream);

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
    if (mant == 0) {
      val = sign;
    } else {
      while ((mant & 0x0400) == 0) {
        mant <<= 1;
        exp--;
      }
      exp++;
      mant &= ~0x0400;
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

// Software math helpers for CPU reference verification (-nostdlib)
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

static uint32_t *readSpirvFile(const char *filename, size_t *outSize) {
  void *fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error: Could not open SPIR-V file: %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, 2);
  long size = ftell(fp);
  fseek(fp, 0, 0);

  uint32_t *buffer = (uint32_t *)malloc(size);
  if (!buffer) {
    printf("Error: Memory allocation failed for SPIR-V code\n");
    exit(1);
  }
  size_t readBytes = fread(buffer, 1, size, fp);
  fclose(fp);
  *outSize = readBytes;
  return buffer;
}

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeBits,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeBits & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if (typeBits & (1 << i)) return i;
  }
  printf("Error: Suitable memory type not found!\n");
  exit(1);
}

static void createBuffer(VkDevice device, VkPhysicalDevice physDev,
                         VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags props, VkBuffer *outBuf,
                         VkDeviceMemory *outMem) {
  VkBufferCreateInfo bufInfo = {0};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, outBuf));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device, *outBuf, &memReq);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex = findMemoryType(physDev, memReq.memoryTypeBits, props);
  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, outMem));
  VK_CHECK(vkBindBufferMemory(device, *outBuf, *outMem, 0));
}

typedef struct {
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t pos;
  float rope_theta;
} RopePushConstants;

typedef struct {
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t seq_len;
  float scale;
} AttnPushConstants;

int mainCRTStartup() {
  puts("================================================================================");
  puts("       Vulkan LLM Decode: RoPE & M=1 GQA KV-Cache Attention Benchmark           ");
  puts("================================================================================");

  // 1. Vulkan Instance & Device
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "AttentionBenchmark";
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
  printf("Timestamp Period:  %.4f ns\n", props.limits.timestampPeriod);

  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
  VkQueueFamilyProperties *qfProps = (VkQueueFamilyProperties *)malloc(sizeof(VkQueueFamilyProperties) * qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);

  uint32_t computeQueueIdx = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qfProps[i].timestampValidBits > 0)) {
      computeQueueIdx = i;
      break;
    }
  }
  free(qfProps);

  VkPhysicalDeviceVulkan12Features features12 = {0};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features12.shaderFloat16 = VK_TRUE;
  features12.shaderSubgroupExtendedTypes = VK_TRUE;

  VkPhysicalDeviceVulkan11Features features11 = {0};
  features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  features11.pNext = &features12;
  features11.storageBuffer16BitAccess = VK_TRUE;

  float qPriority = 1.0f;
  VkDeviceQueueCreateInfo qInfo = {0};
  qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qInfo.queueFamilyIndex = computeQueueIdx;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &qPriority;

  VkDeviceCreateInfo devInfo = {0};
  devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devInfo.pNext = &features11;
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &qInfo;

  VkDevice device;
  VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device));

  VkQueue queue;
  vkGetDeviceQueue(device, computeQueueIdx, 0, &queue);

  // 2. Descriptor Set Layouts (RoPE: 2 bindings, Attention: 4 bindings)
  VkDescriptorSetLayout dslRope, dslAttn;
  {
    VkDescriptorSetLayoutBinding b[4] = {0};
    for (int i = 0; i < 4; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslInfo = {0};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.pBindings = b;

    dslInfo.bindingCount = 2;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslRope));

    dslInfo.bindingCount = 4;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslAttn));
  }

  VkPushConstantRange pcRopeRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RopePushConstants)};
  VkPushConstantRange pcAttnRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AttnPushConstants)};

  VkPipelineLayout plRope, plAttn;
  {
    VkPipelineLayoutCreateInfo plInfo = {0};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pushConstantRangeCount = 1;

    plInfo.pSetLayouts = &dslRope;
    plInfo.pPushConstantRanges = &pcRopeRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plRope));

    plInfo.pSetLayouts = &dslAttn;
    plInfo.pPushConstantRanges = &pcAttnRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plAttn));
  }

  size_t szRope, szAttn;
  uint32_t *spvRope = readSpirvFile("spirv/rope.spv", &szRope);
  uint32_t *spvAttn = readSpirvFile("spirv/attention_decode.spv", &szAttn);

  VkShaderModule smRope, smAttn;
  {
    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = szRope;
    smInfo.pCode = spvRope;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smRope));

    smInfo.codeSize = szAttn;
    smInfo.pCode = spvAttn;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smAttn));
  }
  free(spvRope);
  free(spvAttn);

  VkPipeline pipeRope, pipeAttn;
  {
    VkComputePipelineCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.pName = "main";

    cpInfo.stage.module = smRope;
    cpInfo.layout = plRope;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeRope));

    cpInfo.stage.module = smAttn;
    cpInfo.layout = plAttn;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeAttn));
  }

  // 3. Allocate Buffers (GQA config: 16 Q heads, 4 KV heads, head_dim = 128, max seq_len = 2048)
  const uint32_t NUM_Q_HEADS  = 16;
  const uint32_t NUM_KV_HEADS = 4;
  const uint32_t HEAD_DIM     = 128;
  const uint32_t MAX_SEQ_LEN  = 2048;

  const VkDeviceSize bytesQ       = (VkDeviceSize)NUM_Q_HEADS * HEAD_DIM * sizeof(uint16_t);
  const VkDeviceSize bytesK       = (VkDeviceSize)NUM_KV_HEADS * HEAD_DIM * sizeof(uint16_t);
  const VkDeviceSize bytesKVCache = (VkDeviceSize)MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(uint16_t);

  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkBuffer bufQ, bufK, bufKCache, bufVCache, bufO;
  VkDeviceMemory memQ, memK, memKCache, memVCache, memO;

  createBuffer(device, physDev, bytesQ,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufQ, &memQ);
  createBuffer(device, physDev, bytesK,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufK, &memK);
  createBuffer(device, physDev, bytesKVCache, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufKCache, &memKCache);
  createBuffer(device, physDev, bytesKVCache, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufVCache, &memVCache);
  createBuffer(device, physDev, bytesQ,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufO, &memO);

  VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 2;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descPool));

  VkDescriptorSet descRope, descAttn;
  {
    VkDescriptorSetAllocateInfo allocSet = {0};
    allocSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSet.descriptorPool = descPool;
    allocSet.descriptorSetCount = 1;

    allocSet.pSetLayouts = &dslRope;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descRope));

    allocSet.pSetLayouts = &dslAttn;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descAttn));
  }

  {
    VkDescriptorBufferInfo dbi[2] = {{bufQ, 0, bytesQ}, {bufK, 0, bytesK}};
    VkWriteDescriptorSet w[2] = {0};
    for (int i = 0; i < 2; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descRope;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(device, 2, w, 0, NULL);
  }
  {
    VkDescriptorBufferInfo dbi[4] = {
        {bufQ, 0, bytesQ},
        {bufKCache, 0, bytesKVCache},
        {bufVCache, 0, bytesKVCache},
        {bufO, 0, bytesQ}};
    VkWriteDescriptorSet w[4] = {0};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descAttn;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(device, 4, w, 0, NULL);
  }

  VkQueryPoolCreateInfo qpInfo = {0};
  qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpInfo.queryCount = 2;
  VkQueryPool queryPool;
  VK_CHECK(vkCreateQueryPool(device, &qpInfo, NULL, &queryPool));

  VkCommandPoolCreateInfo cpPoolInfo = {0};
  cpPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpPoolInfo.queueFamilyIndex = computeQueueIdx;
  cpPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cmdPool;
  VK_CHECK(vkCreateCommandPool(device, &cpPoolInfo, NULL, &cmdPool));

  VkCommandBufferAllocateInfo cbAlloc = {0};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = cmdPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAlloc, &cmd));

  puts("--------------------------------------------------------------------------------");

  // =========================================================================
  // TEST 1: RoPE Verification & Benchmark
  // =========================================================================
  {
    uint16_t *pQ, *pK;
    VK_CHECK(vkMapMemory(device, memQ, 0, bytesQ, 0, (void **)&pQ));
    VK_CHECK(vkMapMemory(device, memK, 0, bytesK, 0, (void **)&pK));

    uint32_t qElems = NUM_Q_HEADS * HEAD_DIM;
    uint32_t kElems = NUM_KV_HEADS * HEAD_DIM;
    for (uint32_t i = 0; i < qElems; ++i) pQ[i] = float_to_half(((float)((i % 9) - 4)) * 0.25f);
    for (uint32_t i = 0; i < kElems; ++i) pK[i] = float_to_half(((float)((i % 7) - 3)) * 0.25f);

    const uint32_t testPos = 42;
    const float theta = 10000.0f;
    uint32_t halfDim = HEAD_DIM / 2;

    // CPU Reference for Q head 0
    float cpuQ[128];
    for (uint32_t j = 0; j < halfDim; ++j) {
      float x0 = half_to_float(pQ[j]);
      float x1 = half_to_float(pQ[j + halfDim]);
      float freq = (float)testPos * exp_f(-2.0f * (float)j / (float)HEAD_DIM * log_f(theta));
      float s, c;
      sincos_f(freq, &s, &c);
      cpuQ[j] = x0 * c - x1 * s;
      cpuQ[j + halfDim] = x0 * s + x1 * c;
    }

    vkUnmapMemory(device, memQ);
    vkUnmapMemory(device, memK);

    // Run once to verify in-place correctness
    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRope);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRope, 0, 1, &descRope, 0, NULL);
    RopePushConstants pc = {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, testPos, theta};
    vkCmdPushConstants(cmd, plRope, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t totalPairs = (NUM_Q_HEADS + NUM_KV_HEADS) * halfDim;
    vkCmdDispatch(cmd, (totalPairs + 63) / 64, 1, 1);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo sub = {0};
    sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    VK_CHECK(vkMapMemory(device, memQ, 0, bytesQ, 0, (void **)&pQ));
    int passed = 1;
    float maxDiff = 0.0f;
    for (uint32_t j = 0; j < HEAD_DIM; ++j) {
      float diff = abs_f(half_to_float(pQ[j]) - cpuQ[j]);
      if (diff > maxDiff) maxDiff = diff;
      if (diff > 0.01f) passed = 0;
    }
    vkUnmapMemory(device, memQ);

    // Timing runs
    for (int w = 0; w < 3; ++w) {
      VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
    }
    double bestMs = 1e9;
    uint64_t ts[2] = {0};
    for (int iter = 0; iter < 15; ++iter) {
      VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
      VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
      double ms = ((double)(ts[1] - ts[0]) * (double)props.limits.timestampPeriod) / 1.0e6;
      if (ms < bestMs) bestMs = ms;
    }

    printf("\n[RoPE] Q_heads=%u, KV_heads=%u, head_dim=%u, pos=%u\n", NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, testPos);
    printf("  Best Kernel:   %.4f ms (%.2f us)\n", bestMs, bestMs * 1000.0);
    printf("  Verification:  %s (max diff: %.5f)\n", passed ? "PASSED" : "FAILED", maxDiff);
  }

  // =========================================================================
  // TEST 2: Single-Token (M=1) GQA Online-Softmax Attention
  // =========================================================================
  uint32_t seqLens[] = {128, 512, 2048};
  const float attnScale = 1.0f / sqrt_f((float)HEAD_DIM);

  // Populate Q and max KV cache once
  {
    uint16_t *pQ, *pKc, *pVc;
    VK_CHECK(vkMapMemory(device, memQ, 0, bytesQ, 0, (void **)&pQ));
    VK_CHECK(vkMapMemory(device, memKCache, 0, bytesKVCache, 0, (void **)&pKc));
    VK_CHECK(vkMapMemory(device, memVCache, 0, bytesKVCache, 0, (void **)&pVc));

    for (uint32_t i = 0; i < NUM_Q_HEADS * HEAD_DIM; ++i) {
      pQ[i] = float_to_half(((float)((i % 11) - 5)) * 0.1f);
    }
    uint64_t kvTotal = (uint64_t)MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM;
    for (uint64_t i = 0; i < kvTotal; ++i) {
      pKc[i] = float_to_half(((float)((i % 9) - 4)) * 0.1f);
      pVc[i] = float_to_half(((float)((i % 7) - 3)) * 0.15f);
    }

    vkUnmapMemory(device, memQ);
    vkUnmapMemory(device, memKCache);
    vkUnmapMemory(device, memVCache);
  }

  for (int s = 0; s < 3; ++s) {
    uint32_t seqLen = seqLens[s];

    // CPU reference for Q head 0 (mapped to KV head 0)
    float cpuO[128] = {0};
    {
      uint16_t *pQ, *pKc, *pVc;
      VK_CHECK(vkMapMemory(device, memQ, 0, bytesQ, 0, (void **)&pQ));
      VK_CHECK(vkMapMemory(device, memKCache, 0, (VkDeviceSize)seqLen * NUM_KV_HEADS * HEAD_DIM * 2, 0, (void **)&pKc));
      VK_CHECK(vkMapMemory(device, memVCache, 0, (VkDeviceSize)seqLen * NUM_KV_HEADS * HEAD_DIM * 2, 0, (void **)&pVc));

      float m_prev = -1e30f;
      float l_prev = 0.0f;
      uint32_t kvStride = NUM_KV_HEADS * HEAD_DIM;

      for (uint32_t t = 0; t < seqLen; ++t) {
        uint32_t kvBase = t * kvStride;
        float dot = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
          dot += half_to_float(pQ[d]) * half_to_float(pKc[kvBase + d]);
        }
        float score = dot * attnScale;
        float m_new = (score > m_prev) ? score : m_prev;
        float alpha = exp_f(m_prev - m_new);
        float beta = exp_f(score - m_new);
        l_prev = l_prev * alpha + beta;
        m_prev = m_new;
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
          cpuO[d] = cpuO[d] * alpha + beta * half_to_float(pVc[kvBase + d]);
        }
      }
      for (uint32_t d = 0; d < HEAD_DIM; ++d) cpuO[d] /= l_prev;

      vkUnmapMemory(device, memQ);
      vkUnmapMemory(device, memKCache);
      vkUnmapMemory(device, memVCache);
    }

    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeAttn);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plAttn, 0, 1, &descAttn, 0, NULL);
    AttnPushConstants pc = {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, seqLen, attnScale};
    vkCmdPushConstants(cmd, plAttn, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, NUM_Q_HEADS, 1, 1);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo sub = {0};
    sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &cmd;

    for (int w = 0; w < 3; ++w) {
      VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
    }

    double bestMs = 1e9;
    uint64_t ts[2] = {0};
    for (int iter = 0; iter < 15; ++iter) {
      VK_CHECK(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
      VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
      double ms = ((double)(ts[1] - ts[0]) * (double)props.limits.timestampPeriod) / 1.0e6;
      if (ms < bestMs) bestMs = ms;
    }

    uint16_t *pO;
    VK_CHECK(vkMapMemory(device, memO, 0, bytesQ, 0, (void **)&pO));
    int passed = 1;
    float maxDiff = 0.0f;
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
      float diff = abs_f(half_to_float(pO[d]) - cpuO[d]);
      if (diff > maxDiff) maxDiff = diff;
      if (diff > 0.01f) passed = 0;
    }
    vkUnmapMemory(device, memO);

    // Unique KV cache bytes read = 2 * seqLen * NUM_KV_HEADS * HEAD_DIM * 2 bytes
    double kvBytes = 2.0 * (double)seqLen * (double)NUM_KV_HEADS * (double)HEAD_DIM * 2.0;
    double gbps = (kvBytes / (bestMs / 1000.0)) / 1.0e9;

    printf("\n[M=1 GQA Attention] seq_len=%u, Q=%u, KV=%u, dim=%u (KV=%.2f MB)\n",
           seqLen, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, kvBytes / (1024.0 * 1024.0));
    printf("  Best Kernel:   %.4f ms (%.2f us)\n", bestMs, bestMs * 1000.0);
    printf("  KV Bandwidth:  %.2f GB/s\n", gbps);
    printf("  Verification:  %s (max diff: %.5f)\n", passed ? "PASSED" : "FAILED", maxDiff);
  }

  puts("\n================================================================================");
  puts("Attention & RoPE benchmark completed cleanly!");

  vkDestroyBuffer(device, bufQ, NULL);
  vkFreeMemory(device, memQ, NULL);
  vkDestroyBuffer(device, bufK, NULL);
  vkFreeMemory(device, memK, NULL);
  vkDestroyBuffer(device, bufKCache, NULL);
  vkFreeMemory(device, memKCache, NULL);
  vkDestroyBuffer(device, bufVCache, NULL);
  vkFreeMemory(device, memVCache, NULL);
  vkDestroyBuffer(device, bufO, NULL);
  vkFreeMemory(device, memO, NULL);
  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);
  vkDestroyDescriptorPool(device, descPool, NULL);
  vkDestroyPipeline(device, pipeRope, NULL);
  vkDestroyPipeline(device, pipeAttn, NULL);
  vkDestroyPipelineLayout(device, plRope, NULL);
  vkDestroyPipelineLayout(device, plAttn, NULL);
  vkDestroyShaderModule(device, smRope, NULL);
  vkDestroyShaderModule(device, smAttn, NULL);
  vkDestroyDescriptorSetLayout(device, dslRope, NULL);
  vkDestroyDescriptorSetLayout(device, dslAttn, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);

  exit(0);
}

