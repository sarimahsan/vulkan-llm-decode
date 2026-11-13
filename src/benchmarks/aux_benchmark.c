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

// Software FP32 sqrt & exp for CPU reference verification (-nostdlib)
static inline float sqrt_f(float x) {
  if (x <= 0.0f) return 0.0f;
  float y = x;
  uint32_t i;
  memcpy(&i, &y, 4);
  i = 0x1FBD1DF5 + (i >> 1);
  memcpy(&y, &i, 4);
  for (int k = 0; k < 4; ++k) {
    y = 0.5f * (y + x / y);
  }
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
    if (typeBits & (1 << i)) {
      return i;
    }
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
  allocInfo.memoryTypeIndex =
      findMemoryType(physDev, memReq.memoryTypeBits, props);
  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, outMem));

  VK_CHECK(vkBindBufferMemory(device, *outBuf, *outMem, 0));
}

typedef struct {
  uint32_t K;
  uint32_t K_vec4;
  float eps;
} RmsNormPushConstants;

typedef struct {
  uint32_t N_vec4;
} SwiGluPushConstants;

int mainCRTStartup() {
  puts("================================================================================");
  puts("       Vulkan LLM Decode: RMSNorm & SwiGLU Auxiliary Kernels Benchmark          ");
  puts("================================================================================");

  // 1. Vulkan Instance & Device
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "AuxKernelsBenchmark";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instInfo = {0};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;

  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, NULL);
  if (devCount == 0) {
    puts("Error: No Vulkan physical device found!");
    exit(1);
  }
  VkPhysicalDevice *physDevices =
      (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, physDevices);
  VkPhysicalDevice physDev = physDevices[0];
  free(physDevices);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);
  printf("GPU Device:        %s\n", props.deviceName);
  printf("Timestamp Period:  %.4f ns\n", props.limits.timestampPeriod);

  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
  VkQueueFamilyProperties *qfProps =
      (VkQueueFamilyProperties *)malloc(sizeof(VkQueueFamilyProperties) * qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);

  uint32_t computeQueueIdx = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        (qfProps[i].timestampValidBits > 0)) {
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

  // 2. Descriptor Set Layout (3 bindings for both RMSNorm and SwiGLU)
  VkDescriptorSetLayoutBinding bindings[3] = {0};
  for (int i = 0; i < 3; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslInfo = {0};
  dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslInfo.bindingCount = 3;
  dslInfo.pBindings = bindings;

  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dsl));

  // Pipeline Layouts
  VkPushConstantRange rmsPcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RmsNormPushConstants)};
  VkPushConstantRange swiPcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SwiGluPushConstants)};

  VkPipelineLayout plRms, plSwi;
  {
    VkPipelineLayoutCreateInfo plInfo = {0};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    plInfo.pushConstantRangeCount = 1;

    plInfo.pPushConstantRanges = &rmsPcRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plRms));

    plInfo.pPushConstantRanges = &swiPcRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plSwi));
  }

  // Load SPIR-V
  size_t szRms, szSwi;
  uint32_t *spvRms = readSpirvFile("spirv/rmsnorm.spv", &szRms);
  uint32_t *spvSwi = readSpirvFile("spirv/swiglu.spv", &szSwi);

  VkShaderModule smRms, smSwi;
  {
    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = szRms;
    smInfo.pCode = spvRms;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smRms));

    smInfo.codeSize = szSwi;
    smInfo.pCode = spvSwi;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smSwi));
  }
  free(spvRms);
  free(spvSwi);

  VkPipeline pipeRms, pipeSwi;
  {
    VkComputePipelineCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.pName = "main";

    cpInfo.stage.module = smRms;
    cpInfo.layout = plRms;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeRms));

    cpInfo.stage.module = smSwi;
    cpInfo.layout = plSwi;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeSwi));
  }

  // 3. Allocate Shared Buffers (max 4096 * 1024 = 4M FP16 elements = 8 MB each)
  const uint32_t MAX_ELEMS = 4096 * 1024;
  const VkDeviceSize maxBytes = (VkDeviceSize)MAX_ELEMS * sizeof(uint16_t); // 8 MB

  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkBuffer bufA, bufB, bufC;
  VkDeviceMemory memA, memB, memC;
  createBuffer(device, physDev, maxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufA, &memA);
  createBuffer(device, physDev, maxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufB, &memB);
  createBuffer(device, physDev, maxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufC, &memC);

  VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;

  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descPool));

  VkDescriptorSetAllocateInfo allocSet = {0};
  allocSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocSet.descriptorPool = descPool;
  allocSet.descriptorSetCount = 1;
  allocSet.pSetLayouts = &dsl;

  VkDescriptorSet descSet;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descSet));

  VkDescriptorBufferInfo dbi[3] = {
      {bufA, 0, maxBytes},
      {bufB, 0, maxBytes},
      {bufC, 0, maxBytes}};
  VkWriteDescriptorSet writes[3] = {0};
  for (int i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

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
  // BENCHMARK 1: RMSNorm (Single Token M=1 and Multi-Row Stress)
  // =========================================================================
  uint32_t rmsRows[] = {1, 1024};
  const uint32_t rmsK = 4096;
  const float rmsEps = 1e-5f;

  for (int t = 0; t < 2; ++t) {
    uint32_t rows = rmsRows[t];
    uint16_t *pX, *pG, *pY;
    VK_CHECK(vkMapMemory(device, memA, 0, (VkDeviceSize)rows * rmsK * 2, 0, (void **)&pX));
    VK_CHECK(vkMapMemory(device, memB, 0, (VkDeviceSize)rmsK * 2, 0, (void **)&pG));
    VK_CHECK(vkMapMemory(device, memC, 0, (VkDeviceSize)rows * rmsK * 2, 0, (void **)&pY));

    for (uint32_t j = 0; j < rmsK; ++j) {
      pG[j] = float_to_half(0.5f + ((float)(j % 5)) * 0.25f);
    }
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t j = 0; j < rmsK; ++j) {
        pX[r * rmsK + j] = float_to_half(((float)(((r + j) % 11) - 5)) * 0.3f);
        pY[r * rmsK + j] = 0;
      }
    }

    // CPU reference for row 0
    float sumSq = 0.0f;
    for (uint32_t j = 0; j < rmsK; ++j) {
      float vx = half_to_float(pX[j]);
      sumSq += vx * vx;
    }
    float invRms = 1.0f / sqrt_f(sumSq / (float)rmsK + rmsEps);
    float cpuRef[64];
    for (uint32_t j = 0; j < 64; ++j) {
      cpuRef[j] = half_to_float(pX[j]) * invRms * half_to_float(pG[j]);
    }

    vkUnmapMemory(device, memA);
    vkUnmapMemory(device, memB);
    vkUnmapMemory(device, memC);

    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRms);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRms, 0, 1, &descSet, 0, NULL);
    RmsNormPushConstants pc = {rmsK, rmsK / 4, rmsEps};
    vkCmdPushConstants(cmd, plRms, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, rows, 1, 1);

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

    VK_CHECK(vkMapMemory(device, memC, 0, (VkDeviceSize)rows * rmsK * 2, 0, (void **)&pY));
    int passed = 1;
    float maxDiff = 0.0f;
    for (uint32_t j = 0; j < 64; ++j) {
      float gpuVal = half_to_float(pY[j]);
      float diff = abs_f(gpuVal - cpuRef[j]);
      if (diff > maxDiff) maxDiff = diff;
      if (diff > 0.01f) passed = 0;
    }
    vkUnmapMemory(device, memC);

    double bytes = (double)rows * rmsK * 2.0 * 2.0 + (double)rmsK * 2.0; // Read x + Write y + Read gamma
    double gbps = (bytes / (bestMs / 1000.0)) / 1.0e9;

    printf("\n[RMSNorm] Rows = %u, K = %u (%.2f KB x)\n", rows, rmsK, (double)(rows * rmsK * 2) / 1024.0);
    printf("  Best Kernel:   %.4f ms (%.2f us)\n", bestMs, bestMs * 1000.0);
    if (rows > 1) printf("  Bandwidth:     %.2f GB/s\n", gbps);
    printf("  Verification:  %s (max diff: %.5f)\n", passed ? "PASSED" : "FAILED", maxDiff);
  }

  // =========================================================================
  // BENCHMARK 2: SwiGLU (Single Token M=1 MLP & 8 MB Stress)
  // =========================================================================
  uint32_t swiElems[] = {11008, 4096 * 1024};

  for (int t = 0; t < 2; ++t) {
    uint32_t N = swiElems[t];
    uint16_t *pGate, *pUp, *pOut;
    VK_CHECK(vkMapMemory(device, memA, 0, (VkDeviceSize)N * 2, 0, (void **)&pGate));
    VK_CHECK(vkMapMemory(device, memB, 0, (VkDeviceSize)N * 2, 0, (void **)&pUp));
    VK_CHECK(vkMapMemory(device, memC, 0, (VkDeviceSize)N * 2, 0, (void **)&pOut));

    for (uint32_t i = 0; i < N; ++i) {
      pGate[i] = float_to_half(((float)((i % 13) - 6)) * 0.25f);
      pUp[i]   = float_to_half(((float)((i % 7) - 3)) * 0.4f);
      pOut[i]  = 0;
    }

    float cpuRef[64];
    for (uint32_t i = 0; i < 64; ++i) {
      float g = half_to_float(pGate[i]);
      float u = half_to_float(pUp[i]);
      float silu = g / (1.0f + exp_f(-g));
      cpuRef[i] = silu * u;
    }

    vkUnmapMemory(device, memA);
    vkUnmapMemory(device, memB);
    vkUnmapMemory(device, memC);

    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSwi);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSwi, 0, 1, &descSet, 0, NULL);
    SwiGluPushConstants pc = {N / 4};
    vkCmdPushConstants(cmd, plSwi, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t groupsX = (pc.N_vec4 + 127) / 128;
    vkCmdDispatch(cmd, groupsX, 1, 1);

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

    VK_CHECK(vkMapMemory(device, memC, 0, (VkDeviceSize)N * 2, 0, (void **)&pOut));
    int passed = 1;
    float maxDiff = 0.0f;
    for (uint32_t i = 0; i < 64; ++i) {
      float gpuVal = half_to_float(pOut[i]);
      float diff = abs_f(gpuVal - cpuRef[i]);
      if (diff > maxDiff) maxDiff = diff;
      if (diff > 0.01f) passed = 0;
    }
    vkUnmapMemory(device, memC);

    double bytes = (double)N * 2.0 * 3.0; // Read gate + Read up + Write out
    double gbps = (bytes / (bestMs / 1000.0)) / 1.0e9;

    printf("\n[SwiGLU] Elements = %u (%.2f KB per vector)\n", N, (double)(N * 2) / 1024.0);
    printf("  Best Kernel:   %.4f ms (%.2f us)\n", bestMs, bestMs * 1000.0);
    if (N > 65536) printf("  Bandwidth:     %.2f GB/s\n", gbps);
    printf("  Verification:  %s (max diff: %.5f)\n", passed ? "PASSED" : "FAILED", maxDiff);
  }

  puts("\n================================================================================");
  puts("Auxiliary kernels benchmark completed cleanly!");

  vkDestroyBuffer(device, bufA, NULL);
  vkFreeMemory(device, memA, NULL);
  vkDestroyBuffer(device, bufB, NULL);
  vkFreeMemory(device, memB, NULL);
  vkDestroyBuffer(device, bufC, NULL);
  vkFreeMemory(device, memC, NULL);
  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);
  vkDestroyDescriptorPool(device, descPool, NULL);
  vkDestroyPipeline(device, pipeRms, NULL);
  vkDestroyPipeline(device, pipeSwi, NULL);
  vkDestroyPipelineLayout(device, plRms, NULL);
  vkDestroyPipelineLayout(device, plSwi, NULL);
  vkDestroyShaderModule(device, smRms, NULL);
  vkDestroyShaderModule(device, smSwi, NULL);
  vkDestroyDescriptorSetLayout(device, dsl, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);

  exit(0);
}

