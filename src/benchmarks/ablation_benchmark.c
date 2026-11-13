#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

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
    printf("Error: Memory allocation failed\n");
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
  uint32_t K_param;
  uint32_t N;
} PushConstants;

static VkPipeline createComputePipeline(VkDevice device, const char *spvPath,
                                        VkPipelineLayout layout) {
  size_t sz = 0;
  uint32_t *code = readSpirvFile(spvPath, &sz);
  VkShaderModuleCreateInfo smInfo = {0};
  smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smInfo.codeSize = sz;
  smInfo.pCode = code;
  VkShaderModule mod;
  VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &mod));
  free(code);

  VkComputePipelineCreateInfo pipeInfo = {0};
  pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = mod;
  pipeInfo.stage.pName = "main";
  pipeInfo.layout = layout;

  VkPipeline pipe;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, NULL, &pipe));
  vkDestroyShaderModule(device, mod, NULL);
  return pipe;
}

static void updateSet3(VkDevice device, VkDescriptorSet set,
                       VkBuffer b0, VkDeviceSize s0,
                       VkBuffer b1, VkDeviceSize s1,
                       VkBuffer b2, VkDeviceSize s2) {
  VkDescriptorBufferInfo bi[3] = {
    {b0, 0, s0}, {b1, 0, s1}, {b2, 0, s2}
  };
  VkWriteDescriptorSet w[3] = {0};
  for (uint32_t i = 0; i < 3; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(device, 3, w, 0, NULL);
}

static void updateSet4(VkDevice device, VkDescriptorSet set,
                       VkBuffer b0, VkDeviceSize s0,
                       VkBuffer b1, VkDeviceSize s1,
                       VkBuffer b2, VkDeviceSize s2,
                       VkBuffer b3, VkDeviceSize s3) {
  VkDescriptorBufferInfo bi[4] = {
    {b0, 0, s0}, {b1, 0, s1}, {b2, 0, s2}, {b3, 0, s3}
  };
  VkWriteDescriptorSet w[4] = {0};
  for (uint32_t i = 0; i < 4; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(device, 4, w, 0, NULL);
}

static double benchSinglePipeline(VkDevice device, VkQueue queue,
                                  VkCommandBuffer cmd, VkQueryPool qp,
                                  double tsPeriodNs, VkPipeline pipe,
                                  VkPipelineLayout layout, VkDescriptorSet set,
                                  uint32_t K_param, uint32_t N,
                                  int warmup, int iters) {
  uint32_t groupsX = (N + 3) / 4;
  PushConstants pc = {K_param, N};
  double sumMs = 0.0;

  for (int it = -warmup; it < iters; ++it) {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdResetQueryPool(cmd, qp, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groupsX, 1, 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    if (it >= 0) {
      uint64_t ts[2] = {0};
      VK_CHECK(vkGetQueryPoolResults(device, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      sumMs += (double)(ts[1] - ts[0]) * tsPeriodNs / 1e6;
    }
  }
  return sumMs / (double)iters;
}

static double benchUnfusedInt4(VkDevice device, VkQueue queue,
                               VkCommandBuffer cmd, VkQueryPool qp,
                               double tsPeriodNs,
                               VkPipeline pipeDequant, VkPipeline pipeGemvFp16,
                               VkPipelineLayout layout3,
                               VkDescriptorSet setDequant, VkDescriptorSet setGemvFp16,
                               uint32_t K, uint32_t N, int warmup, int iters) {
  uint32_t groupsX = (N + 3) / 4;
  PushConstants pcDequant = {K / 32, N};
  PushConstants pcFp16 = {K / 4, N};
  double sumMs = 0.0;

  for (int it = -warmup; it < iters; ++it) {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdResetQueryPool(cmd, qp, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);

    // Pass 1: Unpack INT4 -> FP16 in DRAM
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeDequant);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout3, 0, 1, &setDequant, 0, NULL);
    vkCmdPushConstants(cmd, layout3, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcDequant), &pcDequant);
    vkCmdDispatch(cmd, groupsX, 1, 1);

    VkMemoryBarrier bar = {0};
    bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar, 0, NULL, 0, NULL);

    // Pass 2: Read FP16 from DRAM & run FP16 GEMV
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeGemvFp16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout3, 0, 1, &setGemvFp16, 0, NULL);
    vkCmdPushConstants(cmd, layout3, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcFp16), &pcFp16);
    vkCmdDispatch(cmd, groupsX, 1, 1);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    if (it >= 0) {
      uint64_t ts[2] = {0};
      VK_CHECK(vkGetQueryPoolResults(device, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      sumMs += (double)(ts[1] - ts[0]) * tsPeriodNs / 1e6;
    }
  }
  return sumMs / (double)iters;
}

int mainCRTStartup() {
  puts("================================================================================");
  puts("   Vulkan LLM Decode: Research Question Ablation Benchmarks (8192 x 8192)       ");
  puts("================================================================================");

  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "AblationBenchmark";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instInfo = {0};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;

  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, NULL);
  VkPhysicalDevice *physDevs = (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, physDevs);
  VkPhysicalDevice physDev = physDevs[0];
  free(physDevs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);
  printf("GPU Device:       %s\n", props.deviceName);
  double tsPeriodNs = (double)props.limits.timestampPeriod;

  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
  VkQueueFamilyProperties *qfProps = (VkQueueFamilyProperties *)malloc(sizeof(VkQueueFamilyProperties) * qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);
  uint32_t qIdx = 0;
  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfProps[i].timestampValidBits > 0) {
      qIdx = i;
      break;
    }
  }
  free(qfProps);

  VkPhysicalDeviceVulkan13Features f13 = {0};
  f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  f13.shaderIntegerDotProduct = VK_TRUE;

  VkPhysicalDeviceVulkan12Features f12 = {0};
  f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  f12.pNext = &f13;
  f12.shaderFloat16 = VK_TRUE;
  f12.shaderSubgroupExtendedTypes = VK_TRUE;

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

  // Layouts: 3-binding and 4-binding
  VkDescriptorSetLayoutBinding b3[3] = {0};
  VkDescriptorSetLayoutBinding b4[4] = {0};
  for (uint32_t i = 0; i < 4; ++i) {
    b4[i].binding = i;
    b4[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b4[i].descriptorCount = 1;
    b4[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    if (i < 3) b3[i] = b4[i];
  }
  VkDescriptorSetLayoutCreateInfo dl3 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, b3};
  VkDescriptorSetLayoutCreateInfo dl4 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 4, b4};
  VkDescriptorSetLayout dsl3, dsl4;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dl3, NULL, &dsl3));
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dl4, NULL, &dsl4));

  VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo pl3Info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl3, 1, &pcRange};
  VkPipelineLayoutCreateInfo pl4Info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl4, 1, &pcRange};
  VkPipelineLayout pl3, pl4;
  VK_CHECK(vkCreatePipelineLayout(device, &pl3Info, NULL, &pl3));
  VK_CHECK(vkCreatePipelineLayout(device, &pl4Info, NULL, &pl4));

  VkPipeline pFusedInt4   = createComputePipeline(device, "spirv/gemv_int4.spv", pl4);
  VkPipeline pNoSubInt4   = createComputePipeline(device, "spirv/gemv_int4_nosubgroup.spv", pl4);
  VkPipeline pDequantInt4 = createComputePipeline(device, "spirv/dequant_int4_to_fp16.spv", pl3);
  VkPipeline pGemvFp16    = createComputePipeline(device, "spirv/gemv_fp16.spv", pl3);
  VkPipeline pInt8Scalar  = createComputePipeline(device, "spirv/gemv_int8.spv", pl4);
  VkPipeline pInt8Dp4a    = createComputePipeline(device, "spirv/gemv_int8_dp4a.spv", pl4);

  const uint32_t N = 8192, K = 8192;
  VkDeviceSize szX_fp16 = (VkDeviceSize)K * 2;
  VkDeviceSize szX_int8 = (VkDeviceSize)K;
  VkDeviceSize szW_int4 = (VkDeviceSize)N * (K / 2);
  VkDeviceSize szW_int8 = (VkDeviceSize)N * K;
  VkDeviceSize szW_fp16 = (VkDeviceSize)N * K * 2;
  VkDeviceSize szScales = (VkDeviceSize)N * 2;
  VkDeviceSize szY      = (VkDeviceSize)N * 2;

  VkBuffer bufX16, bufX8, bufW4, bufW8, bufW16, bufS, bufY;
  VkDeviceMemory memX16, memX8, memW4, memW8, memW16, memS, memY;
  VkMemoryPropertyFlags hostVis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryPropertyFlags devLoc  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VkBufferUsageFlags stg = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  createBuffer(device, physDev, szX_fp16, stg, hostVis, &bufX16, &memX16);
  createBuffer(device, physDev, szX_int8, stg, hostVis, &bufX8,  &memX8);
  createBuffer(device, physDev, szW_int4, stg, hostVis, &bufW4,  &memW4);
  createBuffer(device, physDev, szW_int8, stg, hostVis, &bufW8,  &memW8);
  createBuffer(device, physDev, szW_fp16, stg, devLoc,  &bufW16, &memW16);
  createBuffer(device, physDev, szScales, stg, hostVis, &bufS,   &memS);
  createBuffer(device, physDev, szY,      stg, hostVis, &bufY,   &memY);

  // Initialize deterministic test data
  uint16_t *pX16 = NULL; vkMapMemory(device, memX16, 0, szX_fp16, 0, (void **)&pX16);
  int8_t   *pX8  = NULL; vkMapMemory(device, memX8,  0, szX_int8, 0, (void **)&pX8);
  for (uint32_t i = 0; i < K; ++i) {
    int8_t v = (int8_t)((i % 5) - 2);
    pX8[i] = v;
    pX16[i] = float_to_half((float)v);
  }
  vkUnmapMemory(device, memX16);
  vkUnmapMemory(device, memX8);

  uint16_t *pS = NULL; vkMapMemory(device, memS, 0, szScales, 0, (void **)&pS);
  for (uint32_t r = 0; r < N; ++r) pS[r] = float_to_half(0.25f);
  vkUnmapMemory(device, memS);

  uint8_t *pW4 = NULL; vkMapMemory(device, memW4, 0, szW_int4, 0, (void **)&pW4);
  for (size_t i = 0; i < szW_int4; ++i) pW4[i] = (uint8_t)(((9) << 4) | 7); // -1 and +1 nibbles
  vkUnmapMemory(device, memW4);

  int8_t *pW8 = NULL; vkMapMemory(device, memW8, 0, szW_int8, 0, (void **)&pW8);
  for (size_t i = 0; i < szW_int8; ++i) pW8[i] = (int8_t)((i & 1) ? 1 : -1);
  vkUnmapMemory(device, memW8);

  VkDescriptorPoolSize dpSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
  VkDescriptorPoolCreateInfo dpInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 8, 1, &dpSize};
  VkDescriptorPool dp;
  VK_CHECK(vkCreateDescriptorPool(device, &dpInfo, NULL, &dp));

  VkDescriptorSetLayout layouts[6] = {dsl4, dsl4, dsl3, dsl3, dsl4, dsl4};
  VkDescriptorSet sets[6];
  VkDescriptorSetAllocateInfo dsAlloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 6, layouts};
  VK_CHECK(vkAllocateDescriptorSets(device, &dsAlloc, sets));

  // 0: Fused INT4, 1: NoSubgroup INT4, 2: Dequant INT4->FP16, 3: FP16 GEMV, 4: INT8 scalar, 5: INT8 DP4A
  updateSet4(device, sets[0], bufX16, szX_fp16, bufW4, szW_int4, bufS, szScales, bufY, szY);
  updateSet4(device, sets[1], bufX16, szX_fp16, bufW4, szW_int4, bufS, szScales, bufY, szY);
  updateSet3(device, sets[2], bufW4, szW_int4, bufS, szScales, bufW16, szW_fp16);
  updateSet3(device, sets[3], bufX16, szX_fp16, bufW16, szW_fp16, bufY, szY);
  updateSet4(device, sets[4], bufX16, szX_fp16, bufW8, szW_int8, bufS, szScales, bufY, szY);
  updateSet4(device, sets[5], bufX8,  szX_int8, bufW8, szW_int8, bufS, szScales, bufY, szY);

  VkCommandPoolCreateInfo cpInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qIdx};
  VkCommandPool cp;
  VK_CHECK(vkCreateCommandPool(device, &cpInfo, NULL, &cp));
  VkCommandBufferAllocateInfo cbAlloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL,
                                       cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAlloc, &cmd));

  VkQueryPoolCreateInfo qpInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
  VkQueryPool qp;
  VK_CHECK(vkCreateQueryPool(device, &qpInfo, NULL, &qp));

  const int WARMUP = 5, ITERS = 20;

  double msFusedInt4 = benchSinglePipeline(device, queue, cmd, qp, tsPeriodNs, pFusedInt4, pl4, sets[0], K / 32, N, WARMUP, ITERS);
  double msUnfused   = benchUnfusedInt4(device, queue, cmd, qp, tsPeriodNs, pDequantInt4, pGemvFp16, pl3, sets[2], sets[3], K, N, WARMUP, ITERS);
  double msNoSub     = benchSinglePipeline(device, queue, cmd, qp, tsPeriodNs, pNoSubInt4, pl4, sets[1], K / 32, N, WARMUP, ITERS);
  double msInt8      = benchSinglePipeline(device, queue, cmd, qp, tsPeriodNs, pInt8Scalar, pl4, sets[4], K / 16, N, WARMUP, ITERS);
  double msInt8Dp4a  = benchSinglePipeline(device, queue, cmd, qp, tsPeriodNs, pInt8Dp4a,   pl4, sets[5], K / 16, N, WARMUP, ITERS);

  puts("\n[Ablation 1: RQ #2 - Fusion Impact (INT4 8192x8192)]");
  printf("  Fused INT4 GEMV (in-register):      %7.3f ms  (1.00x)\n", msFusedInt4);
  printf("  Unfused 2-Pass (INT4->FP16 DRAM):   %7.3f ms  (%.2fx slower)\n", msUnfused, msUnfused / msFusedInt4);

  puts("\n[Ablation 2: RQ #3a - Subgroup Reduction (INT4 8192x8192)]");
  printf("  SIMD32 subgroupAdd:                 %7.3f ms  (1.00x)\n", msFusedInt4);
  printf("  SLM Tree Reduction (no subgroup):   %7.3f ms  (%.2fx slower)\n", msNoSub, msNoSub / msFusedInt4);

  puts("\n[Ablation 3: RQ #3b - Hardware DP4A vs FP16-Unpack (INT8 8192x8192)]");
  printf("  INT8 -> FP16 Unpack + dot:          %7.3f ms\n", msInt8);
  printf("  INT8 Hardware DP4A (dotPacked4x8):  %7.3f ms  (%.2fx vs unpack)\n", msInt8Dp4a, msInt8 / msInt8Dp4a);

  vkDeviceWaitIdle(device);
  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);
  return 0;
}

