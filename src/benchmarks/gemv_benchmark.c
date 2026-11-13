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

// Standard IEEE-754 Half-Precision (FP16) Software Conversion
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
      return (uint16_t)(sign | 0x7C00); // Inf
    return (uint16_t)(sign | 0x7C00 | (mant >> 13)); // NaN
  } else {
    if (exp > 30)
      return (uint16_t)(sign | 0x7C00); // Overflow to Inf
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

// Read SPIR-V Binary File
static uint32_t *readSpirvFile(const char *filename, size_t *outSize) {
  void *fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error: Could not open SPIR-V file: %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, 2); // SEEK_END
  long size = ftell(fp);
  fseek(fp, 0, 0); // SEEK_SET

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

// Find appropriate memory type
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

// Create a GPU buffer and allocate memory
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
  uint32_t K_param; // K / 4 for FP16, K / 16 for INT8, K / 32 for INT4
  uint32_t N;       // Number of rows
} PushConstants;

int mainCRTStartup() {
  puts("================================================================================");
  puts("    Vulkan LLM Decode: FP16 vs INT8 vs INT4 Fused GEMV Roofline Benchmark    ");
  puts("================================================================================");

  // 1. Create Vulkan Instance
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "GemvRooflineBenchmark";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instInfo = {0};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;

  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

  // 2. Select Physical Device
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
  printf("Driver Version:    %u.%u.%u\n", props.driverVersion >> 22,
         (props.driverVersion >> 12) & 0x3ff, props.driverVersion & 0xfff);
  printf("Vulkan API:        %u.%u.%u\n", VK_VERSION_MAJOR(props.apiVersion),
         VK_VERSION_MINOR(props.apiVersion),
         VK_VERSION_PATCH(props.apiVersion));
  printf("Timestamp Period:  %.4f ns\n", props.limits.timestampPeriod);

  // Query Subgroup Properties
  VkPhysicalDeviceSubgroupProperties subgroupProps = {0};
  subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2 = {0};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroupProps;
  vkGetPhysicalDeviceProperties2(physDev, &props2);

  printf("Subgroup Size:     %u threads (Intel SIMD32 EU)\n",
         subgroupProps.subgroupSize);
  printf("Subgroup Ops:      Arithmetic Supported: %s\n",
         (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)
             ? "YES"
             : "NO");

  // Find compute queue
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

  // 3. Create Logical Device with FP16 and Subgroup Extended Types Enabled
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

  // 4. Create Shaders and Pipelines for FP16, INT8, and INT4
  // FP16 Layout: 3 bindings (0: x, 1: W, 2: y)
  // INT8 Layout: 4 bindings (0: x, 1: W, 2: scale, 3: y)
  // INT4 Layout: 4 bindings (0: x, 1: W, 2: scale, 3: y)

  VkDescriptorSetLayout dslFP16, dslQuant;
  {
    VkDescriptorSetLayoutBinding b[3] = {0};
    for (int i = 0; i < 3; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslInfo = {0};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 3;
    dslInfo.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslFP16));
  }
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
    dslInfo.bindingCount = 4;
    dslInfo.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslQuant));
  }

  // Push Constant Range
  VkPushConstantRange pcRange = {0};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = sizeof(PushConstants);

  VkPipelineLayout plLayoutFP16, plLayoutQuant;
  {
    VkPipelineLayoutCreateInfo plInfo = {0};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dslFP16;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plLayoutFP16));

    plInfo.pSetLayouts = &dslQuant;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plLayoutQuant));
  }

  // Load SPIR-V modules
  size_t szFP16, szINT8, szINT4;
  uint32_t *spvFP16 = readSpirvFile("spirv/gemv_fp16.spv", &szFP16);
  uint32_t *spvINT8 = readSpirvFile("spirv/gemv_int8.spv", &szINT8);
  uint32_t *spvINT4 = readSpirvFile("spirv/gemv_int4.spv", &szINT4);

  VkShaderModule smFP16, smINT8, smINT4;
  {
    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = szFP16;
    smInfo.pCode = spvFP16;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smFP16));

    smInfo.codeSize = szINT8;
    smInfo.pCode = spvINT8;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smINT8));

    smInfo.codeSize = szINT4;
    smInfo.pCode = spvINT4;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smINT4));
  }
  free(spvFP16);
  free(spvINT8);
  free(spvINT4);

  // Compute Pipelines
  VkPipeline pipeFP16, pipeINT8, pipeINT4;
  {
    VkComputePipelineCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.pName = "main";

    cpInfo.stage.module = smFP16;
    cpInfo.layout = plLayoutFP16;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeFP16));

    cpInfo.stage.module = smINT8;
    cpInfo.layout = plLayoutQuant;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeINT8));

    cpInfo.stage.module = smINT4;
    cpInfo.layout = plLayoutQuant;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeINT4));
  }

  // 5. Allocate Max GPU Buffers
  const uint32_t MAX_N = 8192;
  const uint32_t MAX_K = 8192;
  const VkDeviceSize maxBytesW_FP16 = (VkDeviceSize)MAX_N * MAX_K * 2;       // 128 MB
  const VkDeviceSize maxBytesW_INT8 = (VkDeviceSize)MAX_N * MAX_K * 1;       // 64 MB
  const VkDeviceSize maxBytesW_INT4 = (VkDeviceSize)MAX_N * MAX_K / 2;       // 32 MB
  const VkDeviceSize maxBytesX      = (VkDeviceSize)MAX_K * sizeof(uint16_t); // 16 KB
  const VkDeviceSize maxBytesScale  = (VkDeviceSize)MAX_N * sizeof(uint16_t); // 16 KB
  const VkDeviceSize maxBytesY      = (VkDeviceSize)MAX_N * sizeof(uint16_t); // 16 KB

  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBuffer bufW_FP16, bufW_INT8, bufW_INT4, bufX, bufScale, bufY;
  VkDeviceMemory memW_FP16, memW_INT8, memW_INT4, memX, memScale, memY;

  createBuffer(device, physDev, maxBytesW_FP16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufW_FP16, &memW_FP16);
  createBuffer(device, physDev, maxBytesW_INT8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufW_INT8, &memW_INT8);
  createBuffer(device, physDev, maxBytesW_INT4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufW_INT4, &memW_INT4);
  createBuffer(device, physDev, maxBytesX,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufX,      &memX);
  createBuffer(device, physDev, maxBytesScale,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufScale,  &memScale);
  createBuffer(device, physDev, maxBytesY,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufY,      &memY);

  // Descriptor Pool & Sets
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}};
  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 4;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = poolSizes;

  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descPool));

  VkDescriptorSet descFP16, descINT8, descINT4;
  {
    VkDescriptorSetAllocateInfo allocSet = {0};
    allocSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSet.descriptorPool = descPool;
    allocSet.descriptorSetCount = 1;

    allocSet.pSetLayouts = &dslFP16;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descFP16));

    allocSet.pSetLayouts = &dslQuant;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descINT8));
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descINT4));
  }

  // Update FP16 Descriptor Set
  {
    VkDescriptorBufferInfo dbiX = {bufX, 0, maxBytesX};
    VkDescriptorBufferInfo dbiW = {bufW_FP16, 0, maxBytesW_FP16};
    VkDescriptorBufferInfo dbiY = {bufY, 0, maxBytesY};
    VkWriteDescriptorSet w[3] = {0};
    for (int i = 0; i < 3; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descFP16;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    w[0].pBufferInfo = &dbiX;
    w[1].pBufferInfo = &dbiW;
    w[2].pBufferInfo = &dbiY;
    vkUpdateDescriptorSets(device, 3, w, 0, NULL);
  }

  // Update INT8 Descriptor Set
  {
    VkDescriptorBufferInfo dbiX = {bufX, 0, maxBytesX};
    VkDescriptorBufferInfo dbiW = {bufW_INT8, 0, maxBytesW_INT8};
    VkDescriptorBufferInfo dbiS = {bufScale, 0, maxBytesScale};
    VkDescriptorBufferInfo dbiY = {bufY, 0, maxBytesY};
    VkWriteDescriptorSet w[4] = {0};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descINT8;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    w[0].pBufferInfo = &dbiX;
    w[1].pBufferInfo = &dbiW;
    w[2].pBufferInfo = &dbiS;
    w[3].pBufferInfo = &dbiY;
    vkUpdateDescriptorSets(device, 4, w, 0, NULL);
  }

  // Update INT4 Descriptor Set
  {
    VkDescriptorBufferInfo dbiX = {bufX, 0, maxBytesX};
    VkDescriptorBufferInfo dbiW = {bufW_INT4, 0, maxBytesW_INT4};
    VkDescriptorBufferInfo dbiS = {bufScale, 0, maxBytesScale};
    VkDescriptorBufferInfo dbiY = {bufY, 0, maxBytesY};
    VkWriteDescriptorSet w[4] = {0};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descINT4;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    w[0].pBufferInfo = &dbiX;
    w[1].pBufferInfo = &dbiW;
    w[2].pBufferInfo = &dbiS;
    w[3].pBufferInfo = &dbiY;
    vkUpdateDescriptorSets(device, 4, w, 0, NULL);
  }

  // 6. Query Pool & Command Pool
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

  // Benchmark Kernel Descriptors
  typedef struct {
    const char *name;
    const char *dtype;
    double bytesPerWeight;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descSet;
    VkDeviceMemory memW;
    int kDivisor;
  } KernelMode;

  KernelMode modes[3] = {
      {"FP16 Baseline GEMV", "FP16 (16-bit)", 2.0, pipeFP16, plLayoutFP16, descFP16, memW_FP16, 4},
      {"INT8 Fused Dequant-GEMV", "INT8 (8-bit)", 1.0, pipeINT8, plLayoutQuant, descINT8, memW_INT8, 16},
      {"INT4 Fused Dequant-GEMV", "INT4 (4-bit)", 0.5, pipeINT4, plLayoutQuant, descINT4, memW_INT4, 32}};

  // Standard Test Shapes
  struct {
    const char *desc;
    uint32_t N;
    uint32_t K;
  } shapes[] = {
      {"LLaMA-3 / Qwen-2.5 Hidden Layer", 4096, 4096},
      {"Full DRAM Bus Saturation (8K x 8K)", 8192, 8192}};

  const double DRAM_ROOFLINE_GBPS = 16.28;

  for (int s = 0; s < 2; ++s) {
    uint32_t N = shapes[s].N;
    uint32_t K = shapes[s].K;
    double gflops = (2.0 * (double)N * (double)K) / 1.0e9;

    printf("\n################################################################################\n");
    printf("   BENCHMARK SHAPE: %s (N = %u, K = %u, %.3f GFLOPs)\n", shapes[s].desc, N, K, gflops);
    printf("################################################################################\n");

    // Populate common Input Vector x and Scales
    uint16_t *ptrX;
    uint16_t *ptrScale;
    VK_CHECK(vkMapMemory(device, memX, 0, (VkDeviceSize)K * sizeof(uint16_t), 0, (void **)&ptrX));
    VK_CHECK(vkMapMemory(device, memScale, 0, (VkDeviceSize)N * sizeof(uint16_t), 0, (void **)&ptrScale));
    for (uint32_t j = 0; j < K; ++j) {
      ptrX[j] = float_to_half(((float)((j % 7) - 3)) * 0.05f);
    }
    const float testScale = 0.015625f; // 1/64
    for (uint32_t i = 0; i < N; ++i) {
      ptrScale[i] = float_to_half(testScale);
    }
    vkUnmapMemory(device, memX);
    vkUnmapMemory(device, memScale);

    for (int m = 0; m < 3; ++m) {
      KernelMode *km = &modes[m];
      double matrixBytes = (double)N * (double)K * km->bytesPerWeight;
      double matrixMB = matrixBytes / (1024.0 * 1024.0);
      double totalTransferred = matrixBytes + ((double)K + (double)N) * 2.0;

      // Initialize Weights for this mode
      uint8_t *ptrW;
      VK_CHECK(vkMapMemory(device, km->memW, 0, (VkDeviceSize)matrixBytes, 0, (void **)&ptrW));
      
      const uint32_t verifyRows = 32;
      float cpuRef[32] = {0};

      if (m == 0) {
        // FP16: w in [-0.1, 0.1]
        uint16_t *wFP16 = (uint16_t *)ptrW;
        for (uint32_t i = 0; i < N; ++i) {
          for (uint32_t j = 0; j < K; ++j) {
            float val = ((float)(((i + j) % 9) - 4)) * 0.02f;
            wFP16[i * K + j] = float_to_half(val);
          }
        }
        for (uint32_t i = 0; i < verifyRows; ++i) {
          float sum = 0.0f;
          for (uint32_t j = 0; j < K; ++j) {
            sum += half_to_float(wFP16[i * K + j]) * half_to_float(ptrX[j]);
          }
          cpuRef[i] = sum;
        }
      } else if (m == 1) {
        // INT8: signed int8 in [-7, 7]
        int8_t *wINT8 = (int8_t *)ptrW;
        for (uint32_t i = 0; i < N; ++i) {
          for (uint32_t j = 0; j < K; ++j) {
            int8_t val = (int8_t)(((i + j) % 15) - 7);
            wINT8[i * K + j] = val;
          }
        }
        for (uint32_t i = 0; i < verifyRows; ++i) {
          float sum = 0.0f;
          for (uint32_t j = 0; j < K; ++j) {
            sum += (float)wINT8[i * K + j] * half_to_float(ptrX[j]);
          }
          cpuRef[i] = sum * testScale;
        }
      } else if (m == 2) {
        // INT4: unsigned nibble in [0, 15] (representing signed [-8, 7])
        memset(ptrW, 0, (size_t)matrixBytes);
        for (uint32_t i = 0; i < N; ++i) {
          for (uint32_t j = 0; j < K; ++j) {
            int signedVal = ((int)((i + j) % 15) - 7); // -7..7
            uint8_t nibble = (uint8_t)(signedVal + 8) & 0x0F;
            uint64_t linear = (uint64_t)i * K + j;
            uint64_t byteIdx = linear / 2;
            if (j % 2 == 0) {
              ptrW[byteIdx] |= nibble;
            } else {
              ptrW[byteIdx] |= (nibble << 4);
            }
          }
        }
        for (uint32_t i = 0; i < verifyRows; ++i) {
          float sum = 0.0f;
          for (uint32_t j = 0; j < K; ++j) {
            int signedVal = ((int)((i + j) % 15) - 7);
            sum += (float)signedVal * half_to_float(ptrX[j]);
          }
          cpuRef[i] = sum * testScale;
        }
      }
      vkUnmapMemory(device, km->memW);

      // Record and execute compute kernel
      VkCommandBufferBeginInfo cbBegin = {0};
      cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));

      vkCmdResetQueryPool(cmd, queryPool, 0, 2);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, km->pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, km->layout, 0, 1, &km->descSet, 0, NULL);

      PushConstants pc;
      pc.K_param = K / km->kDivisor;
      pc.N = N;
      vkCmdPushConstants(cmd, km->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      uint32_t groupsX = (N + 3) / 4;
      vkCmdDispatch(cmd, groupsX, 1, 1);

      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
      VK_CHECK(vkEndCommandBuffer(cmd));

      VkSubmitInfo submitInfo = {0};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &cmd;

      // Warmup (3 iterations)
      for (int w = 0; w < 3; ++w) {
        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
      }

      // Benchmark (15 iterations)
      const int iterations = 15;
      double bestMs = 1e9;
      double totalMs = 0;
      uint64_t timestamps[2] = {0};

      for (int iter = 0; iter < iterations; ++iter) {
        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));

        VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(timestamps),
                                       timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
        double timeNs = (double)(timestamps[1] - timestamps[0]) * (double)props.limits.timestampPeriod;
        double timeMs = timeNs / 1.0e6;
        if (timeMs < bestMs) bestMs = timeMs;
        totalMs += timeMs;
      }

      double avgMs = totalMs / iterations;
      double peakBandwidth = (totalTransferred / (bestMs / 1000.0)) / 1.0e9;
      double avgBandwidth = (totalTransferred / (avgMs / 1000.0)) / 1.0e9;
      double tflops = gflops / bestMs;
      double rooflineEff = (peakBandwidth / DRAM_ROOFLINE_GBPS) * 100.0;

      // Verify Output against CPU reference
      uint16_t *ptrY;
      VK_CHECK(vkMapMemory(device, memY, 0, (VkDeviceSize)N * sizeof(uint16_t), 0, (void **)&ptrY));
      int verified = 1;
      float maxDiff = 0.0f;
      for (uint32_t i = 0; i < verifyRows; ++i) {
        float gpuVal = half_to_float(ptrY[i]);
        float diff = abs_f(gpuVal - cpuRef[i]);
        if (diff > maxDiff) maxDiff = diff;
        float relErr = diff / (abs_f(cpuRef[i]) + 1e-4f);
        if (relErr > 0.05f && diff > 0.05f) {
          printf("    Mismatch at row %u: GPU=%.4f, CPU=%.4f (diff=%.4f)\n", i, gpuVal, cpuRef[i], diff);
          verified = 0;
          break;
        }
      }
      vkUnmapMemory(device, memY);

      printf("\n  ▶ [%s] (%s)\n", km->name, km->dtype);
      printf("    Matrix Weight Size:    %6.2f MB (%.1fx compression vs FP16)\n", matrixMB, 2.0 / km->bytesPerWeight);
      printf("    Kernel Latency:        %6.3f ms (avg %6.3f ms)\n", bestMs, avgMs);
      printf("    Memory Bandwidth:      %6.2f GB/s (%.1f%% of %.2f GB/s DRAM roofline)\n", peakBandwidth, rooflineEff, DRAM_ROOFLINE_GBPS);
      printf("    Throughput:            %6.1f inferences/sec\n", 1000.0 / bestMs);
      printf("    0.5B Model Token Rate: %6.1f tokens/sec (theoretical full-model ceiling)\n",
             (DRAM_ROOFLINE_GBPS / (0.5 * km->bytesPerWeight)));
      printf("    Verification:          %s (max diff: %.5f across %u rows)\n", verified ? "PASSED" : "FAILED", maxDiff, verifyRows);
    }
  }

  puts("\n================================================================================");
  puts("Benchmark completed cleanly!");

  // Cleanup
  vkDestroyBuffer(device, bufW_FP16, NULL);
  vkFreeMemory(device, memW_FP16, NULL);
  vkDestroyBuffer(device, bufW_INT8, NULL);
  vkFreeMemory(device, memW_INT8, NULL);
  vkDestroyBuffer(device, bufW_INT4, NULL);
  vkFreeMemory(device, memW_INT4, NULL);
  vkDestroyBuffer(device, bufX, NULL);
  vkFreeMemory(device, memX, NULL);
  vkDestroyBuffer(device, bufScale, NULL);
  vkFreeMemory(device, memScale, NULL);
  vkDestroyBuffer(device, bufY, NULL);
  vkFreeMemory(device, memY, NULL);

  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);
  vkDestroyDescriptorPool(device, descPool, NULL);
  vkDestroyPipeline(device, pipeFP16, NULL);
  vkDestroyPipeline(device, pipeINT8, NULL);
  vkDestroyPipeline(device, pipeINT4, NULL);
  vkDestroyPipelineLayout(device, plLayoutFP16, NULL);
  vkDestroyPipelineLayout(device, plLayoutQuant, NULL);
  vkDestroyShaderModule(device, smFP16, NULL);
  vkDestroyShaderModule(device, smINT8, NULL);
  vkDestroyShaderModule(device, smINT4, NULL);
  vkDestroyDescriptorSetLayout(device, dslFP16, NULL);
  vkDestroyDescriptorSetLayout(device, dslQuant, NULL);

  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);

  exit(0);
}
