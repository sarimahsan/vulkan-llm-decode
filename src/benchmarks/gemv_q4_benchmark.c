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

// Read SPIR-V Binary File
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

// Native GGUF Q4_0 Block in C (18 bytes per 32 weights)
#pragma pack(push, 1)
typedef struct {
  uint16_t d;     // FP16 scale delta
  uint8_t qs[16]; // 32 nibbles (qs[0..15]: low=0..15, high=16..31)
} GgufBlockQ4_0;
#pragma pack(pop)

typedef struct {
  uint32_t K_param; // K / 32
  uint32_t N;       // Number of rows
} PushConstants;

int mainCRTStartup() {
  puts("================================================================================");
  puts("     Vulkan LLM Decode: Group-32 Quantization & GGUF Q4_0 Roofline Engine      ");
  puts("================================================================================");

  // 1. Create Vulkan Instance
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "GgufQ4Benchmark";
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

  // Subgroup Properties
  VkPhysicalDeviceSubgroupProperties subgroupProps = {0};
  subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2 = {0};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroupProps;
  vkGetPhysicalDeviceProperties2(physDev, &props2);

  printf("Subgroup Size:     %u threads (Intel SIMD32 EU)\n",
         subgroupProps.subgroupSize);

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

  // 3. Create Logical Device with FP16, Subgroup Extended Types, and 8-bit / 16-bit Storage
  VkPhysicalDeviceVulkan12Features features12 = {0};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features12.shaderFloat16 = VK_TRUE;
  features12.shaderSubgroupExtendedTypes = VK_TRUE;
  features12.storageBuffer8BitAccess = VK_TRUE;

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

  // 4. Create Shaders and Pipelines
  // Pipeline 1: Group-32 Linear (AWQ style: x, W, scales, y) -> 4 bindings
  // Pipeline 2: Native GGUF Q4_0 (GGUF style: x, blocks, y)  -> 3 bindings
  VkDescriptorSetLayout dslGroup32, dslGguf;
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
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslGroup32));

    dslInfo.bindingCount = 3;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dslGguf));
  }

  // Push Constant Range
  VkPushConstantRange pcRange = {0};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = sizeof(PushConstants);

  VkPipelineLayout plLayoutGroup32, plLayoutGguf;
  {
    VkPipelineLayoutCreateInfo plInfo = {0};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    plInfo.pSetLayouts = &dslGroup32;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plLayoutGroup32));

    plInfo.pSetLayouts = &dslGguf;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &plLayoutGguf));
  }

  // Load SPIR-V modules
  size_t szGroup32, szGguf;
  uint32_t *spvGroup32 = readSpirvFile("spirv/gemv_int4_group32.spv", &szGroup32);
  uint32_t *spvGguf = readSpirvFile("spirv/gemv_q4_0.spv", &szGguf);

  VkShaderModule smGroup32, smGguf;
  {
    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = szGroup32;
    smInfo.pCode = spvGroup32;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smGroup32));

    smInfo.codeSize = szGguf;
    smInfo.pCode = spvGguf;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &smGguf));
  }
  free(spvGroup32);
  free(spvGguf);

  // Compute Pipelines
  VkPipeline pipeGroup32, pipeGguf;
  {
    VkComputePipelineCreateInfo cpInfo = {0};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.pName = "main";

    cpInfo.stage.module = smGroup32;
    cpInfo.layout = plLayoutGroup32;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeGroup32));

    cpInfo.stage.module = smGguf;
    cpInfo.layout = plLayoutGguf;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeGguf));
  }

  // 5. Allocate Max GPU Buffers
  const uint32_t MAX_N = 8192;
  const uint32_t MAX_K = 8192;
  const uint32_t MAX_BLOCKS = MAX_N * (MAX_K / 32); // 2,097,152 blocks

  const VkDeviceSize maxBytesW_Group32 = (VkDeviceSize)MAX_BLOCKS * 16;              // 32 MB
  const VkDeviceSize maxBytesScale      = (VkDeviceSize)MAX_BLOCKS * sizeof(uint16_t); // 4 MB
  const VkDeviceSize maxBytesW_Gguf     = (VkDeviceSize)MAX_BLOCKS * sizeof(GgufBlockQ4_0); // 36 MB
  const VkDeviceSize maxBytesX          = (VkDeviceSize)MAX_K * sizeof(uint16_t);      // 16 KB
  const VkDeviceSize maxBytesY          = (VkDeviceSize)MAX_N * sizeof(uint16_t);      // 16 KB

  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBuffer bufW_Group32, bufScale, bufW_Gguf, bufX, bufY;
  VkDeviceMemory memW_Group32, memScale, memW_Gguf, memX, memY;

  createBuffer(device, physDev, maxBytesW_Group32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufW_Group32, &memW_Group32);
  createBuffer(device, physDev, maxBytesScale,     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufScale,     &memScale);
  createBuffer(device, physDev, maxBytesW_Gguf,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufW_Gguf,    &memW_Gguf);
  createBuffer(device, physDev, maxBytesX,         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufX,         &memX);
  createBuffer(device, physDev, maxBytesY,         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufY,         &memY);

  // Descriptor Pool & Sets
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}};
  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 2;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = poolSizes;

  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descPool));

  VkDescriptorSet descGroup32, descGguf;
  {
    VkDescriptorSetAllocateInfo allocSet = {0};
    allocSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSet.descriptorPool = descPool;
    allocSet.descriptorSetCount = 1;

    allocSet.pSetLayouts = &dslGroup32;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descGroup32));

    allocSet.pSetLayouts = &dslGguf;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocSet, &descGguf));
  }

  // Bind Group-32 Linear Set (0: x, 1: W, 2: scale, 3: y)
  {
    VkDescriptorBufferInfo dbiX = {bufX, 0, maxBytesX};
    VkDescriptorBufferInfo dbiW = {bufW_Group32, 0, maxBytesW_Group32};
    VkDescriptorBufferInfo dbiS = {bufScale, 0, maxBytesScale};
    VkDescriptorBufferInfo dbiY = {bufY, 0, maxBytesY};
    VkWriteDescriptorSet w[4] = {0};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descGroup32;
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

  // Bind Native GGUF Set (0: x, 1: blocks, 2: y)
  {
    VkDescriptorBufferInfo dbiX = {bufX, 0, maxBytesX};
    VkDescriptorBufferInfo dbiB = {bufW_Gguf, 0, maxBytesW_Gguf};
    VkDescriptorBufferInfo dbiY = {bufY, 0, maxBytesY};
    VkWriteDescriptorSet w[3] = {0};
    for (int i = 0; i < 3; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = descGguf;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    w[0].pBufferInfo = &dbiX;
    w[1].pBufferInfo = &dbiB;
    w[2].pBufferInfo = &dbiY;
    vkUpdateDescriptorSets(device, 3, w, 0, NULL);
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

  // Benchmark Suites
  struct {
    const char *desc;
    uint32_t N;
    uint32_t K;
  } shapes[] = {
      {"LLaMA-3 / Qwen-2.5 Hidden Dimension", 4096, 4096},
      {"Full DRAM Bus Saturation (8K x 8K)", 8192, 8192}};

  const double DRAM_ROOFLINE_GBPS = 16.28;

  for (int s = 0; s < 2; ++s) {
    uint32_t N = shapes[s].N;
    uint32_t K = shapes[s].K;
    uint32_t numBlocksPerRow = K / 32;
    uint32_t totalBlocks = N * numBlocksPerRow;
    double gflops = (2.0 * (double)N * (double)K) / 1.0e9;

    printf("\n################################################################################\n");
    printf("   SHAPE: %s (N = %u, K = %u, %.3f GFLOPs)\n", shapes[s].desc, N, K, gflops);
    printf("################################################################################\n");

    // Initialize Host Input Vector x
    uint16_t *ptrX;
    VK_CHECK(vkMapMemory(device, memX, 0, (VkDeviceSize)K * sizeof(uint16_t), 0, (void **)&ptrX));
    for (uint32_t j = 0; j < K; ++j) {
      ptrX[j] = float_to_half(((float)((j % 7) - 3)) * 0.05f);
    }
    vkUnmapMemory(device, memX);

    // Populate Group-32 Split Data (bufW_Group32 + bufScale) and GGUF Data (bufW_Gguf) with identical numbers
    uint8_t *ptrW_Group32;
    uint16_t *ptrScale;
    GgufBlockQ4_0 *ptrW_Gguf;

    VK_CHECK(vkMapMemory(device, memW_Group32, 0, (VkDeviceSize)totalBlocks * 16, 0, (void **)&ptrW_Group32));
    VK_CHECK(vkMapMemory(device, memScale, 0, (VkDeviceSize)totalBlocks * sizeof(uint16_t), 0, (void **)&ptrScale));
    VK_CHECK(vkMapMemory(device, memW_Gguf, 0, (VkDeviceSize)totalBlocks * sizeof(GgufBlockQ4_0), 0, (void **)&ptrW_Gguf));

    const uint32_t verifyRows = 32;
    float cpuRef[32] = {0};

    for (uint32_t i = 0; i < N; ++i) {
      for (uint32_t b = 0; b < numBlocksPerRow; ++b) {
        uint32_t blockIdx = i * numBlocksPerRow + b;
        float d_val = 0.015625f * (1.0f + ((float)((blockIdx % 5)) * 0.1f));
        uint16_t d_fp16 = float_to_half(d_val);

        // Scales
        ptrScale[blockIdx] = d_fp16;
        ptrW_Gguf[blockIdx].d = d_fp16;

        // Populate 32 weights per block
        // Group-32 linear stores: nibble 0 at bits 0..3, nibble 1 at bits 4..7 ...
        // GGUF Q4_0 stores: low nibbles of qs[0..15] are weights 0..15, high nibbles are weights 16..31
        uint8_t *w_linear = &ptrW_Group32[blockIdx * 16];
        memset(w_linear, 0, 16);

        for (uint32_t w_idx = 0; w_idx < 32; ++w_idx) {
          int signedVal = (int)(((i + b * 32 + w_idx) % 15) - 7); // -7..7
          uint8_t nibble = (uint8_t)(signedVal + 8) & 0x0F;

          // Write to Group-32 linear
          if (w_idx % 2 == 0) {
            w_linear[w_idx / 2] |= nibble;
          } else {
            w_linear[w_idx / 2] |= (nibble << 4);
          }

          // Write to GGUF Q4_0 block
          if (w_idx < 16) {
            ptrW_Gguf[blockIdx].qs[w_idx] = (ptrW_Gguf[blockIdx].qs[w_idx] & 0xF0) | nibble;
          } else {
            uint32_t low_idx = w_idx - 16;
            ptrW_Gguf[blockIdx].qs[low_idx] = (ptrW_Gguf[blockIdx].qs[low_idx] & 0x0F) | (nibble << 4);
          }

          // CPU Reference for the first verifyRows
          if (i < verifyRows) {
            uint32_t col = b * 32 + w_idx;
            cpuRef[i] += (float)signedVal * d_val * half_to_float(ptrX[col]);
          }
        }
      }
    }

    vkUnmapMemory(device, memW_Group32);
    vkUnmapMemory(device, memScale);
    vkUnmapMemory(device, memW_Gguf);

    // Benchmark both kernels
    struct {
      const char *name;
      const char *format;
      double bytesPerParam;
      double totalBytes;
      VkPipeline pipeline;
      VkPipelineLayout layout;
      VkDescriptorSet descSet;
    } subTests[2] = {
        {"Group-32 Linear (AWQ/GPTQ Split)", "INT4 + FP16 scale (4.5 bpp)", 0.5625,
         (double)totalBlocks * 18.0 + (double)(K + N) * 2.0, pipeGroup32, plLayoutGroup32, descGroup32},
        {"Native GGUF Q4_0 (Interleaved)", "18-byte BlockQ4_0 (4.5 bpp)", 0.5625,
         (double)totalBlocks * 18.0 + (double)(K + N) * 2.0, pipeGguf, plLayoutGguf, descGguf}};

    for (int t = 0; t < 2; ++t) {
      // Record command buffer
      VkCommandBufferBeginInfo cbBegin = {0};
      cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));

      vkCmdResetQueryPool(cmd, queryPool, 0, 2);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, subTests[t].pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, subTests[t].layout, 0, 1, &subTests[t].descSet, 0, NULL);

      PushConstants pc;
      pc.K_param = numBlocksPerRow;
      pc.N = N;
      vkCmdPushConstants(cmd, subTests[t].layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

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
      double peakBandwidth = (subTests[t].totalBytes / (bestMs / 1000.0)) / 1.0e9;
      double rooflineEff = (peakBandwidth / DRAM_ROOFLINE_GBPS) * 100.0;
      double matrixMB = (double)totalBlocks * 18.0 / (1024.0 * 1024.0);

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

      printf("\n  ▶ [%s]\n", subTests[t].name);
      printf("    Format:                %s\n", subTests[t].format);
      printf("    Matrix Weight Size:    %6.2f MB (3.56x compression vs FP16)\n", matrixMB);
      printf("    Kernel Latency:        %6.3f ms (avg %6.3f ms)\n", bestMs, avgMs);
      printf("    Memory Bandwidth:      %6.2f GB/s (%.1f%% of %.2f GB/s DRAM roofline)\n", peakBandwidth, rooflineEff, DRAM_ROOFLINE_GBPS);
      printf("    Throughput:            %6.1f inferences/sec\n", 1000.0 / bestMs);
      printf("    0.5B Model Token Rate: %6.1f tokens/sec (full-model decode ceiling)\n",
             (DRAM_ROOFLINE_GBPS / (0.5 * subTests[t].bytesPerParam)));
      printf("    Verification:          %s (max diff: %.5f across %u rows)\n", verified ? "PASSED" : "FAILED", maxDiff, verifyRows);
    }
  }

  puts("\n================================================================================");
  puts("Benchmark completed cleanly!");

  // Cleanup
  vkDestroyBuffer(device, bufW_Group32, NULL);
  vkFreeMemory(device, memW_Group32, NULL);
  vkDestroyBuffer(device, bufScale, NULL);
  vkFreeMemory(device, memScale, NULL);
  vkDestroyBuffer(device, bufW_Gguf, NULL);
  vkFreeMemory(device, memW_Gguf, NULL);
  vkDestroyBuffer(device, bufX, NULL);
  vkFreeMemory(device, memX, NULL);
  vkDestroyBuffer(device, bufY, NULL);
  vkFreeMemory(device, memY, NULL);

  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);
  vkDestroyDescriptorPool(device, descPool, NULL);
  vkDestroyPipeline(device, pipeGroup32, NULL);
  vkDestroyPipeline(device, pipeGguf, NULL);
  vkDestroyPipelineLayout(device, plLayoutGroup32, NULL);
  vkDestroyPipelineLayout(device, plLayoutGguf, NULL);
  vkDestroyShaderModule(device, smGroup32, NULL);
  vkDestroyShaderModule(device, smGguf, NULL);
  vkDestroyDescriptorSetLayout(device, dslGroup32, NULL);
  vkDestroyDescriptorSetLayout(device, dslGguf, NULL);

  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);

  exit(0);
}

