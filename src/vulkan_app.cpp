#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>

#include "Logger/Logger.hpp"
#include "vulkan_app.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

// Validation layers
static const std::vector<const char *> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
static constexpr bool ENABLE_VALIDATION = false;
#else
static constexpr bool ENABLE_VALIDATION = true;
#endif

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult _r = (x);                                                         \
    if (_r != VK_SUCCESS)                                                      \
      throw std::runtime_error(std::string(#x) +                               \
                               " failed: " + std::to_string(_r));              \
  } while (0)

// Entry point
void VulkanApp::run() {
  init_window();
  init_vulkan();
  main_loop();
  cleanup();
}

void VulkanApp::init_window() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window = glfwCreateWindow(W, H, "GPU Raytracer — WASD to move, Esc to quit",
                            nullptr, nullptr);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanApp::init_vulkan() {
  create_instance();
  if (validation_enabled)
    setup_debug_messenger();
  create_surface();
  pick_physical_device();
  create_logical_device();
  {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = gfx_family;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &command_pool));
  }
  create_swapchain();
  create_render_images();
  load_gltf("/media/data/vulkan-gpu/Object3d/dodge_challegner_srt_hellcat.glb");
  upload_scene();
  create_blas();
  create_tlas();
  create_compute_pipeline();
  create_display_pipeline();
  create_commands_and_sync();
}

void VulkanApp::main_loop() {
  uint32_t flight = 0;
  last_time = glfwGetTime();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GLFW_TRUE);

    double now = glfwGetTime();
    float dt = (float)(now - last_time);
    last_time = now;

    if (process_input(dt))
      frame_index = 0;

    update_camera_ubo();
    draw_frame(flight);
    flight = (flight + 1) % MAX_FRAMES;
  }
  vkDeviceWaitIdle(device);
}

// Instance
static bool check_validation_layer_support() {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> available(count);
  vkEnumerateInstanceLayerProperties(&count, available.data());
  for (const char *name : VALIDATION_LAYERS) {
    bool found = false;
    for (const auto &p : available)
      if (strcmp(name, p.layerName) == 0) {
        found = true;
        break;
      }
    if (!found)
      return false;
  }
  return true;
}

void VulkanApp::create_instance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "GPU Raytracer";
  app_info.apiVersion = VK_API_VERSION_1_2;

  validation_enabled = ENABLE_VALIDATION && check_validation_layer_support();
  if (ENABLE_VALIDATION && !validation_enabled)
    LOG_WARN("[Vulkan] Validation layers not available, disabling.");

  uint32_t glfw_count = 0;
  const char **glfw_ext = glfwGetRequiredInstanceExtensions(&glfw_count);
  std::vector<const char *> extensions(glfw_ext, glfw_ext + glfw_count);
  if (validation_enabled)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef __APPLE__
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app_info;
  ci.enabledExtensionCount = (uint32_t)extensions.size();
  ci.ppEnabledExtensionNames = extensions.data();
#ifdef __APPLE__
  ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
  if (validation_enabled) {
    ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
    ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
  }
  VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    std::cerr << "[Vulkan] " << data->pMessage << "\n";
  return VK_FALSE;
}

void VulkanApp::setup_debug_messenger() {
  VkDebugUtilsMessengerCreateInfoEXT ci{};
  ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  ci.pfnUserCallback = debug_callback;

  auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (fn)
    fn(instance, &ci, nullptr, &debug_messenger);
}

void VulkanApp::create_surface() {
  VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

// Physical device
void VulkanApp::pick_physical_device() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  if (!count)
    throw std::runtime_error("No Vulkan GPU found");
  std::vector<VkPhysicalDevice> devs(count);
  vkEnumeratePhysicalDevices(instance, &count, devs.data());

  // Prefer discrete GPU
  for (auto d : devs) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      phys_dev = d;
      LOG_INFO(std::format("Used GPU [{}]", p.deviceName));
      return;
    }
  }
  phys_dev = devs[0];
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(phys_dev, &p);
  LOG_INFO(std::format("Used GPU [{}]", p.deviceName));
}

// Logical device
void VulkanApp::create_logical_device() {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, families.data());

  gfx_family = UINT32_MAX;
  for (uint32_t i = 0; i < count; i++) {
    VkBool32 present = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(phys_dev, i, surface, &present);
    if ((families[i].queueFlags &
         (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) &&
        present) {
      gfx_family = i;
      break;
    }
  }
  if (gfx_family == UINT32_MAX)
    throw std::runtime_error("No suitable queue family");

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = gfx_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

 // 1. Les extensions nécessaires au Hardware RT
  std::vector<const char *> dev_exts = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
  };
#ifdef __APPLE__
  dev_exts.push_back("VK_KHR_portability_subset");
#endif

  VkPhysicalDeviceBufferDeviceAddressFeatures bda_feats{};
  bda_feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  bda_feats.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceDescriptorIndexingFeatures di_feats{};
  di_feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  di_feats.runtimeDescriptorArray                    = VK_TRUE;
  di_feats.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  di_feats.descriptorBindingVariableDescriptorCount  = VK_TRUE;
  di_feats.descriptorBindingPartiallyBound           = VK_TRUE;
  di_feats.pNext = &bda_feats;

  VkPhysicalDeviceRayQueryFeaturesKHR rq_feats{};
  rq_feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
  rq_feats.rayQuery = VK_TRUE;
  rq_feats.pNext = &di_feats;

  VkPhysicalDeviceAccelerationStructureFeaturesKHR as_feats{};
  as_feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  as_feats.accelerationStructure = VK_TRUE;
  as_feats.pNext = &rq_feats;

  VkPhysicalDeviceFeatures feats{};
  feats.shaderInt64 = VK_TRUE;

  VkDeviceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  ci.pNext = &as_feats;
  ci.queueCreateInfoCount = 1;
  ci.pQueueCreateInfos = &qci;
  ci.enabledExtensionCount = (uint32_t)dev_exts.size();
  ci.ppEnabledExtensionNames = dev_exts.data();
  ci.pEnabledFeatures = &feats;

  if (validation_enabled) {
    ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
    ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
  }

  VK_CHECK(vkCreateDevice(phys_dev, &ci, nullptr, &device));
  vkGetDeviceQueue(device, gfx_family, 0, &gfx_queue);
}

// Swapchain
void VulkanApp::create_swapchain() {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev, surface, &caps);

  uint32_t fmt_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmt_count, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(fmt_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmt_count,
                                       fmts.data());

  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto &f : fmts)
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      chosen = f;
      break;
    }

  sc_format = chosen.format;
  sc_extent = caps.currentExtent;
  if (sc_extent.width == UINT32_MAX)
    sc_extent = {W, H};

  uint32_t img_count = caps.minImageCount + 1;
  if (caps.maxImageCount > 0)
    img_count = std::min(img_count, caps.maxImageCount);

  VkSwapchainCreateInfoKHR ci{};
  ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  ci.surface = surface;
  ci.minImageCount = img_count;
  ci.imageFormat = sc_format;
  ci.imageColorSpace = chosen.colorSpace;
  ci.imageExtent = sc_extent;
  ci.imageArrayLayers = 1;
  ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.preTransform = caps.currentTransform;
  ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  ci.clipped = VK_TRUE;
  VK_CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain));

  vkGetSwapchainImagesKHR(device, swapchain, &img_count, nullptr);
  sc_images.resize(img_count);
  vkGetSwapchainImagesKHR(device, swapchain, &img_count, sc_images.data());

  sc_views.resize(img_count);
  for (uint32_t i = 0; i < img_count; i++)
    sc_views[i] = create_image_view(sc_images[i], sc_format);
}

// Render images
void VulkanApp::create_render_images() {
  // Accumulation image: rgba32f, storage image
  create_image2d(W, H, VK_FORMAT_R32G32B32A32_SFLOAT,
                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                 accum_image, accum_mem);
  accum_view = create_image_view(accum_image, VK_FORMAT_R32G32B32A32_SFLOAT);

  // Display image: rgba8, written by compute, sampled by fragment
  create_image2d(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 display_image, display_mem);
  display_view = create_image_view(display_image, VK_FORMAT_R8G8B8A8_UNORM);

  // Transition both to GENERAL
  VkCommandBuffer cmd = begin_one_shot();
  transition_image(cmd, accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_GENERAL);
  transition_image(cmd, display_image, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_GENERAL);
  end_one_shot(cmd);

  // Sampler for display pass
  VkSamplerCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sci.magFilter = VK_FILTER_NEAREST;
  sci.minFilter = VK_FILTER_NEAREST;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VK_CHECK(vkCreateSampler(device, &sci, nullptr, &display_sampler));
}

// Scene
/*
void VulkanApp::build_scene() {
  auto sphere = [](glm::vec3 c, float r, glm::vec3 col, float emit = 0,
                   int mat = 0, float rough = 0.5f,
                   float ior = 1.5f) -> GPUSphere {
    return {c, r, col, emit, mat, rough, ior, 0.0f};
  };

  spheres.push_back(sphere({0, -1000, 0}, 1000.f, {0.45f, 0.45f, 0.45f}));

  spheres.push_back(sphere({0, 12, 0}, 3.0f, {1.0f, 0.92f, 0.8f}, 15.0f));

  spheres.push_back(sphere({12, 5, 0}, 7.5f, {0.8f, 0.15f, 0.15f}));
  spheres.push_back(sphere({-12, 5, 0}, 7.5f, {0.15f, 0.8f, 0.15f}));
  spheres.push_back(sphere({0, 5, -12}, 7.5f, {0.8f, 0.8f, 0.8f}));

  spheres.push_back(sphere({0, 1.5f, 0}, 1.5f, {1, 1, 1}, 0, 2, 0.0f, 1.5f));

  spheres.push_back(
      sphere({3.5f, 1.2f, 1.5f}, 1.2f, {0.9f, 0.8f, 0.7f}, 0, 1, 0.05f));

  spheres.push_back(
      sphere({-3.5f, 1.0f, 1.5f}, 1.0f, {0.7f, 0.3f, 0.8f}, 0, 1, 0.4f));

  spheres.push_back(sphere({-2.0f, 0.5f, -1.5f}, 0.5f, {0.9f, 0.6f, 0.2f}));
  spheres.push_back(sphere({2.0f, 0.5f, -2.0f}, 0.5f, {0.2f, 0.5f, 0.9f}));
  spheres.push_back(sphere({1.0f, 0.4f, 2.5f}, 0.4f, {0.9f, 0.9f, 0.3f}));
  spheres.push_back(sphere({-1.5f, 0.4f, 2.0f}, 0.4f, {0.4f, 0.8f, 0.6f}));
  spheres.push_back(sphere({5.0f, 0.6f, -1.0f}, 0.6f, {0.8f, 0.4f, 0.2f}));
  spheres.push_back(sphere({-4.5f, 0.7f, -0.5f}, 0.7f, {0.3f, 0.3f, 0.9f}));

  spheres.push_back(
      sphere({-1.0f, 0.3f, 3.5f}, 0.3f, {1.0f, 0.5f, 0.1f}, 8.0f));

  bvh.build(spheres);
  LOG_INFO(std::format("Scene: [{}] spheres, [{}] BVH nodes", spheres.size(),
                       bvh.nodes.size()));
}
*/

static void map_copy(VkDevice dev, VkDeviceMemory mem, const void *src,
                     VkDeviceSize sz) {
  void *ptr;
  vkMapMemory(dev, mem, 0, sz, 0, &ptr);
  memcpy(ptr, src, sz);
  vkUnmapMemory(dev, mem);
}

void VulkanApp::upload_scene() {
  LOG_INFO("Upload des sommets, indices et materiaux vers la VRAM...");

  constexpr auto FLAGS = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkDeviceSize vertex_sz = vertices.size() * sizeof(Vertex);
  create_buffer(vertex_sz,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    FLAGS, vertex_buffer, vertex_mem);
  { void* p; vkMapMemory(device,vertex_mem,0,vertex_sz,0,&p); memcpy(p,vertices.data(),vertex_sz); vkUnmapMemory(device,vertex_mem); }

  VkDeviceSize index_sz = indices.size() * sizeof(uint32_t);
  create_buffer(index_sz,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    FLAGS, index_buffer, index_mem);
  { void* p; vkMapMemory(device,index_mem,0,index_sz,0,&p); memcpy(p,indices.data(),index_sz); vkUnmapMemory(device,index_mem); }

  VkDeviceSize cam_sz = sizeof(Camera);
  create_buffer(cam_sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, FLAGS, camera_buf, camera_mem);
  vkMapMemory(device, camera_mem, 0, cam_sz, 0, &camera_map);

  VkDeviceSize mat_sz = gpu_materials.size() * sizeof(GPUMaterial);
  create_buffer(mat_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, FLAGS, material_buffer, material_mem);
  { void* p; vkMapMemory(device,material_mem,0,mat_sz,0,&p); memcpy(p,gpu_materials.data(),mat_sz); vkUnmapMemory(device,material_mem); }

  VkDeviceSize tri_sz = tri_material_ids.size() * sizeof(uint32_t);
  create_buffer(tri_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, FLAGS, tri_mat_id_buffer, tri_mat_id_mem);
  { void* p; vkMapMemory(device,tri_mat_id_mem,0,tri_sz,0,&p); memcpy(p,tri_material_ids.data(),tri_sz); vkUnmapMemory(device,tri_mat_id_mem); }

  LOG_INFO("Upload VRAM termine !");
}
void VulkanApp::create_compute_pipeline() {
  const uint32_t tex_count = (uint32_t)model_textures.size();

  std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[3] = {3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // materials
  bindings[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // tri_mat_ids
  bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex_count,  VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // textures[]

  std::array<VkDescriptorBindingFlags, 9> binding_flags{};
  binding_flags[8] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
                   | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

  VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
  flags_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  flags_ci.bindingCount  = (uint32_t)binding_flags.size();
  flags_ci.pBindingFlags = binding_flags.data();

  VkDescriptorSetLayoutCreateInfo dsli{};
  dsli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsli.pNext        = &flags_ci;
  dsli.bindingCount = (uint32_t)bindings.size();
  dsli.pBindings    = bindings.data();
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &comp_dsl));

  std::array<VkDescriptorPoolSize, 5> pool_sizes{};
  pool_sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              2};
  pool_sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1};
  pool_sizes[2] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
  pool_sizes[3] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     tex_count};
  pool_sizes[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             4};

  VkDescriptorPoolCreateInfo pi{};
  pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pi.maxSets       = 1;
  pi.poolSizeCount = (uint32_t)pool_sizes.size();
  pi.pPoolSizes    = pool_sizes.data();
  VK_CHECK(vkCreateDescriptorPool(device, &pi, nullptr, &comp_pool));

  VkDescriptorSetVariableDescriptorCountAllocateInfo var_alloc{};
  var_alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
  var_alloc.descriptorSetCount = 1;
  var_alloc.pDescriptorCounts  = &tex_count;

  VkDescriptorSetAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.pNext              = &var_alloc;
  ai.descriptorPool     = comp_pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts        = &comp_dsl;
  VK_CHECK(vkAllocateDescriptorSets(device, &ai, &comp_set));

  VkDescriptorImageInfo  accum_ii{nullptr, accum_view,        VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo  disp_ii {nullptr, display_view,      VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo cam_bi  {camera_buf,        0,       sizeof(Camera)};
  VkDescriptorBufferInfo v_bi    {vertex_buffer,     0,       VK_WHOLE_SIZE};
  VkDescriptorBufferInfo i_bi    {index_buffer,      0,       VK_WHOLE_SIZE};
  VkDescriptorBufferInfo mat_bi  {material_buffer,   0,       VK_WHOLE_SIZE};
  VkDescriptorBufferInfo tri_bi  {tri_mat_id_buffer, 0,       VK_WHOLE_SIZE};

  VkWriteDescriptorSetAccelerationStructureKHR as_info{};
  as_info.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  as_info.accelerationStructureCount = 1;
  as_info.pAccelerationStructures    = &tlas;

  std::vector<VkDescriptorImageInfo> tex_infos(tex_count);
  for (uint32_t t = 0; t < tex_count; t++)
    tex_infos[t] = {texture_sampler, model_textures[t].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

  std::array<VkWriteDescriptorSet, 9> writes{};
  for (auto& w : writes) {
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = comp_set;
    w.descriptorCount = 1;
  }
  writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;              writes[0].pImageInfo  = &accum_ii;
  writes[1].dstBinding = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;              writes[1].pImageInfo  = &disp_ii;
  writes[2].dstBinding = 2; writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;             writes[2].pBufferInfo = &cam_bi;
  writes[3].dstBinding = 3; writes[3].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; writes[3].pNext       = &as_info;
  writes[4].dstBinding = 4; writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             writes[4].pBufferInfo = &v_bi;
  writes[5].dstBinding = 5; writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             writes[5].pBufferInfo = &i_bi;
  writes[6].dstBinding = 6; writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             writes[6].pBufferInfo = &mat_bi;
  writes[7].dstBinding = 7; writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             writes[7].pBufferInfo = &tri_bi;
  writes[8].dstBinding = 8; writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;     writes[8].pImageInfo  = tex_infos.data(); writes[8].descriptorCount = tex_count;
  vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);

  VkPipelineLayoutCreateInfo pli{};
  pli.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts    = &comp_dsl;
  VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &comp_layout));

  VkShaderModule cs = load_shader(SHADER_DIR "raytracer.comp.spv");
  VkComputePipelineCreateInfo cpci{};
  cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage  = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs, "main", nullptr};
  cpci.layout = comp_layout;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &comp_pipeline));
  vkDestroyShaderModule(device, cs, nullptr);
}
void VulkanApp::create_display_pipeline() {
  // Render pass
  VkAttachmentDescription att{};
  att.format = sc_format;
  att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub{};
  sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  sub.colorAttachmentCount = 1;
  sub.pColorAttachments = &ref;

  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass = 0;
  dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = 0;
  dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo rpci{};
  rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rpci.attachmentCount = 1;
  rpci.pAttachments = &att;
  rpci.subpassCount = 1;
  rpci.pSubpasses = &sub;
  rpci.dependencyCount = 1;
  rpci.pDependencies = &dep;
  VK_CHECK(vkCreateRenderPass(device, &rpci, nullptr, &disp_pass));

  // Framebuffers
  disp_fbs.resize(sc_images.size());
  for (uint32_t i = 0; i < sc_images.size(); i++) {
    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = disp_pass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &sc_views[i];
    fbci.width = sc_extent.width;
    fbci.height = sc_extent.height;
    fbci.layers = 1;
    VK_CHECK(vkCreateFramebuffer(device, &fbci, nullptr, &disp_fbs[i]));
  }

  VkDescriptorSetLayoutBinding b{};
  b.binding = 0;
  b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  b.descriptorCount = 1;
  b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo dsli{};
  dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsli.bindingCount = 1;
  dsli.pBindings = &b;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &disp_dsl));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo pi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &ps};
  VK_CHECK(vkCreateDescriptorPool(device, &pi, nullptr, &disp_pool));

  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                 nullptr, disp_pool, 1, &disp_dsl};
  VK_CHECK(vkAllocateDescriptorSets(device, &ai, &disp_set));

  VkDescriptorImageInfo dii{display_sampler, display_view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{};
  wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wr.dstSet = disp_set;
  wr.dstBinding = 0;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(device, 1, &wr, 0, nullptr);

  // Pipeline layout
  VkPipelineLayoutCreateInfo pli{};
  pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &disp_dsl;
  VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &disp_layout));

  // Shaders
  VkShaderModule vs = load_shader(SHADER_DIR "display.vert.spv");
  VkShaderModule fs = load_shader(SHADER_DIR "display.frag.spv");

  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               nullptr,
               0,
               VK_SHADER_STAGE_VERTEX_BIT,
               vs,
               "main",
               nullptr};
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               nullptr,
               0,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               fs,
               "main",
               nullptr};

  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
  VkRect2D sc_rect{{0, 0}, {W, H}};
  VkPipelineViewportStateCreateInfo vs_ci{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vs_ci.viewportCount = 1;
  vs_ci.pViewports = &vp;
  vs_ci.scissorCount = 1;
  vs_ci.pScissors = &sc_rect;

  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blend{};
  blend.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &blend;

  VkGraphicsPipelineCreateInfo gpci{};
  gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  gpci.stageCount = 2;
  gpci.pStages = stages.data();
  gpci.pVertexInputState = &vi;
  gpci.pInputAssemblyState = &ia;
  gpci.pViewportState = &vs_ci;
  gpci.pRasterizationState = &rs;
  gpci.pMultisampleState = &ms;
  gpci.pColorBlendState = &cb;
  gpci.layout = disp_layout;
  gpci.renderPass = disp_pass;
  VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr,
                                     &disp_pipeline));

  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);
}

// Commands & sync
void VulkanApp::create_commands_and_sync() {
  cmds.resize(MAX_FRAMES);
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = command_pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = MAX_FRAMES;
  VK_CHECK(vkAllocateCommandBuffers(device, &ai, cmds.data()));

  img_available.resize(MAX_FRAMES);
  render_done.resize(MAX_FRAMES);
  fences.resize(MAX_FRAMES);
  VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (int i = 0; i < MAX_FRAMES; i++) {
    VK_CHECK(vkCreateSemaphore(device, &semi, nullptr, &img_available[i]));
    VK_CHECK(vkCreateSemaphore(device, &semi, nullptr, &render_done[i]));
    VK_CHECK(vkCreateFence(device, &fi, nullptr, &fences[i]));
  }
}

// Camera input + UBO update
bool VulkanApp::process_input(float dt) {
  bool moved = false;

  glm::vec3 forward = glm::normalize(
      glm::vec3(std::sin(cam_yaw) * std::cos(cam_pitch), std::sin(cam_pitch),
                -std::cos(cam_yaw) * std::cos(cam_pitch)));
  glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));

  float speed = cam_speed * dt;
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    speed *= 4.0f;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    cam_pos += forward * speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    cam_pos -= forward * speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    cam_pos -= right * speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    cam_pos += right * speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    cam_pos.y += speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
    cam_pos.y -= speed;
    moved = true;
  }
  if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
    if (cam_fov > 10) {
      cam_fov--;
      moved = true;
    }
  }
  if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
    if (cam_fov < 120) {
      cam_fov++;
      moved = true;
    }
  }

  // Mouse look
  double mx, my;
  glfwGetCursorPos(window, &mx, &my);
  if (!first_mouse) {
    float dx = (float)(mx - last_mx) * 0.002f;
    float dy = (float)(my - last_my) * 0.002f;
    if (dx != 0.0f || dy != 0.0f) {
      cam_yaw += dx;
      cam_pitch -= dy;
      cam_pitch = glm::clamp(cam_pitch, -1.48f, 1.48f);
      moved = true;
    }
  }
  first_mouse = false;
  last_mx = mx;
  last_my = my;

  return moved;
}

void VulkanApp::update_camera_ubo() {
  frame_index++;

  glm::vec3 forward = glm::normalize(
      glm::vec3(std::sin(cam_yaw) * std::cos(cam_pitch), std::sin(cam_pitch),
                -std::cos(cam_yaw) * std::cos(cam_pitch)));
  glm::vec3 world_up = glm::vec3(0, 1, 0);
  glm::vec3 right = glm::normalize(glm::cross(forward, world_up));
  glm::vec3 up = glm::cross(right, forward);

  float VFOV = cam_fov * 3.14159265f / 180.0f;
  constexpr float ASPECT = float(W) / float(H);
  float h = std::tan(VFOV * 0.5f);

  glm::vec3 horizontal = 2.0f * ASPECT * h * right;
  glm::vec3 vertical = 2.0f * h * up;

  Camera cam{};
  cam.origin     = glm::vec4(cam_pos, 1.0f);
  cam.horizontal = glm::vec4(horizontal, 0.0f);
  cam.vertical   = glm::vec4(vertical, 0.0f);
  cam.lower_left = glm::vec4(cam_pos + forward - horizontal * 0.5f - vertical * 0.5f, 1.0f);
  
  cam.dimensions = glm::uvec4(W, H, frame_index, 0);

  memcpy(camera_map, &cam, sizeof(Camera));
}

// Draw frame
void VulkanApp::draw_frame(uint32_t flight) {
  vkWaitForFences(device, 1, &fences[flight], VK_TRUE, UINT64_MAX);
  vkResetFences(device, 1, &fences[flight]);

  uint32_t img_idx = 0;
  VK_CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, img_available[flight], VK_NULL_HANDLE, &img_idx));

  VkCommandBuffer cmd = cmds[flight];
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

  // Compute Shader
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_layout, 0, 1, &comp_set, 0, nullptr);
  vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);

  // Memory barrier: compute writes done before fragment reads
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.image = display_image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  // Display render pass
  VkRenderPassBeginInfo rpbi{};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = disp_pass;
  rpbi.framebuffer = disp_fbs[img_idx];
  rpbi.renderArea = {{0, 0}, sc_extent};
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, disp_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, disp_layout, 0, 1, &disp_set, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);

  // Transition display_image back to GENERAL for next compute pass
  barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &img_available[flight];
  si.pWaitDstStageMask = &wait_stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &render_done[flight];
  VK_CHECK(vkQueueSubmit(gfx_queue, 1, &si, fences[flight]));

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &render_done[flight];
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain;
  present.pImageIndices = &img_idx;
  vkQueuePresentKHR(gfx_queue, &present);
}

// Vulkan utilities
uint32_t VulkanApp::find_memory(uint32_t mask, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((mask & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  throw std::runtime_error("No suitable memory type");
}

void VulkanApp::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer &buf,
                              VkDeviceMemory &mem) {
  VkBufferCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &ci, nullptr, &buf));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device, buf, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory(req.memoryTypeBits, props);

  VkMemoryAllocateFlagsInfo flagsInfo{};
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
      flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
      flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
      ai.pNext = &flagsInfo;
  }

  VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
  VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
}

void VulkanApp::create_image2d(uint32_t w, uint32_t h, VkFormat fmt,
                               VkImageUsageFlags usage, VkImage &img,
                               VkDeviceMemory &mem) {
  VkImageCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = fmt;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(device, &ci, nullptr, &img));

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(device, img, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex =
      find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
  VK_CHECK(vkBindImageMemory(device, img, mem, 0));
}

VkImageView VulkanApp::create_image_view(VkImage img, VkFormat fmt) {
  VkImageViewCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ci.image = img;
  ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ci.format = fmt;
  ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view;
  VK_CHECK(vkCreateImageView(device, &ci, nullptr, &view));
  return view;
}

void VulkanApp::transition_image(VkCommandBuffer cmd, VkImage img,
                                 VkImageLayout from, VkImageLayout to) {
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

VkCommandBuffer VulkanApp::begin_one_shot() {
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = command_pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  return cmd;
}

void VulkanApp::end_one_shot(VkCommandBuffer cmd) {
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(gfx_queue, 1, &si, VK_NULL_HANDLE));
  vkQueueWaitIdle(gfx_queue);
  vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

VkShaderModule VulkanApp::load_shader(const std::string &path) {
  auto code = read_file(path);
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule m;
  VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &m));
  return m;
}

std::vector<char> VulkanApp::read_file(const std::string &path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Cannot open: " + path);
  size_t sz = f.tellg();
  std::vector<char> buf(sz);
  f.seekg(0);
  f.read(buf.data(), sz);
  return buf;
}

void VulkanApp::cleanup() {
  // 1. On attend que le GPU ait fini de travailler
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);
  }

  if (texture_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, texture_sampler, nullptr);
    }
    for (auto& tex : model_textures) {
        if (tex.view != VK_NULL_HANDLE) vkDestroyImageView(device, tex.view, nullptr);
        if (tex.image != VK_NULL_HANDLE) vkDestroyImage(device, tex.image, nullptr);
        if (tex.mem != VK_NULL_HANDLE) vkFreeMemory(device, tex.mem, nullptr);
    }
    model_textures.clear();

  // 2. On vérifie la taille des tableaux avant de boucler !
  for (size_t i = 0; i < img_available.size(); i++) {
    if (img_available[i]) vkDestroySemaphore(device, img_available[i], nullptr);
  }
  for (size_t i = 0; i < render_done.size(); i++) {
    if (render_done[i]) vkDestroySemaphore(device, render_done[i], nullptr);
  }
  for (size_t i = 0; i < fences.size(); i++) {
    if (fences[i]) vkDestroyFence(device, fences[i], nullptr);
  }

  if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);

  if (comp_pipeline) vkDestroyPipeline(device, comp_pipeline, nullptr);
  if (comp_layout) vkDestroyPipelineLayout(device, comp_layout, nullptr);
  if (comp_pool) vkDestroyDescriptorPool(device, comp_pool, nullptr);
  if (comp_dsl) vkDestroyDescriptorSetLayout(device, comp_dsl, nullptr);

  if (disp_pipeline) vkDestroyPipeline(device, disp_pipeline, nullptr);
  if (disp_layout) vkDestroyPipelineLayout(device, disp_layout, nullptr);
  for (auto fb : disp_fbs) {
    if (fb) vkDestroyFramebuffer(device, fb, nullptr);
  }
  if (disp_pass) vkDestroyRenderPass(device, disp_pass, nullptr);
  if (disp_pool) vkDestroyDescriptorPool(device, disp_pool, nullptr);
  if (disp_dsl) vkDestroyDescriptorSetLayout(device, disp_dsl, nullptr);

  // Plus de vkUnmapMemory sauvage ici !
  if (camera_buf) {
    vkDestroyBuffer(device, camera_buf, nullptr);
    vkUnmapMemory(device, camera_mem);
  }
  if (camera_mem) vkFreeMemory(device, camera_mem, nullptr);

  // Destroy Tlas
  if (tlas) {
    auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    if (vkDestroyAccelerationStructureKHR) vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
  }
  if (tlas_buffer) vkDestroyBuffer(device, tlas_buffer, nullptr);
  if (tlas_mem) vkFreeMemory(device, tlas_mem, nullptr);
  
  if (instance_buffer) vkDestroyBuffer(device, instance_buffer, nullptr);
  if (instance_mem) vkFreeMemory(device, instance_mem, nullptr);

  // Destroy Blas
  if (blas) {
    auto vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    if (vkDestroyAccelerationStructureKHR) vkDestroyAccelerationStructureKHR(device, blas, nullptr);
  }
  if (blas_buffer) vkDestroyBuffer(device, blas_buffer, nullptr);
  if (blas_mem) vkFreeMemory(device, blas_mem, nullptr);

  if (tri_mat_id_buffer) vkDestroyBuffer(device, tri_mat_id_buffer, nullptr);
  if (tri_mat_id_mem)    vkFreeMemory(device, tri_mat_id_mem, nullptr);
  if (material_buffer)   vkDestroyBuffer(device, material_buffer, nullptr);
  if (material_mem)      vkFreeMemory(device, material_mem, nullptr);
  if (vertex_buffer) vkDestroyBuffer(device, vertex_buffer, nullptr);
  if (vertex_mem) vkFreeMemory(device, vertex_mem, nullptr);
  if (index_buffer) vkDestroyBuffer(device, index_buffer, nullptr);
  if (index_mem) vkFreeMemory(device, index_mem, nullptr);

  if (display_sampler) vkDestroySampler(device, display_sampler, nullptr);
  if (accum_view) vkDestroyImageView(device, accum_view, nullptr);
  if (accum_image) vkDestroyImage(device, accum_image, nullptr);
  if (accum_mem) vkFreeMemory(device, accum_mem, nullptr);
  if (display_view) vkDestroyImageView(device, display_view, nullptr);
  if (display_image) vkDestroyImage(device, display_image, nullptr);
  if (display_mem) vkFreeMemory(device, display_mem, nullptr);

  for (auto v : sc_views) {
    if (v) vkDestroyImageView(device, v, nullptr);
  }
  if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
  if (device) vkDestroyDevice(device, nullptr);
  if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
  
  if (debug_messenger) {
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (fn) fn(instance, debug_messenger, nullptr);
  }
  
  if (instance) vkDestroyInstance(instance, nullptr);
  if (window) glfwDestroyWindow(window);
  glfwTerminate();
}

/*Load glb with gltf lib*/
void VulkanApp::load_gltf(const std::string& filepath) {
  tinygltf::Model    model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;

  LOG_INFO(std::string("Load 3d glb: ") + filepath);
  bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
  if (!warn.empty()) LOG_WARN("GLTF Warn: " + warn);
  if (!err.empty())  LOG_ERROR("GLTF Err: " + err);
  if (!ret) throw std::runtime_error("impossible to load GLB");

  // Charger toutes les images
  for (const auto& img : model.images)
    create_texture_from_image(img);

  if (texture_sampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkCreateSampler(device, &sci, nullptr, &texture_sampler);
  }

  // Construire la table des materiaux GPU
  gpu_materials.clear();
  gpu_materials.reserve(model.materials.size());
  for (const auto& mat : model.materials) {
    GPUMaterial gm{};

    // Base color factor (toujours present)
    const auto& bcf = mat.pbrMetallicRoughness.baseColorFactor;
    gm.base_r = (float)bcf[0];
    gm.base_g = (float)bcf[1];
    gm.base_b = (float)bcf[2];
    gm.base_a = (float)bcf[3];

    // Metallic / roughness factors
    gm.metallic  = (float)mat.pbrMetallicRoughness.metallicFactor;
    gm.roughness = (float)mat.pbrMetallicRoughness.roughnessFactor;

    // Emissive
    const auto& ef = mat.emissiveFactor;
    float emax = (float)std::max({ef[0], ef[1], ef[2]});
    gm.emissive = emax;

    // Base color texture (optionnel)
    int base_idx = mat.pbrMetallicRoughness.baseColorTexture.index;
    if (base_idx >= 0) {
      int src = model.textures[base_idx].source;
      if (src >= 0) gm.base_tex = (uint32_t)src;
    }

    // Metallic/roughness texture (optionnel)
    int mr_idx = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
    if (mr_idx >= 0) {
      int src = model.textures[mr_idx].source;
      if (src >= 0) gm.mr_tex = (uint32_t)src;
    }

    gpu_materials.push_back(gm);
  }
  if (gpu_materials.empty())
    gpu_materials.push_back(GPUMaterial{});

  LOG_INFO(std::format("Materiaux GPU construits : [{}] / Images dispo : [{}]", gpu_materials.size(), model_textures.size()));
  for (size_t mi = 0; mi < gpu_materials.size(); mi++) {
    const auto& gm = gpu_materials[mi];
    LOG_INFO(std::format("  Mat[{}] '{}': color=[{:.2f},{:.2f},{:.2f}] metal={:.2f} rough={:.2f} base_tex={} mr_tex={}",
      mi, model.materials[mi].name,
      gm.base_r, gm.base_g, gm.base_b,
      gm.metallic, gm.roughness,
      gm.base_tex == 0xFFFFFFFF ? -1 : (int)gm.base_tex,
      gm.mr_tex   == 0xFFFFFFFF ? -1 : (int)gm.mr_tex));
  }

  // Extraction de la geometrie
  vertices.clear();
  indices.clear();
  tri_material_ids.clear();

  for (const auto& mesh : model.meshes) {
    for (const auto& primitive : mesh.primitives) {

      uint32_t first_vertex    = (uint32_t)vertices.size();
      uint32_t tri_count_before = (uint32_t)(indices.size() / 3);

      uint32_t matID = 0;
      if (primitive.material >= 0 &&
          (uint32_t)primitive.material < (uint32_t)gpu_materials.size())
        matID = (uint32_t)primitive.material;

      // Indices
      if (primitive.indices >= 0) {
        const auto& acc  = model.accessors[primitive.indices];
        const auto& bv   = model.bufferViews[acc.bufferView];
        const uint8_t* p = model.buffers[bv.buffer].data.data()
                           + bv.byteOffset + acc.byteOffset;
        if (acc.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
          const uint32_t* buf = reinterpret_cast<const uint32_t*>(p);
          for (size_t i = 0; i < acc.count; i++) indices.push_back(buf[i] + first_vertex);
        } else if (acc.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
          const uint16_t* buf = reinterpret_cast<const uint16_t*>(p);
          for (size_t i = 0; i < acc.count; i++) indices.push_back(buf[i] + first_vertex);
        }
      }

      uint32_t tri_count_after = (uint32_t)(indices.size() / 3);
      for (uint32_t t = tri_count_before; t < tri_count_after; t++)
        tri_material_ids.push_back(matID);

      // Sommets
      const float* posPtr  = nullptr;
      const float* normPtr = nullptr;
      const float* uvPtr   = nullptr;
      size_t vertexCount   = 0;
      size_t posStride = 0, normStride = 0, uvStride = 0;

      auto get_attr = [&](const std::string& name, const float*& ptr, size_t& stride) {
        auto it = primitive.attributes.find(name);
        if (it == primitive.attributes.end()) return;
        const auto& acc = model.accessors[it->second];
        const auto& bv  = model.bufferViews[acc.bufferView];
        ptr    = reinterpret_cast<const float*>(
                     model.buffers[bv.buffer].data.data()
                     + bv.byteOffset + acc.byteOffset);
        stride = acc.ByteStride(bv)
                     ? (acc.ByteStride(bv) / sizeof(float))
                     : (name == "TEXCOORD_0" ? 2u : 3u);
        vertexCount = acc.count;
      };

      get_attr("POSITION",   posPtr,  posStride);
      get_attr("NORMAL",     normPtr, normStride);
      get_attr("TEXCOORD_0", uvPtr,   uvStride);

      for (size_t i = 0; i < vertexCount; i++) {
        Vertex v{};
        v.pos    = posPtr  ? glm::vec3(posPtr [i*posStride],  posPtr [i*posStride+1],  posPtr [i*posStride+2])  : glm::vec3(0.f);
        v.normal = normPtr ? glm::vec3(normPtr[i*normStride], normPtr[i*normStride+1], normPtr[i*normStride+2]) : glm::vec3(0,1,0);
        v.uv     = uvPtr   ? glm::vec2(uvPtr  [i*uvStride],   uvPtr  [i*uvStride+1])                           : glm::vec2(0.f);
        vertices.push_back(v);
      }
    }
  }

  if (!vertices.empty()) {
    glm::vec3 mn = vertices[0].pos, mx = vertices[0].pos;
    for (const auto& v : vertices) { mn = glm::min(mn,v.pos); mx = glm::max(mx,v.pos); }
    glm::vec3 c = (mn+mx)*0.5f, s = mx-mn;
    LOG_INFO(std::format("Centre [{:.2f},{:.2f},{:.2f}]  Taille [{:.2f},{:.2f},{:.2f}]", c.x,c.y,c.z, s.x,s.y,s.z));
  }
  LOG_INFO(std::format("Extraction : [{}] sommets, [{}] indices, [{}] triangles",
           vertices.size(), indices.size(), tri_material_ids.size()));
}
void VulkanApp::create_blas() {
  LOG_INFO("Build BLAS (Hardware BVH)");

  auto vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
  auto vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
  auto vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");

  VkBufferDeviceAddressInfo v_bda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, vertex_buffer};
  VkDeviceAddress vertex_addr = vkGetBufferDeviceAddress(device, &v_bda);

  VkBufferDeviceAddressInfo i_bda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, index_buffer};
  VkDeviceAddress index_addr = vkGetBufferDeviceAddress(device, &i_bda);

  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // x, y, z
  geom.geometry.triangles.vertexData.deviceAddress = vertex_addr;
  geom.geometry.triangles.vertexStride = sizeof(Vertex);
  geom.geometry.triangles.maxVertex = (uint32_t)(vertices.size() - 1);
  geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32; //uint32_t !
  geom.geometry.triangles.indexData.deviceAddress = index_addr;
  geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR; // Optimization no 

  VkAccelerationStructureBuildGeometryInfoKHR build_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build_info.geometryCount = 1;
  build_info.pGeometries = &geom;

  uint32_t num_triangles = (uint32_t)(indices.size() / 3);

  VkAccelerationStructureBuildSizesInfoKHR size_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &num_triangles, &size_info);

  create_buffer(size_info.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                blas_buffer, blas_mem);

  VkAccelerationStructureCreateInfoKHR as_ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  as_ci.buffer = blas_buffer;
  as_ci.size = size_info.accelerationStructureSize;
  as_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  VK_CHECK(vkCreateAccelerationStructureKHR(device, &as_ci, nullptr, &blas));

  VkBuffer scratch_buf;
  VkDeviceMemory scratch_mem;
  create_buffer(size_info.buildScratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                scratch_buf, scratch_mem);

  VkBufferDeviceAddressInfo s_bda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratch_buf};
  build_info.scratchData.deviceAddress = vkGetBufferDeviceAddress(device, &s_bda);
  build_info.dstAccelerationStructure = blas;

  VkCommandBuffer cmd = begin_one_shot();

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.primitiveCount = num_triangles;
  offset.primitiveOffset = 0;
  offset.firstVertex = 0;
  offset.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR* pOffset = &offset;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &pOffset);

  end_one_shot(cmd);

  vkDestroyBuffer(device, scratch_buf, nullptr);
  vkFreeMemory(device, scratch_mem, nullptr);

  LOG_INFO("Succes : BLAS compile materiellement ! Ta Dodge est prete.");
}

void VulkanApp::create_tlas() {
  LOG_INFO("Construction du TLAS (La scene) en cours...");

  auto vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
  auto vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
  auto vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
  auto vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");

  // 1. Récupérer l'adresse matérielle (GPU) de ton BLAS
  VkAccelerationStructureDeviceAddressInfoKHR blas_addr_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  blas_addr_info.accelerationStructure = blas;
  VkDeviceAddress blas_addr = vkGetAccelerationStructureDeviceAddressKHR(device, &blas_addr_info);

  // 2. Créer l'Instance de la voiture (Matrice + Lien vers le BLAS)
  VkAccelerationStructureInstanceKHR instance{};

  glm::mat4 model = glm::mat4(1.0f);

  model = glm::rotate(model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
  // La matrice de transformation (Ici : Matrice identité = Position 0,0,0 et pas de rotation)
  instance.transform = {
      model[0][0], model[1][0], model[2][0], model[3][0],
      model[0][1], model[1][1], model[2][1], model[3][1],
      model[0][2], model[1][2], model[2][2], model[3][2]
  };
  instance.instanceCustomIndex = 0; // Si on avait plusieurs objets, ça serait leur ID
  instance.mask = 0xFF; // Visible par tous les rayons (Lumière, Ombres, etc)
  instance.instanceShaderBindingTableRecordOffset = 0;
  instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  instance.accelerationStructureReference = blas_addr; // LE PONT : L'instance pointe vers la Dodge

  // 3. Envoyer cette instance dans un Buffer sur la VRAM
  VkDeviceSize inst_size = sizeof(VkAccelerationStructureInstanceKHR);
  create_buffer(inst_size,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                instance_buffer, instance_mem);

  void* data;
  vkMapMemory(device, instance_mem, 0, inst_size, 0, &data);
  memcpy(data, &instance, inst_size);
  vkUnmapMemory(device, instance_mem);

  VkBufferDeviceAddressInfo inst_addr_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, instance_buffer};
  VkDeviceAddress inst_addr = vkGetBufferDeviceAddress(device, &inst_addr_info);

  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geom.geometry.instances.arrayOfPointers = VK_FALSE;
  geom.geometry.instances.data.deviceAddress = inst_addr;

  VkAccelerationStructureBuildGeometryInfoKHR build_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build_info.geometryCount = 1;
  build_info.pGeometries = &geom;

  uint32_t count = 1; // On a 1 seule voiture dans la scène

  VkAccelerationStructureBuildSizesInfoKHR size_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &count, &size_info);

  create_buffer(size_info.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                tlas_buffer, tlas_mem);

  VkAccelerationStructureCreateInfoKHR as_ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  as_ci.buffer = tlas_buffer;
  as_ci.size = size_info.accelerationStructureSize;
  as_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  VK_CHECK(vkCreateAccelerationStructureKHR(device, &as_ci, nullptr, &tlas));

  VkBuffer scratch_buf;
  VkDeviceMemory scratch_mem;
  create_buffer(size_info.buildScratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                scratch_buf, scratch_mem);

  VkBufferDeviceAddressInfo s_bda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratch_buf};
  build_info.scratchData.deviceAddress = vkGetBufferDeviceAddress(device, &s_bda);
  build_info.dstAccelerationStructure = tlas;

  VkCommandBuffer cmd = begin_one_shot();
  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.primitiveCount = 1;
  const VkAccelerationStructureBuildRangeInfoKHR* pOffset = &offset;
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &pOffset);
  end_one_shot(cmd);

  vkDestroyBuffer(device, scratch_buf, nullptr);
  vkFreeMemory(device, scratch_mem, nullptr);

  LOG_INFO("Succes : TLAS compile ! L'Architecture Hardware est 100% prete.");
}

void VulkanApp::create_texture_from_image(const tinygltf::Image& img) {
    VkDeviceSize size = img.width * img.height * 4;
    
    // 1. Créer un buffer de staging pour uploader les pixels
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                  staging_buf, staging_mem);
    
    map_copy(device, staging_mem, img.image.data(), size);

    // 2. Créer l'image Vulkan
    Texture tex;
    create_image2d(img.width, img.height, VK_FORMAT_R8G8B8A8_UNORM, 
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 
                   tex.image, tex.mem);

    // 3. Copier le buffer vers l'image
    VkCommandBuffer cmd = begin_one_shot();
    
    // Transition vers TRANSFER_DST
    transition_image(cmd, tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {(uint32_t)img.width, (uint32_t)img.height, 1};
    vkCmdCopyBufferToImage(cmd, staging_buf, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    // Transition vers SHADER_READ
    transition_image(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    end_one_shot(cmd);

    vkDestroyBuffer(device, staging_buf, nullptr);
    vkFreeMemory(device, staging_mem, nullptr);

    tex.view = create_image_view(tex.image, VK_FORMAT_R8G8B8A8_UNORM);
    model_textures.push_back(tex);
}