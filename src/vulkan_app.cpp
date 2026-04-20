#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>

#include "vulkan_app.h"

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
  build_scene();
  upload_scene();
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
    std::cerr << "[Vulkan] Validation layers not available, disabling.\n";

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
      std::cout << "GPU: " << p.deviceName << "\n";
      return;
    }
  }
  phys_dev = devs[0];
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(phys_dev, &p);
  std::cout << "GPU: " << p.deviceName << "\n";
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

  std::vector<const char *> dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef __APPLE__
  dev_exts.push_back("VK_KHR_portability_subset");
#endif
  VkPhysicalDeviceFeatures feats{};
  feats.shaderInt64 = VK_FALSE;

  VkDeviceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
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
  ci.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
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
void VulkanApp::build_scene() {
  auto sphere = [](glm::vec3 c, float r, glm::vec3 col, float emit = 0,
                   int mat = 0, float rough = 0.5f,
                   float ior = 1.5f) -> GPUSphere {
    return {c, r, col, emit, mat, rough, ior, 0.0f};
  };

  spheres.push_back(sphere({0, -1000, 0}, 1000.f, {0.45f, 0.45f, 0.45f}));

  spheres.push_back(sphere({0, 12, 0}, 3.0f, {1.0f, 0.92f, 0.8f}, 15.0f));

  spheres.push_back(
      sphere({12, 5, 0}, 7.5f, {0.8f, 0.15f, 0.15f}));
  spheres.push_back(
      sphere({-12, 5, 0}, 7.5f, {0.15f, 0.8f, 0.15f}));
  spheres.push_back(
      sphere({0, 5, -12}, 7.5f, {0.8f, 0.8f, 0.8f}));

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
  std::cout << "Scene: " << spheres.size() << " spheres, " << bvh.nodes.size()
            << " BVH nodes\n";
}

static void map_copy(VkDevice dev, VkDeviceMemory mem, const void *src,
                     VkDeviceSize sz) {
  void *ptr;
  vkMapMemory(dev, mem, 0, sz, 0, &ptr);
  memcpy(ptr, src, sz);
  vkUnmapMemory(dev, mem);
}

void VulkanApp::upload_scene() {
  constexpr auto FLAGS = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkDeviceSize sphere_sz = spheres.size() * sizeof(GPUSphere);
  create_buffer(sphere_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, FLAGS,
                sphere_buf, sphere_mem);
  map_copy(device, sphere_mem, spheres.data(), sphere_sz);

  VkDeviceSize bvh_sz = bvh.nodes.size() * sizeof(BVHNode);
  create_buffer(bvh_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, FLAGS, bvh_buf,
                bvh_mem);
  map_copy(device, bvh_mem, bvh.nodes.data(), bvh_sz);

  VkDeviceSize prim_sz = bvh.prim_indices.size() * sizeof(int);
  create_buffer(prim_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, FLAGS, prim_buf,
                prim_mem);
  map_copy(device, prim_mem, bvh.prim_indices.data(), prim_sz);

  create_buffer(sizeof(Camera), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, FLAGS,
                camera_buf, camera_mem);
  vkMapMemory(device, camera_mem, 0, sizeof(Camera), 0, &camera_map);
}

// Compute pipeline
void VulkanApp::create_compute_pipeline() {
  std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
  // 0: accum image  (storage)
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // 1: display image (storage)
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // 2: camera UBO
  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // 3: spheres SSBO
  bindings[3].binding = 3;
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // 4: BVH nodes SSBO
  bindings[4].binding = 4;
  bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[4].descriptorCount = 1;
  bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // 5: prim indices SSBO
  bindings[5].binding = 5;
  bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[5].descriptorCount = 1;
  bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsli{};
  dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsli.bindingCount = (uint32_t)bindings.size();
  dsli.pBindings = bindings.data();
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &comp_dsl));

  std::array<VkDescriptorPoolSize, 3> pool_sizes{};
  pool_sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
  pool_sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
  pool_sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo pi{};
  pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pi.maxSets = 1;
  pi.poolSizeCount = (uint32_t)pool_sizes.size();
  pi.pPoolSizes = pool_sizes.data();
  VK_CHECK(vkCreateDescriptorPool(device, &pi, nullptr, &comp_pool));

  VkDescriptorSetAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = comp_pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &comp_dsl;
  VK_CHECK(vkAllocateDescriptorSets(device, &ai, &comp_set));

  // Write descriptors
  VkDescriptorImageInfo accum_ii{nullptr, accum_view, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo disp_ii{nullptr, display_view, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo cam_bi{camera_buf, 0, sizeof(Camera)};
  VkDescriptorBufferInfo sph_bi{sphere_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo bvh_bi{bvh_buf, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo pri_bi{prim_buf, 0, VK_WHOLE_SIZE};

  std::array<VkWriteDescriptorSet, 6> writes{};
  auto W_img = [&](int b, VkDescriptorImageInfo *ii) {
    writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[b].dstSet = comp_set;
    writes[b].dstBinding = (uint32_t)b;
    writes[b].descriptorCount = 1;
    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[b].pImageInfo = ii;
  };
  auto W_buf = [&](int b, VkDescriptorType type, VkDescriptorBufferInfo *bi) {
    writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[b].dstSet = comp_set;
    writes[b].dstBinding = (uint32_t)b;
    writes[b].descriptorCount = 1;
    writes[b].descriptorType = type;
    writes[b].pBufferInfo = bi;
  };
  W_img(0, &accum_ii);
  W_img(1, &disp_ii);
  W_buf(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &cam_bi);
  W_buf(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &sph_bi);
  W_buf(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bvh_bi);
  W_buf(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pri_bi);
  vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0,
                         nullptr);

  VkPipelineLayoutCreateInfo pli{};
  pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &comp_dsl;
  VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &comp_layout));

  VkShaderModule cs = load_shader(SHADER_DIR "raytracer.comp.spv");
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.layout = comp_layout;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = cs;
  cpci.stage.pName = "main";
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                    &comp_pipeline));
  vkDestroyShaderModule(device, cs, nullptr);
}

// Display pipeline
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
  cam.origin = cam_pos;
  cam.horizontal = horizontal;
  cam.vertical = vertical;
  cam.lower_left = cam_pos + forward - horizontal * 0.5f - vertical * 0.5f;
  cam.width = W;
  cam.height = H;
  cam.frame_count = frame_index;
  memcpy(camera_map, &cam, sizeof(Camera));
}

// Draw frame
void VulkanApp::draw_frame(uint32_t flight) {
  vkWaitForFences(device, 1, &fences[flight], VK_TRUE, UINT64_MAX);
  vkResetFences(device, 1, &fences[flight]);

  uint32_t img_idx = 0;
  VK_CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                 img_available[flight], VK_NULL_HANDLE,
                                 &img_idx));

  VkCommandBuffer cmd = cmds[flight];
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_layout, 0,
                          1, &comp_set, 0, nullptr);
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
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  // Display render pass
  VkRenderPassBeginInfo rpbi{};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = disp_pass;
  rpbi.framebuffer = disp_fbs[img_idx];
  rpbi.renderArea = {{0, 0}, sc_extent};
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, disp_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, disp_layout, 0,
                          1, &disp_set, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);

  // Transition display_image back to GENERAL for next compute pass
  barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &img_available[flight];
  si.pWaitDstStageMask = &wait_stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &render_done[flight];
  VK_CHECK(vkQueueSubmit(gfx_queue, 1, &si, fences[flight]));

  VkPresentInfoKHR present{};
  present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
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
  for (int i = 0; i < MAX_FRAMES; i++) {
    vkDestroySemaphore(device, img_available[i], nullptr);
    vkDestroySemaphore(device, render_done[i], nullptr);
    vkDestroyFence(device, fences[i], nullptr);
  }
  vkDestroyCommandPool(device, command_pool, nullptr);

  vkDestroyPipeline(device, comp_pipeline, nullptr);
  vkDestroyPipelineLayout(device, comp_layout, nullptr);
  vkDestroyDescriptorPool(device, comp_pool, nullptr);
  vkDestroyDescriptorSetLayout(device, comp_dsl, nullptr);

  vkDestroyPipeline(device, disp_pipeline, nullptr);
  vkDestroyPipelineLayout(device, disp_layout, nullptr);
  for (auto fb : disp_fbs)
    vkDestroyFramebuffer(device, fb, nullptr);
  vkDestroyRenderPass(device, disp_pass, nullptr);
  vkDestroyDescriptorPool(device, disp_pool, nullptr);
  vkDestroyDescriptorSetLayout(device, disp_dsl, nullptr);

  vkUnmapMemory(device, camera_mem);
  vkDestroyBuffer(device, camera_buf, nullptr);
  vkFreeMemory(device, camera_mem, nullptr);
  vkDestroyBuffer(device, sphere_buf, nullptr);
  vkFreeMemory(device, sphere_mem, nullptr);
  vkDestroyBuffer(device, bvh_buf, nullptr);
  vkFreeMemory(device, bvh_mem, nullptr);
  vkDestroyBuffer(device, prim_buf, nullptr);
  vkFreeMemory(device, prim_mem, nullptr);

  vkDestroySampler(device, display_sampler, nullptr);
  vkDestroyImageView(device, accum_view, nullptr);
  vkDestroyImage(device, accum_image, nullptr);
  vkFreeMemory(device, accum_mem, nullptr);
  vkDestroyImageView(device, display_view, nullptr);
  vkDestroyImage(device, display_image, nullptr);
  vkFreeMemory(device, display_mem, nullptr);

  for (auto v : sc_views)
    vkDestroyImageView(device, v, nullptr);
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  if (debug_messenger) {
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (fn)
      fn(instance, debug_messenger, nullptr);
  }
  vkDestroyInstance(instance, nullptr);
  glfwDestroyWindow(window);
  glfwTerminate();
}
