#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace tinygltf { struct Image; struct Model; }

// Textures
struct Texture {
    VkImage       image = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkImageView   view  = VK_NULL_HANDLE;
};

// Triangle point in space
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

// Pour chaque matériau glTF — couleurs PBR + textures optionnelles
struct GPUMaterial {
    // vec4 aligné : base color RGBA (used when base_tex == 0xFFFFFFFF)
    float    base_r = 1.0f, base_g = 1.0f, base_b = 1.0f, base_a = 1.0f;
    // metallic, roughness, emissive strength, padding
    float    metallic = 0.0f, roughness = 0.5f, emissive = 0.0f, _pad = 0.0f;
    // texture indices (0xFFFFFFFF = no texture, use factor)
    uint32_t base_tex = 0xFFFFFFFF;
    uint32_t mr_tex   = 0xFFFFFFFF;
    uint32_t _pad0    = 0;
    uint32_t _pad1    = 0;
};

// GPU structs (must match shader std140/std430 layouts exactly)
struct Camera {
    glm::vec4  origin;
    glm::vec4  horizontal;
    glm::vec4  vertical;
    glm::vec4  lower_left;
    glm::uvec4 dimensions;
};

class VulkanApp {
public:
#ifdef __APPLE__
    static constexpr uint32_t W = 1280 * 3;
    static constexpr uint32_t H = 720  * 3;
#else
    static constexpr uint32_t W = 1920;
    static constexpr uint32_t H = 1080;
#endif
    static constexpr int MAX_FRAMES = 2;

    void run();

private:
    // Window
    GLFWwindow *window = nullptr;

    // Core Vulkan
    VkInstance               instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         phys_dev       = VK_NULL_HANDLE;
    VkDevice                 device         = VK_NULL_HANDLE;
    VkQueue                  gfx_queue      = VK_NULL_HANDLE;
    uint32_t                 gfx_family     = 0;

    // Swapchain
    VkSwapchainKHR           swapchain  = VK_NULL_HANDLE;
    VkFormat                 sc_format  = VK_FORMAT_UNDEFINED;
    VkExtent2D               sc_extent  = {};
    std::vector<VkImage>     sc_images;
    std::vector<VkImageView> sc_views;

    // Render images
    VkImage        accum_image    = VK_NULL_HANDLE;
    VkDeviceMemory accum_mem      = VK_NULL_HANDLE;
    VkImageView    accum_view     = VK_NULL_HANDLE;
    VkImage        display_image  = VK_NULL_HANDLE;
    VkDeviceMemory display_mem    = VK_NULL_HANDLE;
    VkImageView    display_view   = VK_NULL_HANDLE;
    VkSampler      display_sampler = VK_NULL_HANDLE;

    // Camera buffer
    VkBuffer       camera_buf  = VK_NULL_HANDLE;
    VkDeviceMemory camera_mem  = VK_NULL_HANDLE;
    void          *camera_map  = nullptr;

    // Geometry buffers
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    VkBuffer       vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory vertex_mem    = VK_NULL_HANDLE;
    VkBuffer       index_buffer  = VK_NULL_HANDLE;
    VkDeviceMemory index_mem     = VK_NULL_HANDLE;

    // Acceleration structures
    VkBuffer                    blas_buffer     = VK_NULL_HANDLE;
    VkDeviceMemory              blas_mem        = VK_NULL_HANDLE;
    VkAccelerationStructureKHR  blas            = VK_NULL_HANDLE;

    VkBuffer                    tlas_buffer     = VK_NULL_HANDLE;
    VkDeviceMemory              tlas_mem        = VK_NULL_HANDLE;
    VkAccelerationStructureKHR  tlas            = VK_NULL_HANDLE;
    VkBuffer                    instance_buffer = VK_NULL_HANDLE;
    VkDeviceMemory              instance_mem    = VK_NULL_HANDLE;

    // Textures du modèle (toutes les images glTF)
    std::vector<Texture>     model_textures;
    VkSampler                texture_sampler = VK_NULL_HANDLE;

    // ── NOUVEAU : matériaux et IDs par triangle ───────────────────────────────
    std::vector<GPUMaterial> gpu_materials;     // 1 entrée par matériau glTF
    std::vector<uint32_t>    tri_material_ids;  // 1 entrée par triangle (primID)

    VkBuffer       material_buffer    = VK_NULL_HANDLE;
    VkDeviceMemory material_mem       = VK_NULL_HANDLE;
    VkBuffer       tri_mat_id_buffer  = VK_NULL_HANDLE;
    VkDeviceMemory tri_mat_id_mem     = VK_NULL_HANDLE;
    // ─────────────────────────────────────────────────────────────────────────

    // Compute pipeline
    VkDescriptorSetLayout comp_dsl      = VK_NULL_HANDLE;
    VkDescriptorPool      comp_pool     = VK_NULL_HANDLE;
    VkDescriptorSet       comp_set      = VK_NULL_HANDLE;
    VkPipelineLayout      comp_layout   = VK_NULL_HANDLE;
    VkPipeline            comp_pipeline = VK_NULL_HANDLE;

    // Display pipeline
    VkRenderPass              disp_pass     = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> disp_fbs;
    VkDescriptorSetLayout     disp_dsl      = VK_NULL_HANDLE;
    VkDescriptorPool          disp_pool     = VK_NULL_HANDLE;
    VkDescriptorSet           disp_set      = VK_NULL_HANDLE;
    VkPipelineLayout          disp_layout   = VK_NULL_HANDLE;
    VkPipeline                disp_pipeline = VK_NULL_HANDLE;

    // Commands & sync
    VkCommandPool              command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmds;
    std::vector<VkSemaphore>   img_available;
    std::vector<VkSemaphore>   render_done;
    std::vector<VkFence>       fences;

    bool     validation_enabled = false;
    uint32_t frame_index        = 0;

    // Camera state
    glm::vec3 cam_pos   {0.0f, 1.0f, 6.0f};
    float     cam_yaw   = 0.0f;
    float     cam_pitch = -0.10f;
    float     cam_speed = 8.0f;
    float     cam_fov   = 40.0f;
    double    last_time = 0.0;
    double    last_mx   = 0.0;
    double    last_my   = 0.0;
    bool      first_mouse = true;

    // Init
    void init_window();
    void init_vulkan();
    void main_loop();
    void cleanup();

    void create_instance();
    void setup_debug_messenger();
    void create_surface();
    void pick_physical_device();
    void create_logical_device();
    void create_swapchain();
    void create_render_images();
    void upload_scene();
    void create_compute_pipeline();
    void create_display_pipeline();
    void create_commands_and_sync();

    void load_gltf(const std::string &filepath);
    void create_blas();
    void create_tlas();
    void create_texture_from_image(const tinygltf::Image &img);

    void draw_frame(uint32_t flight);
    bool process_input(float dt);
    void update_camera_ubo();

    // Vulkan utilities
    VkShaderModule load_shader(const std::string &path);
    uint32_t       find_memory(uint32_t mask, VkMemoryPropertyFlags props);
    void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props,
                       VkBuffer &buf, VkDeviceMemory &mem);
    void create_image2d(uint32_t w, uint32_t h, VkFormat fmt,
                        VkImageUsageFlags usage,
                        VkImage &img, VkDeviceMemory &mem);
    VkImageView    create_image_view(VkImage img, VkFormat fmt);
    void transition_image(VkCommandBuffer cmd, VkImage img,
                          VkImageLayout from, VkImageLayout to);
    VkCommandBuffer begin_one_shot();
    void            end_one_shot(VkCommandBuffer cmd);

    static std::vector<char> read_file(const std::string &path);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT *, void *);
};