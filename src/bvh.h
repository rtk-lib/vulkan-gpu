#pragma once
#include <glm/glm.hpp>
#include <vector>

struct GPUSphere {
  glm::vec3 center;
  float radius;
  glm::vec3 albedo;
  float emission;
  int mat_type;
  float roughness;
  float ior;
  float _pad;
}; // 48 bytes

struct BVHNode {
  glm::vec3 bbox_min;
  int left_child; // -1 if leaf
  glm::vec3 bbox_max;
  int right_child;
  int prim_count;  // 0 = internal node, >0 = leaf
  int first_prim;
  int _pad[2];
}; // 48 bytes

static_assert(sizeof(GPUSphere) == 48);
static_assert(sizeof(BVHNode) == 48);

class BVH {
public:
  std::vector<BVHNode> nodes;
  std::vector<int> prim_indices;

  void build(const std::vector<GPUSphere> &spheres);

private:
  int build_node(const std::vector<GPUSphere> &spheres, std::vector<int> &work,
                 int start, int end);
};
