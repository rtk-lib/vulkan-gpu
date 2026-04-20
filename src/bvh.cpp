#include <algorithm>
#include <numeric>

#include "bvh.h"

void BVH::build(const std::vector<GPUSphere> &spheres) {
  nodes.clear();
  prim_indices.clear();

  if (spheres.empty()) {
    nodes.push_back({glm::vec3(0), -1, glm::vec3(0), -1, 0, 0, {0, 0}});
    return;
  }

  std::vector<int> work(spheres.size());
  std::iota(work.begin(), work.end(), 0);
  build_node(spheres, work, 0, (int)work.size());
}

int BVH::build_node(const std::vector<GPUSphere> &spheres,
                    std::vector<int> &work, int start, int end) {
  int idx = (int)nodes.size();
  nodes.push_back({});

  // Compute AABB over all primitives in range
  glm::vec3 bmin(1e30f);
  glm::vec3 bmax(-1e30f);
  for (int i = start; i < end; i++) {
    const auto &s = spheres[work[i]];
    bmin = glm::min(bmin, s.center - glm::vec3(s.radius));
    bmax = glm::max(bmax, s.center + glm::vec3(s.radius));
  }
  nodes[idx].bbox_min = bmin;
  nodes[idx].bbox_max = bmax;

  int count = end - start;

  // Leaf: 4 or fewer primitives
  if (count <= 4) {
    nodes[idx].prim_count = count;
    nodes[idx].first_prim = (int)prim_indices.size();
    nodes[idx].left_child = -1;
    nodes[idx].right_child = -1;
    for (int i = start; i < end; i++)
      prim_indices.push_back(work[i]);
    return idx;
  }

  // Split along longest axis, at centroid midpoint
  glm::vec3 extent = bmax - bmin;
  int axis = 0;
  if (extent.y > extent[axis])
    axis = 1;
  if (extent.z > extent[axis])
    axis = 2;
  float mid = 0.5f * (bmin[axis] + bmax[axis]);

  auto split_it =
      std::partition(work.begin() + start, work.begin() + end,
                     [&](int i) { return spheres[i].center[axis] < mid; });
  int split = (int)(split_it - work.begin());

  // Degenerate split fallback: split in half
  if (split == start || split == end)
    split = start + count / 2;

  nodes[idx].prim_count = 0;
  nodes[idx].first_prim = 0;

  int left = build_node(spheres, work, start, split);
  int right = build_node(spheres, work, split, end);
  nodes[idx].left_child = left;
  nodes[idx].right_child = right;

  return idx;
}
