/*
 *  Max memory bandwidth on intel i5 6600k = 34.1 GB/s
 */

#define IDX(x, y, c) (2 * ((x) + (STRIDE) * (y)) + (c))

__kernel void diffuse_bad(__global float * dest, __global float * src, float a)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int center_id_a = IDX(gid_x, gid_y, 0);
  int center_id_b = center_id_a + 1;

  int right_id_a = center_id_a + 2;
  int right_id_b = right_id_a + 1;
  int left_id_a = center_id_a - 2;
  int left_id_b = left_id_a + 1;
  int up_id_a = center_id_a - DOUBLE_STRIDE;
  int up_id_b = up_id_a + 1;
  int down_id_a = center_id_a + DOUBLE_STRIDE;
  int down_id_b = down_id_a + 1;

  float center_src_a = src[center_id_a];
  float center_src_b = src[center_id_b];

  dest[center_id_a] = center_src_a + a * (src[left_id_a] + src[right_id_a] + src[up_id_a] + src[down_id_a] - 4 * center_src_a);
  dest[center_id_b] = center_src_b + a * (src[left_id_b] + src[right_id_b] + src[up_id_b] + src[down_id_b] - 4 * center_src_b);
}

__kernel void diffuse(__global float * dest, __global float * src, float a, float denominator)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int center_id_a = IDX(gid_x, gid_y, 0);
  int center_id_b = center_id_a + 1;

  int right_id_a = center_id_a + 2;
  int right_id_b = right_id_a + 1;
  int left_id_a = center_id_a - 2;
  int left_id_b = left_id_a + 1;
  int up_id_a = center_id_a - DOUBLE_STRIDE;
  int up_id_b = up_id_a + 1;
  int down_id_a = center_id_a + DOUBLE_STRIDE;
  int down_id_b = down_id_a + 1;

  dest[center_id_a] = (src[center_id_a] + a * (dest[left_id_a] + dest[right_id_a] + dest[up_id_a] + dest[down_id_a])) * denominator;
  dest[center_id_b] = (src[center_id_b] + a * (dest[left_id_b] + dest[right_id_b] + dest[up_id_b] + dest[down_id_b])) * denominator;
}

__kernel void advect(__global float * dest, __global float * src, __global float * vel, float dt)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int idx_a = IDX(gid_x, gid_y, 0);
  int idx_b = idx_a + 1;

  const float clamp_max = SIM_SIZE + 0.5f;

  float x = clamp(gid_x + dt * vel[idx_a], 0.5f, clamp_max);
  float y = clamp(gid_y + dt * vel[idx_b], 0.5f, clamp_max);

  int left = (int)(x);
  int up = (int)(y);

  float s1 = x - left;
  float s0 = 1 - s1;
  float t1 = y - up;
  float t0 = 1 - t1;

  int upper_left_a = IDX(left, up, 0);
  int upper_left_b = upper_left_a + 1;
  int upper_right_a = upper_left_a + 2;
  int upper_right_b = upper_right_a + 1;
  int lower_left_a = upper_left_a + DOUBLE_STRIDE;
  int lower_left_b = lower_left_a + 1;
  int lower_right_a = lower_left_a + 2;
  int lower_right_b = lower_right_a + 1;

  dest[idx_a] = s0 * (t0 * src[upper_left_a] + t1 * src[lower_left_a]) + s1 * (t0 * src[upper_right_a] + t1 * src[lower_right_a]);
  dest[idx_b] = s0 * (t0 * src[upper_left_b] + t1 * src[lower_left_b]) + s1 * (t0 * src[upper_right_b] + t1 * src[lower_right_b]);
}

__kernel void project_A(__global float * tmp, __global float * vel, float h)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int center_id_a = IDX(gid_x, gid_y, 0);
  int center_id_b = center_id_a + 1;

  int right_id_a = center_id_a + 2;
  int left_id_a = center_id_a - 2;
  int up_id_b = center_id_b - DOUBLE_STRIDE;
  int down_id_b = center_id_b + DOUBLE_STRIDE;

  tmp[center_id_a] = h * (vel[left_id_a] - vel[right_id_a] + vel[up_id_b] - vel[down_id_b]);
  tmp[center_id_b] = 0;
}

__kernel void project_B(__global float * tmp)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int center_id_a = IDX(gid_x, gid_y, 0);
  int center_id_b = center_id_a + 1;

  int right_id_b = center_id_b + 2;
  int left_id_b = center_id_b - 2;
  int up_id_b = center_id_b - DOUBLE_STRIDE;
  int down_id_b = center_id_b + DOUBLE_STRIDE;

  tmp[center_id_b] = 0.25f * (tmp[center_id_a] + tmp[left_id_b] + tmp[right_id_b] + tmp[up_id_b] + tmp[down_id_b]);
}

__kernel void project_C(__global float * vel, __global float * tmp, float h)
{
  int gid_x = get_global_id(0) + 1;
  int gid_y = get_global_id(1) + 1;

  int center_id_a = IDX(gid_x, gid_y, 0);
  int center_id_b = center_id_a + 1;

  int right_id_b = center_id_b + 2;
  int left_id_b = center_id_b - 2;
  int up_id_b = center_id_b - DOUBLE_STRIDE;
  int down_id_b = center_id_b + DOUBLE_STRIDE;

  vel[center_id_a] += h * (tmp[left_id_b] - tmp[right_id_b]);
  vel[center_id_b] += h * (tmp[up_id_b] - tmp[down_id_b]);
}

__kernel void add_source(__global float * dest, __global float * src, float dt)
{
  int gid = get_global_id(0);
  dest[gid] += dt * src[gid];
}

__kernel void add_event_sources(__global float * dest, __constant int * x, __constant int * y, __constant float * strength, __constant int * max_radius_sqrd, int num_events, int vec_type)
{
  // should probably set full grid
  const int gid_x = get_global_id(0) + 1;
  const int gid_y = get_global_id(1) + 1;

  const int channel = !(vec_type == IS_A_DENSITY || vec_type == IS_U_VELOCITY);

  float result = 0;

  for (int i = 0; i < num_events; i++)
  {
    int delta_x = gid_x - x[i];
    int delta_y = gid_y - y[i];

    // The distance is never less than 1 so that the added source is always less than strength[i]
    int dist_sqrd = max((delta_x * delta_x) + (delta_y * delta_y), 1);

    if (dist_sqrd < max_radius_sqrd[i])
    {
      result += strength[i] * rsqrt((float)dist_sqrd);
    }
  }

  dest[IDX(gid_x, gid_y, channel)] += result;

  // maybe we should cap dest[idx] to MAX_DENSITY here
}

// TODO: clean up this function
__kernel void set_bnd(__global float * dest, int vec_type)
{
  const int gid = get_global_id(0) + 1;

  const float vel_sign = 1 - 2 * (vec_type == IS_VELOCITY);

  // 1 <= gid <= width
  // if we are on an edge
  if (gid <= SIM_SIZE)
  {
    // left edge
    dest[2 * (STRIDE * gid) + 0] = vel_sign * dest[2 * (1 + STRIDE * gid) + 0];
    dest[2 * (STRIDE * gid) + 1] = dest[2 * (1 + STRIDE * gid) + 1];
    // right edge
    dest[2 * (SIM_SIZE + 1 + STRIDE * gid) + 0] = vel_sign * dest[2 * (SIM_SIZE + STRIDE * gid) + 0];
    dest[2 * (SIM_SIZE + 1 + STRIDE * gid) + 1] = dest[2 * (SIM_SIZE + STRIDE * gid) + 1];
    // top edge
    dest[2 * (gid) + 0] = dest[2 * (gid + STRIDE) + 0];
    dest[2 * (gid) + 1] = vel_sign * dest[2 * (gid + STRIDE) + 1];
    // bottom edge
    dest[2 * (gid + STRIDE * (SIM_SIZE + 1)) + 0] = dest[2 * (gid + STRIDE * SIM_SIZE) + 0];
    dest[2 * (gid + STRIDE * (SIM_SIZE + 1)) + 1] = vel_sign * dest[2 * (gid + STRIDE * SIM_SIZE) + 1];
  }
  else  { // if we are a corner
    // top left
    dest[0] = 0.5f * (dest[2] + dest[DOUBLE_STRIDE]);
    dest[1] = 0.5f * (dest[3] + dest[DOUBLE_STRIDE + 1]);
    // bottom left
    dest[2 * (STRIDE * (SIM_SIZE + 1)) + 0] = 0.5f * (dest[2 * (1 + STRIDE * (SIM_SIZE + 1)) + 0] + dest[2 * (STRIDE * SIM_SIZE) + 0]);
    dest[2 * (STRIDE * (SIM_SIZE + 1)) + 1] = 0.5f * (dest[2 * (1 + STRIDE * (SIM_SIZE + 1)) + 1] + dest[2 * (STRIDE * SIM_SIZE) + 1]);
    // top right
    dest[2 * (SIM_SIZE + 1) + 0] = 0.5f * (dest[2 * SIM_SIZE + 0] + dest[2 * (SIM_SIZE + 1 + STRIDE) + 0]);
    dest[2 * (SIM_SIZE + 1) + 1] = 0.5f * (dest[2 * SIM_SIZE + 1] + dest[2 * (SIM_SIZE + 1 + STRIDE) + 1]);
    // bottom right
    dest[2 * (SIM_SIZE + 1 + STRIDE * (SIM_SIZE + 1)) + 0] = 0.5f * (dest[2 * (SIM_SIZE + STRIDE * (SIM_SIZE + 1)) + 0] + dest[2 * (SIM_SIZE + 1 + STRIDE * SIM_SIZE) + 0]);
    dest[2 * (SIM_SIZE + 1 + STRIDE * (SIM_SIZE + 1)) + 1] = 0.5f * (dest[2 * (SIM_SIZE + STRIDE * (SIM_SIZE + 1)) + 1] + dest[2 * (SIM_SIZE + 1 + STRIDE * SIM_SIZE) + 1]);
  }
}

__kernel void make_framebuffer(write_only image2d_t dest, __global float * src)
{
  // each channel should sum to no more than 1.f
  //const float3 first_color = (float3)(1.f, 0.54f, 0.f);
  //const float3 second_color = (float3)(0.f, 0.f, 1.f);
  //const float3 first_color = (float3)(0.f, 0.5f, 0.f);
  //const float3 second_color = (float3)(0.5f, 0.5f, 0.f);
  const float3 first_color = (float3)(0.f, 0.f, 1.f);
  const float3 second_color = (float3)(1.f, 0.f, 1.f);

  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int idx_a = IDX(gid_x + 1, gid_y + 1, 0);
  int idx_b = IDX(gid_x + 1, gid_y + 1, 1);

  float3 final_color = first_color * min(src[idx_a], 1.f) + second_color * min(src[idx_b], 1.f);

  write_imagef(dest, (int2)(gid_x, gid_y), (float4)(final_color, 1.f));
}
