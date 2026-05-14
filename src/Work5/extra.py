import taichi as ti

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

# 画面分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
ior = ti.field(ti.f32, shape=())
msaa_sample_count = ti.field(ti.i32, shape=())

# 材质ID枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# ---------------------- 工具函数 ----------------------
@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """计算折射方向（修复版）"""
    out_refract_dir = ti.Vector([0.0, 0.0, 0.0])
    out_is_tir = False
    
    cos_theta_i = -I.dot(N)
    local_N = N
    local_eta = eta
    
    if cos_theta_i < 0.0:
        cos_theta_i = -cos_theta_i
        local_N = -N
        local_eta = 1.0 / eta
    
    sin_theta_i = ti.sqrt(ti.max(0.0, 1.0 - cos_theta_i * cos_theta_i))
    sin_theta_t = local_eta * sin_theta_i
    
    if sin_theta_t >= 1.0:
        out_is_tir = True
    else:
        cos_theta_t = ti.sqrt(ti.max(0.0, 1.0 - sin_theta_t * sin_theta_t))
        out_refract_dir = local_eta * I + (local_eta * cos_theta_i - cos_theta_t) * local_N
        out_refract_dir = normalize(out_refract_dir)
        out_is_tir = False
    
    return out_refract_dir, out_is_tir

@ti.func
def schlick_fresnel(cos_theta, ior):
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

# ---------------------- 场景求交 ----------------------
@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE # 明确类型

    # 1. 玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 1.0])
        hit_mat = MAT_GLASS

    # 2. 银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 棋盘格地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        hit_c = ti.Vector([0.3, 0.3, 0.3]) if (ix + iz) % 2 == 0 else ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

# ---------------------- 渲染核心（关键修复区） ----------------------
@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    sample_count = msaa_sample_count[None]
    glass_ior = ior[None]

    for i, j in pixels:
        final_pixel_color = ti.Vector([0.0, 0.0, 0.0])
        
        # MSAA多重采样
        for _ in range(sample_count):
            offset_x = ti.random() - 0.5
            offset_y = ti.random() - 0.5
            u = (i + offset_x - res_x / 2.0) / res_y * 2.0
            v = (j + offset_y - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            sample_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            # 迭代式光线追踪
            for bounce in range(max_bounces[None]):
                # 【修复1】：不用 _ 占位符，显式声明所有变量
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                # 未击中物体
                if t > 1e9:
                    sample_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                cos_theta = -rd.dot(N)

                # 分支1：镜面材质
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color

                # 分支2：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    # 阴影检测
                    shadow_ray_orig = p + N * 1e-4
                    # 【修复2】：这里也不能用 _，必须显式接住所有返回值
                    shadow_t, shadow_n, shadow_c, shadow_m = scene_intersect(shadow_ray_orig, L)
                    
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0
                    
                    # Phong光照
                    ambient = 0.2 * obj_color
                    direct_light = ambient
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    sample_color += throughput * direct_light
                    break

                # 分支3：玻璃材质
                elif mat_id == MAT_GLASS:
                    fresnel = schlick_fresnel(cos_theta, glass_ior)
                    refract_dir, is_tir = refract(rd, N, 1.0 / glass_ior)

                    if not is_tir:
                        if ti.random() < fresnel:
                            # 反射
                            ro = p + N * 1e-4
                            rd = normalize(reflect(rd, N))
                            throughput *= obj_color
                        else:
                            # 折射
                            ro = p - N * 1e-4
                            rd = refract_dir
                            throughput *= obj_color
                    else:
                        # 全内反射
                        ro = p + N * 1e-4
                        rd = normalize(reflect(rd, N))
                        throughput *= obj_color
            
            # 【修复3】：把累加放到采样循环里面
            final_pixel_color += sample_color

        # 采样平均
        pixels[i, j] = ti.math.clamp(final_pixel_color / sample_count, 0.0, 1.0)

# ---------------------- 主函数 ----------------------
def main():
    window = ti.ui.Window("Ray Tracing Demo (Fixed)", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化参数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    ior[None] = 1.5
    msaa_sample_count[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Base Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
        
        with gui.sub_window("Extra Controls", 0.75, 0.28, 0.23, 0.18):
            ior[None] = gui.slider_float('Glass IOR', ior[None], 1.0, 2.0)
            msaa_sample_count[None] = gui.slider_int('MSAA Samples', msaa_sample_count[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()