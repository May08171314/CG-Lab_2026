import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ==================== 物理与网格参数 ====================
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 结构弹簧劲度系数 (Structural)
k_shear = 8000.0   # 剪切弹簧劲度系数 (Shear)
k_bend = 4000.0    # 弯曲弹簧劲度系数 (Bending)
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限

# ==================== 选做内容：球体碰撞参数 ====================
sphere_radius = 0.22
sphere_center = ti.Vector([0.0, 0.25, 0.0]) 
# 创建一个专门用于渲染的单个元素的 field
sphere_center_field = ti.Vector.field(3, dtype=float, shape=1)

# ==================== 定义 Taichi 数据场 ====================
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据场 (增加了剪切和弯曲弹簧，扩大容量)
max_springs = N * N * 12
spring_indices = ti.field(dtype=int, shape=max_springs * 2) 
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_stiffness = ti.field(dtype=float, shape=max_springs) # 新增：每根弹簧特有的劲度系数
num_springs = ti.field(dtype=int, shape=())

# ==================== 初始化 Kernel ====================

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # 将布料水平放置在球体上方
        x[idx] = ti.Vector([i * 0.045 - 0.425, 0.6, j * 0.045 - 0.425])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点，让布料自然下落挂在球上
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化三种类型的弹簧"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        
        # 1. 结构弹簧 (邻近 1 步：横向、纵向)
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_stiffness[c] = k_s
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_stiffness[c] = k_s

        # 2. 剪切弹簧 (对角线 1 步：抗扭曲)
        if i < N - 1 and j < N - 1:
            idx_diag1 = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag1])
            spring_lengths[c] = (x[idx] - x[idx_diag1]).norm()
            spring_stiffness[c] = k_shear
        if i > 0 and j < N - 1:
            idx_diag2 = (i - 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
            spring_stiffness[c] = k_shear

        # 3. 弯曲弹簧 (跨越 2 步：抗折叠)
        if i < N - 2:
            idx_bend_r = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_r])
            spring_lengths[c] = (x[idx] - x[idx_bend_r]).norm()
            spring_stiffness[c] = k_bend
        if j < N - 2:
            idx_bend_d = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_d])
            spring_lengths[c] = (x[idx] - x[idx_bend_d]).norm()
            spring_stiffness[c] = k_bend

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    sphere_center_field[0] = sphere_center  # 将球心坐标写入渲染 Field
    init_positions()
    init_springs()
    init_spring_indices()

# ==================== 物理计算库 (ti.func) ====================

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 三种弹簧力)"""
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            # 使用针对该弹簧类型的特有劲度系数
            f_spring = -spring_stiffness[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """选做内容：球体碰撞处理"""
    dir = pos[idx] - sphere_center
    dist = dir.norm()
    # 稍微留一点间隙摩擦碰撞边界 (加一个小 epsilon 0.002)
    min_dist = sphere_radius + 0.002
    if dist < min_dist:
        normal = dir / dist
        # 1. 位置投影：将穿透的质点推回球体表面
        pos[idx] = sphere_center + normal * min_dist
        
        # 2. 速度消除：移除向球心运动的法向速度分量并施加表面摩擦
        v_normal = vel[idx].dot(normal)
        if v_normal < 0:
            vel[idx] -= v_normal * normal  # 完美的滑移
            vel[idx] *= 0.85               # 模拟表面摩擦阻尼

# ==================== 积分 Kernel (含碰撞响应) ====================

@ti.kernel
def step_explicit():
    """显式欧拉"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉 (定点迭代法)"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
        
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                resolve_collision(x_next, v_next, i) # 在迭代内约束碰撞
                
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ==================== 主函数与渲染 ====================
def main():
    init_cloth()

    window = ti.ui.Window("Games101 - Mass Spring System (Bonus)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.8, 1.6)
    camera.lookat(0.0, 0.2, 0.0)

    current_method = 1 
    paused = False

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)
        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("") 
        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.end()

        # =========== 物理步更新 ===========
        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # =========== 渲染场景 ===========
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(1, 1, 1))

        # 1. 绘制碰撞体（红色球体）
        scene.particles(sphere_center_field, radius=sphere_radius, color=(0.9, 0.3, 0.3))
        
        # 2. 绘制布料网格顶点和弹簧线框
        scene.particles(x, radius=0.012, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.0, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()