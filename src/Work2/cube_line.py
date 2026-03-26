## 线框丝滑版本

import taichi as ti
import math

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32):
    rx = angle_x * math.pi / 180.0
    ry = angle_y * math.pi / 180.0
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)

    return ti.Matrix([
        [cy,  0,   sy,    0],
        [sx*sy, cx, -sx*cy, 0],
        [-cx*sy, sx, cx*cy, 0],
        [0,    0,    0,    1]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 12.0])
    model = get_model_matrix(angle_x, angle_y)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(60.0, 1.0, 0.1, 100.0)
    mvp = proj @ view @ model

    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    vertices[0] = [-1, -1, -1]
    vertices[1] = [1, -1, -1]
    vertices[2] = [1, 1, -1]
    vertices[3] = [-1, 1, -1]
    vertices[4] = [-1, -1, 1]
    vertices[5] = [1, -1, 1]
    vertices[6] = [1, 1, 1]
    vertices[7] = [-1, 1, 1]

    gui = ti.GUI("3D Cube ✨ 完美自由旋转 ✨", res=(800, 800))

    angle_x = 0.0
    angle_y = 0.0

    while gui.running:
        # ====================== 这里改成了平滑版本！======================
        gui.get_event()  # 每帧都获取事件
        if gui.is_pressed('a'): angle_y += 2.5  # 按住就连续转
        if gui.is_pressed('d'): angle_y -= 2.5
        if gui.is_pressed('w'): angle_x += 2.5
        if gui.is_pressed('s'): angle_x -= 2.5
        if gui.is_pressed(ti.GUI.ESCAPE): gui.running = False

        compute_transform(angle_x, angle_y)

        for (i, j) in edges:
            a = screen_coords[i]
            b = screen_coords[j]
            gui.line(a, b, radius=2, color=0x00FFFF)

        gui.show()

if __name__ == '__main__':
    main()