import taichi as ti
import math

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)
# 存储每个顶点变换后的 3D 坐标（用于计算深度）
transformed_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

faces = [
    (0, 1, 2, 3),
    (1, 5, 6, 2),
    (5, 4, 7, 6),
    (4, 0, 3, 7),
    (3, 2, 6, 7),
    (4, 5, 1, 0)
]

face_colors = [
    0x90FCBE,
    0xEBFC90,
    0xFCCB90,
    0xFC9090,
    0x90DEFC,
    0xA190FC
]


@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32):
    rx = angle_x * math.pi / 180.0
    ry = angle_y * math.pi / 180.0
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    return ti.Matrix([
        [cy, 0, sy, 0],
        [sx * sy, cx, -sx * cy, 0],
        [-cx * sy, sx, cx * cy, 0],
        [0, 0, 0, 1]
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
def get_projection_matrix(eye_fov, aspect, zNear, zFar):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2) * abs(n)
    b = -t
    r = aspect * t
    l = -r
    M_p2o = ti.Matrix([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n + f, -n * f],
        [0, 0, 1, 0]
    ])
    M_ortho = ti.Matrix([
        [2 / (r - l), 0, 0, -(r + l) / 2],
        [0, 2 / (t - b), 0, -(t + b) / 2],
        [0, 0, 2 / (n - f), -(n + f) / 2],
        [0, 0, 0, 1]
    ])
    return M_ortho @ M_p2o


@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32):
    eye_pos = ti.Vector([0, 0, 12])
    model = get_model_matrix(angle_x, angle_y)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(60, 1, 0.1, 100)
    mvp = proj @ view @ model

    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1])
        r = mvp @ v4
        r /= r[3]
        screen_coords[i] = [(r.x + 1) / 2, (r.y + 1) / 2]
        # 保存变换后的 3D 坐标
        transformed_vertices[i] = [r.x, r.y, r.z]


def main():
    vertices[0] = [-1, -1, -1]
    vertices[1] = [1, -1, -1]
    vertices[2] = [1, 1, -1]
    vertices[3] = [-1, 1, -1]
    vertices[4] = [-1, -1, 1]
    vertices[5] = [1, -1, 1]
    vertices[6] = [1, 1, 1]
    vertices[7] = [-1, 1, 1]

    gui = ti.GUI("3D Cube Faces", res=(800, 800))
    ax, ay = 0.0, 0.0

    while gui.running:
        gui.get_event()
        if gui.is_pressed('a'): ay += 2.5
        if gui.is_pressed('d'): ay -= 2.5
        if gui.is_pressed('w'): ax += 2.5
        if gui.is_pressed('s'): ax -= 2.5
        if gui.is_pressed(ti.GUI.ESCAPE): gui.running = False

        compute_transform(ax, ay)

        face_depth = []
        for i in range(6):
            a, b, c, d = faces[i]
            # 用四个顶点的平均深度代表面的深度
            z = (transformed_vertices[a].z + transformed_vertices[b].z +
                 transformed_vertices[c].z + transformed_vertices[d].z) / 4
            face_depth.append((z, i))

        # 按深度从远到近排序
        face_depth.sort()

        for z, f in face_depth:
            a, b, c, d = faces[f]
            p0 = screen_coords[a]
            p1 = screen_coords[b]
            p2 = screen_coords[c]
            p3 = screen_coords[d]
            gui.triangle(p0, p1, p2, color=face_colors[f])
            gui.triangle(p0, p2, p3, color=face_colors[f])

        # 绘制边框
        for i, j in edges:
            gui.line(screen_coords[i], screen_coords[j], radius=2, color=0xFFFFFF)

        gui.show()


if __name__ == '__main__':
    main()