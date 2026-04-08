import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 反走样参数
ALPHA = 6.0


def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_antialiased_curve(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        fx = pt[0] * WIDTH
        fy = pt[1] * HEIGHT

        # 3x3 邻域反走样
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = ti.cast(fx + 0.5, ti.i32) + dx
                py = ti.cast(fy + 0.5, ti.i32) + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    dx_p = fx - (px + 0.5)
                    dy_p = fy - (py + 0.5)
                    dist = ti.sqrt(dx_p ** 2 + dy_p ** 2)
                    weight = ti.exp(-dist * dist / (2 * ALPHA * ALPHA))
                    pixels[px, py] += ti.Vector([0.0, 1.0, 0.0]) * weight


def main():
    window = ti.ui.Window("Anti-Aliasing Bezier Curve", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c':
                control_points = []

        clear_pixels()
        current_count = len(control_points)

        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)

            curve_points_field.from_numpy(curve_points_np)
            draw_antialiased_curve(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        window.show()


if __name__ == '__main__':
    main()