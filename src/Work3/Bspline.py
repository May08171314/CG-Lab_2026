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

# 模式：0=贝塞尔，1=B样条
mode = 0


def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)


def b_spline(points, t):
    n = len(points)
    if n < 4:
        return de_casteljau(points, t)
    seg = n - 3
    idx = int(t * seg)
    idx = min(max(idx, 0), seg - 1)
    t_local = t * seg - idx

    p0 = points[idx]
    p1 = points[idx + 1]
    p2 = points[idx + 2]
    p3 = points[idx + 3]

    t2 = t_local * t_local
    t3 = t2 * t_local

    x = ((-t3 + 3 * t2 - 3 * t_local + 1) * p0[0] +
         (3 * t3 - 6 * t2 + 4) * p1[0] +
         (-3 * t3 + 3 * t2 + 3 * t_local + 1) * p2[0] +
         t3 * p3[0]) / 6

    y = ((-t3 + 3 * t2 - 3 * t_local + 1) * p0[1] +
         (3 * t3 - 6 * t2 + 4) * p1[1] +
         (-3 * t3 + 3 * t2 + 3 * t_local + 1) * p2[1] +
         t3 * p3[1]) / 6
    return [x, y]


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = [0.0, 0.0, 0.0]


@ti.kernel
def draw_curve(n: ti.i32, m: ti.i32):
    for i in range(n):
        x, y = curve_points_field[i]
        px = ti.cast(x * WIDTH, ti.i32)
        py = ti.cast(y * HEIGHT, ti.i32)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            if m == 0:
                pixels[px, py] = [0.0, 1.0, 0.0]
            else:
                pixels[px, py] = [1.0, 0.0, 1.0]


def main():
    global mode
    window = ti.ui.Window("Bezier / B-Spline Switch (Press B)", (WIDTH, HEIGHT))
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
            elif e.key == 'b':
                mode = 1 - mode
                print("Switched to:", "B-Spline" if mode else "Bezier")

        clear_pixels()
        cnt = len(control_points)

        if cnt >= 2:
            cps = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                if mode == 0:
                    cps[i] = de_casteljau(control_points, t)
                else:
                    cps[i] = b_spline(control_points, t)
            curve_points_field.from_numpy(cps)
            draw_curve(NUM_SEGMENTS + 1, mode)

        canvas.set_image(pixels)

        if cnt > 0:
            np_p = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_p[:cnt] = control_points
            gui_points.from_numpy(np_p)
            canvas.circles(gui_points, radius=0.006, color=(1, 0, 0))

            if cnt >= 2:
                idx = []
                for i in range(cnt - 1):
                    idx += [i, i + 1]
                np_idx = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_idx[:len(idx)] = idx
                gui_indices.from_numpy(np_idx)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        window.show()


if __name__ == '__main__':
    main()