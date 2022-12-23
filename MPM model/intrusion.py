import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

mixed = 0
quality = 1
real_x = 0.3  # %%m
n_particles = 40000 * quality ** 2
n_g_particles = ti.field(dtype=int, shape=())
n_w_particles = ti.field(dtype=int, shape=())
t = ti.field(dtype=float, shape=())
n_grid = 256 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
gravity = ti.Vector([0, -9.8])
d = 2
v_b = 0.01
pos_g = ti.Vector.field(2, dtype=float, shape=n_particles)
pos_w = ti.Vector.field(2, dtype=float, shape=n_particles)

x_b = ti.field(dtype=float, shape=())
F_b = ti.field(dtype=float, shape=())

# glass particle properties
x_g = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v_g = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C_g = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity matrix
F_g = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
phi_g = ti.field(dtype=float, shape=n_particles)  # cohesion and saturation
c_C0 = ti.field(dtype=float, shape=n_particles)  # initial cohesion (as maximum)
vc_g = ti.field(dtype=float, shape=n_particles)  # tracks changes in the log of the volume gained during extension
alpha_g = ti.field(dtype=float, shape=n_particles)  # yield surface size
q_g = ti.field(dtype=float, shape=n_particles)  # harding state

# glass grid properties
grid_gv = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_gm = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
grid_gf = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # forces in the sand

# water particle properties
x_w = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v_w = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C_w = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity matrix
J_w = ti.field(dtype=float, shape=n_particles)  # ratio of volume increase

grid_wv = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_wm = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
grid_wf = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # forces in the water


# constant values
p_vol, g_rho, w_rho = (dx * 0.5) ** 2, 1442, 1442
g_mass, w_mass = p_vol * g_rho, p_vol * w_rho

w_k, w_gamma = 50, 3

n, k_hat = 0.4, 0.2

E_g, nu_g = 3.537e5, 0.3  # glass's Young's modulus and Poisson's ratio
mu_g, lambda_g = E_g / (2 * (1 + nu_g)), E_g * nu_g / ((1 + nu_g) * (1 - 2 * nu_g))

mu_b = 0.3  # coefficient of friction

a, b, c0, sC = 1442 / 20 * -3.0, 0, 1e-2, 0.15


@ti.func
def h_s(z):
    ret = 0.0
    if z < 0: ret = 1
    if z > 1: ret = 0
    ret = 1 - 10 * (z ** 3) + 15 * (z ** 4) - 6 * (z ** 5)
    return ret


# multiplier
sqrt2 = ti.sqrt(2)


@ti.func
def h(e):
    u = e.trace() / sqrt2
    v = ti.abs(ti.Vector([e[0, 0] - u / sqrt2, e[1, 1] - u / sqrt2]).norm())
    fe = c0 * (v ** 4) / (1 + v ** 3)

    ret = 0.0
    if u + fe < a + sC: ret = 1
    if u + fe > b + sC: ret = 0
    ret = h_s((u + fe - a - sC) / (b - a))
    ret = 1
    return ret


state = ti.field(dtype=int, shape=n_particles)
pi = 3.14159265358979


@ti.func
def project(e0, p):
    e = e0 + vc_g[p] / d * ti.Matrix.identity(float, 2)
    e += (c_C0[p] * (1.0 - phi_g[p])) * ti.Matrix.identity(float, 2)
    ehat = e - e.trace() / d * ti.Matrix.identity(float, 2)
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2)
    yp = Fnorm + (d * lambda_g + 2 * mu_g) / (2 * mu_g) * e.trace() * alpha_g[p]
    new_e = ti.Matrix.zero(float, 2, 2)
    delta_q = 0.0
    if Fnorm <= 0 or e.trace() > 0:
        new_e = ti.Matrix.zero(float, 2, 2)
        delta_q = ti.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2)
        state[p] = 0
    elif yp <= 0:
        new_e = e0
        delta_q = 0
        state[p] = 1
    else:
        new_e = e - yp / Fnorm * ehat
        delta_q = yp
        state[p] = 2

    return new_e, delta_q


h0, h1, h2, h3 = 35, 9, 0.3, 10


@ti.func
def hardening(dq, p):
    q_g[p] += dq
    phi = h0 + (h1 * q_g[p] - h3) * ti.exp(-h2 * q_g[p])
    phi = phi / 180 * pi
    sin_phi = ti.sin(phi)
    alpha_g[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)


@ti.kernel
def substep():
    # set zero initial state for glass grid
    for i, j in grid_gm:
        grid_gv[i, j], grid_wv[i, j] = [0, 0], [0, 0]
        grid_gm[i, j], grid_wm[i, j] = 0, 0
        grid_gf[i, j], grid_wf[i, j] = [0, 0], [0, 0]

    grid_b = int(x_b[None] * inv_dx - 0.5)
    F_b[None] = 0
    grid_by = int(0.15 * inv_dx - 0.5)
    grid_by2 = int(0.2 * inv_dx - 0.5)
    for p in x_g:
        base = (x_g[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        fx = x_g[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F_g[p])
        inv_sig = sig.inverse()
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        stress = U @ (2 * mu_g * inv_sig @ e + lambda_g * e.trace() * inv_sig) @ V.transpose()  # formula (25)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress @ F_g[p].transpose()
        affine = g_mass * C_g[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_gv[base + offset] += weight * (g_mass * v_g[p] + affine @ dpos)
            grid_gm[base + offset] += weight * g_mass
            grid_gf[base + offset] += weight * stress @ dpos

    for p in x_w:
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = w_k * (1 - 1 / (J_w[p] ** w_gamma))
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress * J_w[p]
        affine = w_mass * C_w[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_wv[base + offset] += weight * (w_mass * v_w[p] + affine @ dpos)
            grid_wm[base + offset] += weight * w_mass
            grid_wf[base + offset] += weight * stress * dpos

    # Update Grids Momentum
    for i, j in grid_gm:
        if grid_gm[i, j] > 0:
            grid_gv[i, j] = (1 / grid_gm[i, j]) * grid_gv[i, j]  # Momentum to velocity
        if grid_wm[i, j] > 0:
            grid_wv[i, j] = (1 / grid_wm[i, j]) * grid_wv[i, j]

        cE = (n * n * w_rho * gravity[1]) / k_hat
        if grid_gm[i, j] > 0 and grid_wm[i, j] > 0:
            sm, wm = grid_gm[i, j], grid_wm[i, j]
            sv, wv = grid_gv[i, j], grid_wv[i, j]
            dd = cE * sm * wm
            M = ti.Matrix([[sm, 0], [0, wm]])
            D = ti.Matrix([[-dd, dd], [dd, -dd]])
            V = ti.Matrix.rows([grid_gv[i, j], grid_wv[i, j]])
            G = ti.Matrix.rows([gravity, gravity])
            F = ti.Matrix.rows([grid_gf[i, j], grid_wf[i, j]])

            A = M + dt * D
            B = M @ V + dt * (M @ G + F)
            X = A.inverse() @ B
            grid_gv[i, j], grid_wv[i, j] = ti.Vector([X[0, 0], X[0, 1]]), ti.Vector([X[1, 0], X[1, 1]])

        elif grid_gm[i, j] > 0:
            grid_gv[i, j] += dt * (gravity + grid_gf[i, j] / grid_gm[i, j])  # Update explicit force
        elif grid_wm[i, j] > 0:
            grid_wv[i, j] += dt * (gravity + grid_wf[i, j] / grid_wm[i, j])

        normal = ti.Vector.zero(float, 2)
        if grid_gm[i, j] > 0:
            if i < 3 and grid_gv[i, j][0] < 0:          normal = ti.Vector([1, 0])
            if i > n_grid - 3 and grid_gv[i, j][0] > 0: normal = ti.Vector([-1, 0])
            if j < 3 and grid_gv[i, j][1] < 0:          normal = ti.Vector([0, 1])
            if j > n_grid - 3 and grid_gv[i, j][1] > 0: normal = ti.Vector([0, -1])
        if not (normal[0] == 0 and normal[1] == 0):  # Apply friction
            s = grid_gv[i, j].dot(normal)
            if s <= 0:
                v_normal = s * normal
                v_tangent = grid_gv[i, j] - v_normal
                vt = v_tangent.norm()
                if vt > 1e-12: grid_gv[i, j] = v_tangent - (vt if vt < -mu_b * s else -mu_b * s) * (v_tangent / vt)
        normal2 = ti.Vector.zero(float, 2)
        if grid_gm[i, j] > 0:
            if t[None] <= 2.0:
                continue
            if i >= grid_b and i < grid_b + 1 and grid_gv[i, j][0] < v_b and j >= grid_by and j < grid_by2:
                normal2 = ti.Vector([1, 0])
            elif i < grid_b and i > grid_b - 2 and grid_gv[i, j][0] > v_b and j >= grid_by and j < grid_by2:
                normal2 = ti.Vector([1, 0])
        if not (normal2[0] == 0):  # Apply friction
            s = grid_gv[i, j].dot(normal2)
            v_normal = s * normal2
            v_tangent = grid_gv[i, j] - v_normal
            vt = v_tangent.norm()
            F_b[None] += (v_b - grid_gv[i, j][0]) * grid_gm[i, j] / dt
            if vt > 1e-12:
                grid_gv[i, j] = v_tangent - (vt if vt < -mu_b * s else -mu_b * s) * (v_tangent / vt)
            grid_gv[i, j][0] = v_b


        if grid_wm[i, j] > 0:
            if i < 3 and grid_wv[i, j][0] < 0:          grid_wv[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_wv[i, j][0] > 0: grid_wv[i, j][0] = 0
            if j < 3 and grid_wv[i, j][1] < 0:          grid_wv[i, j][1] = 0
            if j > n_grid - 3 and grid_wv[i, j][1] > 0: grid_wv[i, j][1] = 0

    for p in range(n_w_particles[None]):
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_wv[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        J_w[p] = (1 + dt * new_C.trace()) * J_w[p]
        v_w[p], C_w[p] = new_v, new_C
        x_w[p] += dt * v_w[p]

    # G2P (glass's part)
    for p in x_g:
        base = (x_g[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        fx = x_g[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        phi_g[p] = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_gv[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            if grid_gm[base + ti.Vector([i, j])] > 0 and grid_wm[base + ti.Vector([i, j])] > 0:
                phi_g[p] += weight

        F_g[p] = (ti.Matrix.identity(float, 2) + dt * new_C) @ F_g[p]
        v_g[p], C_g[p] = new_v, new_C
        x_g[p] += dt * v_g[p]

        U, sig, V = ti.svd(F_g[p])
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        new_e, dq = project(e, p)
        hardening(dq, p)
        new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0], [0, ti.exp(new_e[1, 1])]]) @ V.transpose()
        vc_g[p] += -ti.log(new_F.determinant()) + ti.log(F_g[p].determinant())  # formula (26)
        F_g[p] = new_F
    t[None] += dt
    if t[None] > 2.0:
        x_b[None] += v_b * dt
    print(F_b[None])



@ti.kernel
def initialize():
    x_b[None] = 0.0
    t[None] = 0.0
    n_g_particles[None] = 100
    for i in x_g:
        x_g[i] = [ti.random() * 0.9 + 0.05, ti.random() * 0.001111 + 0.04]
        v_g[i] = ti.Matrix([0, 0])
        F_g[i] = ti.Matrix([[1, 0], [0, 1]])
        c_C0[i] = 0
        alpha_g[i] = 0
    pos_y[None] = 0.5
    n_w_particles[None] = 0


pos_y = ti.field(dtype=float, shape=())


@ti.kernel
def sand_jet():
    if n_g_particles[None] < 40000 - 2250:
        for i in range(n_g_particles[None], n_g_particles[None] + 2250):
            x_g[i] = [ti.random() * 1 , ti.random() * 0.025 + 0.95]
            v_g[i] = ti.Matrix([0, -2.5])
            F_g[i] = ti.Matrix([[1, 0], [0, 1]])
            c_C0[i] = 0
            alpha_g[i] = 0

        n_g_particles[None] += 2250


@ti.kernel
def update_jet():
    if n_w_particles[None] < 40000 - 41:
        for i in range(n_w_particles[None], n_w_particles[None] + 50):
            x_w[i] = [ti.random() * 0.03 + 0.92, ti.random() * 0.03 + pos_y[None]]
            v_w[i] = ti.Matrix([-1.5, 0])
            J_w[i] = 1

        n_w_particles[None] += 41


@ti.kernel
def add_block(x: ti.f32):
    if n_g_particles[None] < 40000 - 1000:
        for i in range(n_g_particles[None], n_g_particles[None] + 1000):
            x_g[i] = [ti.min(0.87, x) + ti.random() * 0.1, ti.random() * 0.1 + 0.87]
            v_g[i] = ti.Matrix([0, -0.25])
            F_g[i] = ti.Matrix([[1, 0], [0, 1]])
            # c_C0[i] = -0.01
            c_C0[i] = 0
            alpha_g[i] = 0

    n_g_particles[None] += 1000


@ti.func
def color_lerp(r1, g1, b1, r2, g2, b2, t):
    return int((r1 * (1 - t) + r2 * t) * 0x100) * 0x10000 + int((g1 * (1 - t) + g2 * t) * 0x100) * 0x100 + int(
        (b1 * (1 - t) + b2 * t) * 0x100)


color_s = ti.field(dtype=int, shape=n_particles)
color_w = ti.field(dtype=int, shape=n_particles)


@ti.kernel
def update_color():
    for i in range(n_g_particles[None]):
        color_s[i] = color_lerp(0.2, 0.231, 0.792, 0.867, 0.886, 0.886, v_g[i].norm() * 5)
    for i in range(n_w_particles[None]):
        color_w[i] = color_lerp(0.2, 0.231, 0.792, 0.867, 0.886, 0.886, v_w[i].norm() / 3.0)


@ti.kernel
def update_pos():
    for i in range(n_g_particles[None]):
        pos_g[i] = x_g[i]
    for i in range(n_w_particles[None]):
        pos_w[i] = x_w[i]


initialize()

project_view = False
frame = 0
gui = ti.GUI("glass_bed", res=512, background_color=0xFFFFFF)
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == gui.SPACE:
            project_view = not project_view
        elif e.key == 'w':
            pos_y[None] += 0.01
        elif e.key == 's':
            pos_y[None] -= 0.01
        elif e.key == ti.GUI.LMB:
            add_block(e.pos[0])
        elif e.key == 'm':
            mixed = 1
        elif e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()

    if n_g_particles[None] >= 20000 * quality ** 2 - 500 and mixed == 1:
        update_jet()
    sand_jet()
    for s in range(120):
        substep()

    update_pos()
    if project_view:
        gui.circles(pos_w.to_numpy(), radius=1.5, color=0x068587)
        colors = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.uint32)
        gui.circles(pos_g.to_numpy(), radius=1.5, color=colors[state.to_numpy()])
    else:
        update_color()
        gui.circles(pos_w.to_numpy(), radius=1.5, color=color_w.to_numpy())
        gui.circles(pos_g.to_numpy(), radius=1.5, color=color_s.to_numpy())
    # gui.show(f'img4/{frame:06d}.png')
    gui.show()
    # print(max(alpha_s.to_numpy()))
    # print(n_s_particles[None])
    # print(frame)
    frame += 1