import numpy as np
import warp as wp
import warp.render

@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), gravity: wp.vec3, dt: float):
    tid = wp.tid()
    
    v_new = v[tid] + gravity * dt
    x_new = x[tid] + v_new * dt

    if x_new.y < 0.05:
        x_new.y = 0.05
        v_new.y = -v_new.y * 0.5

    v[tid] = v_new
    x[tid] = x_new

class Example:

    def __init__(self):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.renderer = wp.render.OpenGLRenderer(vsync=False)
        self.dim_x = 20
        self.dim_y = 20

        self.point_radius = 0.05


        self.start_x = self.create_particles_grid(self.dim_x, self.dim_y, 0.1)
        self.x = wp.array(self.start_x, dtype=wp.vec3)
        self.v = wp.array(np.zeros([len(self.x), 3]), dtype=wp.vec3)
        self.renderer.render_ground()

        self.use_cuda_graph = False
        # self.use_cuda_graph = wp.get_device().is_cuda
        # if self.use_cuda_graph:
        #     with wp.ScopedCapture() as capture:
        #         self.simulate()
        #     self.graph = capture.graph

    def create_particles_grid(self, dimx, dimy, spacing):
        x = []
        for i in range (dimx):
            for j in range (dimy):
                v = (i * spacing, 1.0, j * spacing)
                x.append(v)
        return np.array(x, dtype=np.float32)
        

    def simulate(self):
        with wp.ScopedTimer("simulate"):
            for _ in range(self.sim_substeps):
                wp.launch(
                    kernel=integrate,
                    dim=len(self.x),
                    inputs=[
                        self.x,
                        self.v,
                        wp.vec3(0.0, -9.81, 0.0),
                        self.sim_dt
                    ],
                )

    def step(self):
        with wp.ScopedTimer("step"):

            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

            self.sim_time += self.frame_dt

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)

        self.renderer.render_points(
                points=self.x, radius=self.point_radius, name="points", colors=(0.8, 0.3, 0.2)
            )

        self.renderer.end_frame()

    def clear(self):
        self.renderer.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--render_mode", type=str, choices=("depth", "rgb"), default="depth", help="")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        while example.renderer.is_running():
            example.step()
            example.render()

        example.clear()