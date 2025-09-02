import torch
import torch.fft

from .integrator import SemiImplicitEulerIntegrator
from .PDEmodel import PDEModel
import pygame
import numpy as np

class SpectralSolver:
    def __init__(self, shape, L=2 * torch.pi, dt=0.01, batch_size = 1, device='cuda'):

        self.shape = shape
        self.L = L
        self.dt = dt
        self.device = device
        self.batch_size = batch_size

        self._init_q_space()

        self.model = PDEModel(shape, device, batch_size=batch_size)
        self.integrator_cl = SemiImplicitEulerIntegrator

    def _init_q_space(self):
        # Support 1D, 2D, or 3D grids depending on self.shape
        dims = len(self.shape)
        axes = []
        q_axes = []
        for i, N in enumerate(self.shape):
            x = torch.linspace(0, self.L - self.L / N, N, device=self.device)
            axes.append(x)
            q = torch.fft.fftfreq(N, d=self.L / N).to(self.device) * 2 * torch.pi
            q_axes.append(q)

        # Create spatial grids
        self.spatial_grids = torch.meshgrid(*axes, indexing='ij')

        # Create wavenumber grids
        q_grids = torch.meshgrid(*q_axes, indexing='ij')
        self.q_grids = q_grids

        # Assign qx, qy, qz if present
        self.qx = q_grids[0]
        self.qy = q_grids[1] if dims > 1 else None
        self.qz = q_grids[2] if dims > 2 else None

        # Compute q^2
        self.q2 = sum(q**2 for q in q_grids)
        if dims == 1:
            self.q2[0] = 1e-10
        elif dims == 2:
            self.q2[0, 0] = 1e-10
        elif dims == 3:
            self.q2[0, 0, 0] = 1e-10


    def build(self):
        self.model.fields.set_wavenumbers(self.qx, self.qy, self.qz, self.q2)
        self.model.build()
        self.integrator = self.integrator_cl(self.model, self.dt, self.qx, self.qy, self.q2)

    # add ability to reset initial state of dynamic fields
    def reset(self, inits={}):
        self.model.build()
        for name, val in inits.items():
            self.model.fields[name] = val
        self.model.fields.spectral = self.model.fields.fftn()

    def run(self, steps, callback = None):
        for step in range(steps):
            self.integrator.step()
            
            if callback is not None:
                callback(self,step)


    def visualize(self, data, filename="output.mp4", fps=20, cmap="viridis"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        im = ax.imshow(data[0].cpu().numpy(), cmap=cmap, origin='lower', extent=[0, self.L, 0, self.L])
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_data(data[frame].cpu().numpy())
            im.set_clim(vmin=0, vmax=1)
            # im.set_clim(vmin=data[frame].min().item(), vmax=data[frame].max().item())
            fig.canvas.draw_idle()
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=data.shape[0], interval=1000/fps, blit=True
        )
        ani.save(filename, writer='ffmpeg', fps=fps)
        plt.show()
        plt.close(fig)
        

    def visualize_pygame(self, data, scale=2, cmap="viridis"):
        import matplotlib.pyplot as plt

        pygame.init()
        n_frames, N, _ = data.shape
        width, height = N * scale, N * scale

        # Prepare colormap
        cmap_func = plt.get_cmap(cmap)
        norm = lambda arr: (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PDE Visualization (Interactive)")

        running = True
        frame = 0
        clock = pygame.time.Clock()

        while running:
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Move frames while holding arrow keys
            if keys[pygame.K_RIGHT]:
                frame = min(frame + 1, n_frames - 1)
            if keys[pygame.K_LEFT]:
                frame = max(frame - 1, 0)

            arr = data[frame].cpu().numpy()
            arr_norm = norm(arr)
            arr_rgb = (cmap_func(arr_norm)[..., :3] * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(np.kron(arr_rgb, np.ones((scale, scale, 1))), (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()