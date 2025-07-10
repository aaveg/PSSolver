import torch
import torch.fft

from integrator import SemiImplicitEulerIntegrator
from PDEModel import PDEModel
import pygame
import numpy as np

class System:
    def __init__(self, N, L=2 * torch.pi, dt=0.01, device='cuda', record_every_n_steps=1000):

        self.N = N
        self.L = L
        self.dt = dt
        self.device = device
        self.record_every_n_steps = record_every_n_steps

        self._init_q_space()

        self.integrator = SemiImplicitEulerIntegrator(dt, self.qx, self.qy, self.q2)
        self.model = PDEModel(N, device)

    def _init_q_space(self):
        # Spatial grid (not strictly needed but kept for extensibility)
        x = torch.linspace(0, self.L - self.L / self.N, self.N, device=self.device)
        self.X, self.Y = torch.meshgrid(x, x, indexing='ij')

        # Wavenumbers
        q = torch.fft.fftfreq(self.N, d=self.L/self.N).to(self.device) * 2 * torch.pi
        self.qx, self.qy = torch.meshgrid(q, q, indexing='ij')
        self.q2 = self.qx**2 + self.qy**2
        self.q2[0, 0] = 1e-10  # avoid div by 0 if using 1/q^2 later

    
    def run(self, steps, callback = None):
        for step in range(steps):
            self.integrator.step(self.model)
            
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
            im.set_clim(vmin=data[frame].min().item(), vmax=data[frame].max().item())
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