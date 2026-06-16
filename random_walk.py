import numpy as np
from typing import Tuple, Optional, Union, List


class RandomWalkService:
    """一维/二维随机游走模拟服务"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def simulate_1d(
        self,
        n_steps: int = 1000,
        start: float = 0.0,
        step_size: float = 1.0,
        p: float = 0.5,
    ) -> np.ndarray:
        """
        一维随机游走模拟

        参数:
            n_steps: 游走步数
            start: 起始位置
            step_size: 每步步长
            p: 向右走的概率 (0~1)

        返回:
            一维轨迹数组，形状 (n_steps + 1,)
        """
        if n_steps < 0:
            raise ValueError("n_steps 必须是非负整数")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p 必须在 [0, 1] 范围内")

        steps = self._rng.choice([-step_size, step_size], size=n_steps, p=[1 - p, p])
        trajectory = np.concatenate([[start], np.cumsum(steps)])
        return trajectory

    def simulate_2d(
        self,
        n_steps: int = 1000,
        start: Tuple[float, float] = (0.0, 0.0),
        step_size: float = 1.0,
        mode: str = "lattice",
        bounds: Optional[Tuple[float, float, float, float]] = None,
        boundary_type: str = "none",
    ) -> np.ndarray:
        """
        二维随机游走模拟

        参数:
            n_steps: 游走步数
            start: 起始坐标 (x, y)
            step_size: 每步步长
            mode: 游走模式
                - "lattice": 格点游走（上下左右）
                - "continuous": 连续方向游走（任意角度）
            bounds: 边界范围 (x_min, x_max, y_min, y_max)，None 表示无边界
            boundary_type: 边界处理类型
                - "none": 无边界（默认）
                - "reflecting": 反射边界（碰到边界后反弹）
                - "absorbing": 吸收边界（碰到边界后停留在边界，不再移动）

        返回:
            二维轨迹数组，形状 (n_steps + 1, 2)，每行为 (x, y)
        """
        if n_steps < 0:
            raise ValueError("n_steps 必须是非负整数")
        if boundary_type not in ("none", "reflecting", "absorbing"):
            raise ValueError(
                "boundary_type 必须是 'none', 'reflecting' 或 'absorbing'"
            )
        if boundary_type != "none" and bounds is None:
            raise ValueError("指定 boundary_type 时必须同时提供 bounds")

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("边界范围无效，需满足 x_min < x_max 且 y_min < y_max")
            if not (x_min <= start[0] <= x_max and y_min <= start[1] <= y_max):
                raise ValueError("起始位置必须在边界范围内")

        if mode == "lattice":
            directions = self._rng.integers(0, 4, size=n_steps)
            dx_all = np.zeros(n_steps)
            dy_all = np.zeros(n_steps)
            dx_all[directions == 0] = step_size
            dx_all[directions == 1] = -step_size
            dy_all[directions == 2] = step_size
            dy_all[directions == 3] = -step_size
        elif mode == "continuous":
            angles = self._rng.uniform(0, 2 * np.pi, size=n_steps)
            dx_all = step_size * np.cos(angles)
            dy_all = step_size * np.sin(angles)
        else:
            raise ValueError("mode 必须是 'lattice' 或 'continuous'")

        if boundary_type == "none" or bounds is None:
            x = np.concatenate([[start[0]], np.cumsum(dx_all)])
            y = np.concatenate([[start[1]], np.cumsum(dy_all)])
            return np.column_stack([x, y])

        x_min, x_max, y_min, y_max = bounds
        trajectory = np.zeros((n_steps + 1, 2))
        trajectory[0] = start
        absorbed = False

        for i in range(n_steps):
            if absorbed:
                trajectory[i + 1] = trajectory[i]
                continue

            x_curr, y_curr = trajectory[i]
            dx = dx_all[i]
            dy = dy_all[i]

            x_next = x_curr + dx
            y_next = y_curr + dy

            if boundary_type == "reflecting":
                if x_next < x_min:
                    x_next = x_min + (x_min - x_next)
                    dx_all[i] = -dx
                elif x_next > x_max:
                    x_next = x_max - (x_next - x_max)
                    dx_all[i] = -dx

                if y_next < y_min:
                    y_next = y_min + (y_min - y_next)
                    dy_all[i] = -dy
                elif y_next > y_max:
                    y_next = y_max - (y_next - y_max)
                    dy_all[i] = -dy

            elif boundary_type == "absorbing":
                if x_next <= x_min or x_next >= x_max or y_next <= y_min or y_next >= y_max:
                    x_next = max(x_min, min(x_next, x_max))
                    y_next = max(y_min, min(y_next, y_max))
                    absorbed = True

            trajectory[i + 1] = (x_next, y_next)

        return trajectory

    def simulate_batch_1d(
        self,
        n_walkers: int,
        n_steps: int = 1000,
        start: Union[float, np.ndarray] = 0.0,
        step_size: float = 1.0,
        p: float = 0.5,
    ) -> np.ndarray:
        """
        批量一维随机游走模拟

        参数:
            n_walkers: 游走者数量
            n_steps: 游走步数
            start: 起始位置（标量或形状为 (n_walkers,) 的数组）
            step_size: 每步步长
            p: 向右走的概率

        返回:
            轨迹数组，形状 (n_walkers, n_steps + 1)
        """
        if n_walkers <= 0:
            raise ValueError("n_walkers 必须是正整数")

        steps = self._rng.choice(
            [-step_size, step_size], size=(n_walkers, n_steps), p=[1 - p, p]
        )
        if np.isscalar(start):
            start_arr = np.full((n_walkers, 1), start, dtype=float)
        else:
            start_arr = np.asarray(start, dtype=float).reshape(n_walkers, 1)

        trajectory = np.hstack([start_arr, np.cumsum(steps, axis=1)])
        return trajectory

    def simulate_batch_2d(
        self,
        n_walkers: int,
        n_steps: int = 1000,
        start: Union[Tuple[float, float], np.ndarray] = (0.0, 0.0),
        step_size: float = 1.0,
        mode: str = "lattice",
        bounds: Optional[Tuple[float, float, float, float]] = None,
        boundary_type: str = "none",
    ) -> np.ndarray:
        """
        批量二维随机游走模拟

        参数:
            n_walkers: 游走者数量
            n_steps: 游走步数
            start: 起始坐标（二元组或形状为 (n_walkers, 2) 的数组）
            step_size: 每步步长
            mode: 游走模式 "lattice" 或 "continuous"
            bounds: 边界范围 (x_min, x_max, y_min, y_max)，None 表示无边界
            boundary_type: 边界处理类型 "none", "reflecting", "absorbing"

        返回:
            轨迹数组，形状 (n_walkers, n_steps + 1, 2)
        """
        if n_walkers <= 0:
            raise ValueError("n_walkers 必须是正整数")

        trajectories = []
        for _ in range(n_walkers):
            if isinstance(start, np.ndarray) and start.ndim == 2:
                s = tuple(start[_])
            else:
                s = start
            trajectories.append(
                self.simulate_2d(
                    n_steps, s, step_size, mode, bounds, boundary_type
                )
            )
        return np.array(trajectories)


def main():
    service = RandomWalkService(seed=42)

    traj_1d = service.simulate_1d(n_steps=10, start=0.0, step_size=1.0, p=0.5)
    print("=== 一维随机游走轨迹 ===")
    print(traj_1d)

    traj_2d = service.simulate_2d(n_steps=10, start=(0, 0), step_size=1.0, mode="lattice")
    print("\n=== 二维格点随机游走轨迹 (x, y) ===")
    print(traj_2d)

    traj_2d_cont = service.simulate_2d(
        n_steps=5, start=(0, 0), step_size=1.0, mode="continuous"
    )
    print("\n=== 二维连续方向随机游走轨迹 (x, y) ===")
    print(traj_2d_cont)

    bounds = (-2.0, 2.0, -2.0, 2.0)
    traj_refl = service.simulate_2d(
        n_steps=30,
        start=(0, 0),
        step_size=1.0,
        mode="lattice",
        bounds=bounds,
        boundary_type="reflecting",
    )
    print("\n=== 反射边界示例 (范围: x∈[-2,2], y∈[-2,2]) ===")
    print(traj_refl)
    print(f"  x 范围: [{traj_refl[:, 0].min():.1f}, {traj_refl[:, 0].max():.1f}]")
    print(f"  y 范围: [{traj_refl[:, 1].min():.1f}, {traj_refl[:, 1].max():.1f}]")

    traj_abs = service.simulate_2d(
        n_steps=30,
        start=(0, 0),
        step_size=1.0,
        mode="lattice",
        bounds=bounds,
        boundary_type="absorbing",
    )
    print("\n=== 吸收边界示例 (范围: x∈[-2,2], y∈[-2,2]) ===")
    print(traj_abs)
    print(f"  x 范围: [{traj_abs[:, 0].min():.1f}, {traj_abs[:, 0].max():.1f}]")
    print(f"  y 范围: [{traj_abs[:, 1].min():.1f}, {traj_abs[:, 1].max():.1f}]")
    absorbed_step = None
    for i in range(1, len(traj_abs)):
        if np.allclose(traj_abs[i], traj_abs[i - 1]):
            absorbed_step = i - 1
            break
    if absorbed_step is not None:
        print(f"  第 {absorbed_step} 步被吸收，之后位置不再变化")

    batch_1d = service.simulate_batch_1d(n_walkers=3, n_steps=5)
    print("\n=== 批量一维随机游走 (3 个游走者, 5 步) ===")
    print(batch_1d)

    batch_2d = service.simulate_batch_2d(
        n_walkers=2, n_steps=3, bounds=bounds, boundary_type="reflecting"
    )
    print("\n=== 批量二维反射边界随机游走 (2 个游走者, 3 步) ===")
    print(batch_2d)


if __name__ == "__main__":
    main()
