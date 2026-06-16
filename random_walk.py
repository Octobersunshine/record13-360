import numpy as np
from typing import Tuple, Optional, Union


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

        返回:
            二维轨迹数组，形状 (n_steps + 1, 2)，每行为 (x, y)
        """
        if n_steps < 0:
            raise ValueError("n_steps 必须是非负整数")

        if mode == "lattice":
            directions = self._rng.integers(0, 4, size=n_steps)
            dx = np.zeros(n_steps)
            dy = np.zeros(n_steps)
            dx[directions == 0] = step_size
            dx[directions == 1] = -step_size
            dy[directions == 2] = step_size
            dy[directions == 3] = -step_size
        elif mode == "continuous":
            angles = self._rng.uniform(0, 2 * np.pi, size=n_steps)
            dx = step_size * np.cos(angles)
            dy = step_size * np.sin(angles)
        else:
            raise ValueError("mode 必须是 'lattice' 或 'continuous'")

        x = np.concatenate([[start[0]], np.cumsum(dx)])
        y = np.concatenate([[start[1]], np.cumsum(dy)])
        return np.column_stack([x, y])

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
    ) -> np.ndarray:
        """
        批量二维随机游走模拟

        参数:
            n_walkers: 游走者数量
            n_steps: 游走步数
            start: 起始坐标（二元组或形状为 (n_walkers, 2) 的数组）
            step_size: 每步步长
            mode: 游走模式 "lattice" 或 "continuous"

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
                self.simulate_2d(n_steps, s, step_size, mode)
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

    batch_1d = service.simulate_batch_1d(n_walkers=3, n_steps=5)
    print("\n=== 批量一维随机游走 (3 个游走者, 5 步) ===")
    print(batch_1d)

    batch_2d = service.simulate_batch_2d(n_walkers=2, n_steps=3)
    print("\n=== 批量二维随机游走 (2 个游走者, 3 步) ===")
    print(batch_2d)


if __name__ == "__main__":
    main()
