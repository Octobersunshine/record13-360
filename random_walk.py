import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Callable


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

    @staticmethod
    def first_passage_time_1d(
        trajectory: np.ndarray,
        target: float,
        direction: str = "any",
        include_start: bool = False,
        tol: float = 1e-9,
    ) -> Optional[int]:
        """
        计算一维轨迹的首达时（首次到达目标位置的步数）

        参数:
            trajectory: 一维轨迹数组，形状 (n_steps + 1,)
            target: 目标位置
            direction: 到达方向
                - "any": 任意方向（默认）
                - "above": 从下方到达（位置从 < target 变为 >= target）
                - "below": 从上方到达（位置从 > target 变为 <= target）
            include_start: 若起始位置已满足条件，是否返回 0
            tol: 数值比较容差

        返回:
            首次到达的步数（0-based），若未到达则返回 None
        """
        if trajectory.ndim != 1:
            raise ValueError("trajectory 必须是一维数组")

        if direction == "any":
            reached = np.abs(trajectory - target) <= tol
        elif direction == "above":
            reached = trajectory >= target - tol
        elif direction == "below":
            reached = trajectory <= target + tol
        else:
            raise ValueError("direction 必须是 'any', 'above' 或 'below'")

        start_idx = 0 if include_start else 1
        idx = np.where(reached[start_idx:])[0]
        if len(idx) == 0:
            return None
        return int(idx[0] + start_idx)

    @staticmethod
    def first_passage_time_2d(
        trajectory: np.ndarray,
        target: Union[Tuple[float, float], Tuple[float, float, float, float], Callable],
        tol: float = 1e-9,
        include_start: bool = False,
    ) -> Optional[int]:
        """
        计算二维轨迹的首达时

        参数:
            trajectory: 二维轨迹数组，形状 (n_steps + 1, 2)
            target: 目标，支持三种形式：
                - (x, y): 目标点（精确到达，考虑容差 tol）
                - (x_min, x_max, y_min, y_max): 目标矩形区域
                - callable(x, y) -> bool: 自定义判定函数
            tol: 目标点判定的数值容差
            include_start: 若起始位置已满足条件，是否返回 0

        返回:
            首次到达的步数（0-based），若未到达则返回 None
        """
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError("trajectory 必须是形状为 (N, 2) 的二维数组")

        xs = trajectory[:, 0]
        ys = trajectory[:, 1]

        if callable(target):
            reached = np.array([bool(target(x, y)) for x, y in zip(xs, ys)])
        elif len(target) == 2:
            tx, ty = target
            reached = (np.abs(xs - tx) <= tol) & (np.abs(ys - ty) <= tol)
        elif len(target) == 4:
            x_min, x_max, y_min, y_max = target
            reached = (xs >= x_min - tol) & (xs <= x_max + tol) & \
                      (ys >= y_min - tol) & (ys <= y_max + tol)
        else:
            raise ValueError("target 必须是二元组、四元组或可调用对象")

        start_idx = 0 if include_start else 1
        idx = np.where(reached[start_idx:])[0]
        if len(idx) == 0:
            return None
        return int(idx[0] + start_idx)

    def analyze_first_passage_1d(
        self,
        n_walkers: int,
        target: float,
        max_steps: int = 10000,
        start: float = 0.0,
        step_size: float = 1.0,
        p: float = 0.5,
        direction: str = "any",
        include_start: bool = False,
    ) -> Dict:
        """
        批量一维首达时统计分析

        参数:
            n_walkers: 游走者数量
            target: 目标位置
            max_steps: 最大模拟步数
            start: 起始位置
            step_size: 每步步长
            p: 向右走的概率
            direction: 到达方向 ("any", "above", "below")
            include_start: 起始位置是否计为已到达

        返回:
            包含统计信息的字典:
                - "fpt_list": 所有游走者的首达时列表（未到达者为 None）
                - "reached_count": 成功到达的游走者数
                - "reached_ratio": 到达比例
                - "mean": 到达者的平均首达时
                - "std": 到达者的首达时标准差
                - "min": 最短首达时
                - "max": 最长首达时
                - "median": 首达时中位数
        """
        trajectories = self.simulate_batch_1d(
            n_walkers, max_steps, start, step_size, p
        )

        fpt_list: List[Optional[int]] = []
        for traj in trajectories:
            fpt = self.first_passage_time_1d(
                traj, target, direction, include_start
            )
            fpt_list.append(fpt)

        reached = [t for t in fpt_list if t is not None]
        reached_arr = np.array(reached, dtype=float)

        result: Dict = {
            "fpt_list": fpt_list,
            "reached_count": len(reached),
            "reached_ratio": len(reached) / n_walkers,
        }
        if len(reached) > 0:
            result.update({
                "mean": float(np.mean(reached_arr)),
                "std": float(np.std(reached_arr)),
                "min": int(np.min(reached_arr)),
                "max": int(np.max(reached_arr)),
                "median": float(np.median(reached_arr)),
            })
        else:
            result.update({
                "mean": None, "std": None,
                "min": None, "max": None, "median": None,
            })
        return result

    def analyze_first_passage_2d(
        self,
        n_walkers: int,
        target: Union[Tuple[float, float], Tuple[float, float, float, float], Callable],
        max_steps: int = 10000,
        start: Tuple[float, float] = (0.0, 0.0),
        step_size: float = 1.0,
        mode: str = "lattice",
        bounds: Optional[Tuple[float, float, float, float]] = None,
        boundary_type: str = "none",
        tol: float = 1e-9,
        include_start: bool = False,
    ) -> Dict:
        """
        批量二维首达时统计分析

        参数:
            n_walkers: 游走者数量
            target: 目标（目标点/目标区域/自定义函数）
            max_steps: 最大模拟步数
            start: 起始坐标
            step_size: 每步步长
            mode: 游走模式 "lattice" 或 "continuous"
            bounds: 边界范围
            boundary_type: 边界处理类型
            tol: 目标点判定容差
            include_start: 起始位置是否计为已到达

        返回:
            与 analyze_first_passage_1d 相同结构的统计字典
        """
        trajectories = self.simulate_batch_2d(
            n_walkers, max_steps, start, step_size, mode, bounds, boundary_type
        )

        fpt_list: List[Optional[int]] = []
        for traj in trajectories:
            fpt = self.first_passage_time_2d(
                traj, target, tol, include_start
            )
            fpt_list.append(fpt)

        reached = [t for t in fpt_list if t is not None]
        reached_arr = np.array(reached, dtype=float)

        result: Dict = {
            "fpt_list": fpt_list,
            "reached_count": len(reached),
            "reached_ratio": len(reached) / n_walkers,
        }
        if len(reached) > 0:
            result.update({
                "mean": float(np.mean(reached_arr)),
                "std": float(np.std(reached_arr)),
                "min": int(np.min(reached_arr)),
                "max": int(np.max(reached_arr)),
                "median": float(np.median(reached_arr)),
            })
        else:
            result.update({
                "mean": None, "std": None,
                "min": None, "max": None, "median": None,
            })
        return result


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

    print("\n" + "=" * 60)
    print("首达时分析示例")
    print("=" * 60)

    traj_1d_long = service.simulate_1d(n_steps=500, start=0.0, step_size=1.0, p=0.5)
    fpt_1d = RandomWalkService.first_passage_time_1d(
        traj_1d_long, target=5.0, direction="above"
    )
    print(f"\n=== 一维首达时 (首次到达 x=5) ===")
    print(f"  首达时: {fpt_1d} 步")
    if fpt_1d is not None:
        print(f"  轨迹第 {fpt_1d} 步位置: {traj_1d_long[fpt_1d]:.1f}")
    else:
        print("  (500 步内未到达)")

    traj_2d_long = service.simulate_2d(
        n_steps=1000, start=(0, 0), step_size=1.0, mode="lattice"
    )
    target_region = (3.0, 5.0, 3.0, 5.0)
    fpt_2d_region = RandomWalkService.first_passage_time_2d(
        traj_2d_long, target=target_region
    )
    print(f"\n=== 二维首达时 (首次进入矩形区域 x∈[3,5], y∈[3,5]) ===")
    print(f"  首达时: {fpt_2d_region} 步")
    if fpt_2d_region is not None:
        pos = traj_2d_long[fpt_2d_region]
        print(f"  轨迹第 {fpt_2d_region} 步位置: ({pos[0]:.1f}, {pos[1]:.1f})")
    else:
        print("  (1000 步内未到达)")

    def inside_circle(x, y):
        return (x - 3.0) ** 2 + (y - 3.0) ** 2 <= 2.0 ** 2

    fpt_2d_circle = RandomWalkService.first_passage_time_2d(
        traj_2d_long, target=inside_circle
    )
    print(f"\n=== 二维首达时 (首次进入圆心 (3,3) 半径 2 的圆) ===")
    print(f"  首达时: {fpt_2d_circle} 步")
    if fpt_2d_circle is not None:
        pos = traj_2d_long[fpt_2d_circle]
        dist = np.sqrt((pos[0] - 3) ** 2 + (pos[1] - 3) ** 2)
        print(f"  轨迹第 {fpt_2d_circle} 步位置: ({pos[0]:.1f}, {pos[1]:.1f}), 距圆心: {dist:.2f}")
    else:
        print("  (1000 步内未到达)")

    print("\n=== 一维批量首达时统计 (200 个游走者, 目标 x=5) ===")
    stats_1d = service.analyze_first_passage_1d(
        n_walkers=200, target=5.0, max_steps=500, p=0.5, direction="above"
    )
    print(f"  到达数/总数: {stats_1d['reached_count']}/{200}")
    print(f"  到达比例: {stats_1d['reached_ratio']:.2%}")
    if stats_1d["mean"] is not None:
        print(f"  平均首达时: {stats_1d['mean']:.1f} ± {stats_1d['std']:.1f} 步")
        print(f"  最短/最长: {stats_1d['min']} / {stats_1d['max']} 步")
        print(f"  中位数: {stats_1d['median']:.1f} 步")

    print("\n=== 二维批量首达时统计 (200 个游走者, 目标区域 x∈[3,5], y∈[3,5]) ===")
    stats_2d = service.analyze_first_passage_2d(
        n_walkers=200,
        target=(3.0, 5.0, 3.0, 5.0),
        max_steps=1000,
        mode="lattice",
    )
    print(f"  到达数/总数: {stats_2d['reached_count']}/{200}")
    print(f"  到达比例: {stats_2d['reached_ratio']:.2%}")
    if stats_2d["mean"] is not None:
        print(f"  平均首达时: {stats_2d['mean']:.1f} ± {stats_2d['std']:.1f} 步")
        print(f"  最短/最长: {stats_2d['min']} / {stats_2d['max']} 步")
        print(f"  中位数: {stats_2d['median']:.1f} 步")


if __name__ == "__main__":
    main()
