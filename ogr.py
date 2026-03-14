import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from typing import Optional, List, Dict, Any, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# параметры задачи
P: float = 4e6
sigma_max: float = 9.5e6
rho: float = 7800
V_min: float = 0.4

def mass(x: np.ndarray) -> float:
    """масса сосуда"""
    R: float
    t: float
    R, t = x
    V_material: float = (4/3) * np.pi * ((R + t)**3 - R**3)
    return rho * V_material

def volume(x: np.ndarray) -> float:
    """внутренний объем"""
    R: float
    R, _ = x
    return (4/3) * np.pi * R**3

def stress(x: np.ndarray) -> float:
    """напряжение в стенке"""
    R: float
    t: float
    R, t = x
    return P * R / (2 * t)

def constraints(x: np.ndarray) -> np.ndarray:
    """вектор ограничений"""
    R: float
    t: float
    R, t = x
    g1: float = V_min - volume(x)
    g2: float = stress(x) - sigma_max
    return np.array([g1, g2])

def check_feasibility(x: np.ndarray, tol: float = 1e-3) -> bool:
    """проверка допустимости"""
    return bool(np.all(constraints(x) <= tol))

def violation(x: np.ndarray) -> float:
    """суммарное нарушение"""
    c: np.ndarray = constraints(x)
    return float(np.sum(np.maximum(0, c)))

# теоретический оптимум
R_teor: float = (V_min * 3/(4*np.pi))**(1/3)
t_teor: float = P * R_teor / (2 * sigma_max)
m_teor: float = mass(np.array([R_teor, t_teor]))

bounds: np.ndarray = np.array([
    [0.4, 0.6],
    [0.05, 0.15]
])

# базовый класс
class BO:
    """байесовская оптимизация"""

    def __init__(self, bounds: np.ndarray, n_init: int = 10) -> None:
        self.bounds: np.ndarray = bounds
        self.n_init: int = n_init
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.c1: List[float] = []
        self.c2: List[float] = []
        self.hist: List[float] = []
        self.first: Optional[int] = None
        self.viol: List[float] = []

        self.gp_f: Optional[GaussianProcessRegressor] = None
        self.gp_c1: Optional[GaussianProcessRegressor] = None
        self.gp_c2: Optional[GaussianProcessRegressor] = None

    def sample(self) -> np.ndarray:
        """лат гиперкуб"""
        lhc: LatinHypercube = LatinHypercube(d=2)
        s: np.ndarray = lhc.random(n=self.n_init)
        X: np.ndarray = np.zeros((self.n_init, 2))
        for i in range(2):
            X[:, i] = s[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        return X

    def init(self) -> None:
        """начальная выборка"""
        X: np.ndarray = self.sample()
        for x in X:
            self.X.append(x)
            self.y.append(mass(x))
            c: np.ndarray = constraints(x)
            self.c1.append(c[0])
            self.c2.append(c[1])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.c1 = np.array(self.c1)
        self.c2 = np.array(self.c2)

    def fit(self) -> None:
        """обучение gp"""
        kernel: C * Matern = C(1.0) * Matern(length_scale=0.1, nu=2.5)
        self.gp_f = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_f.fit(self.X, self.y)
        self.gp_c1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_c1.fit(self.X, self.c1)
        self.gp_c2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_c2.fit(self.X, self.c2)

    def ei(self, x: np.ndarray, y_best: float) -> float:
        """expected improvement"""
        x = x.reshape(1, -1)
        mu: np.ndarray
        s: np.ndarray
        mu, s = self.gp_f.predict(x, return_std=True)  # type: ignore
        s = s.reshape(-1, 1)
        if s < 1e-10:
            return 0.0
        gamma: float = ((y_best - mu) / s)[0, 0]
        return float((s * (gamma * norm.cdf(gamma) + norm.pdf(gamma)))[0, 0])

    def pof(self, x: np.ndarray) -> float:
        """вероятность допустимости"""
        x = x.reshape(1, -1)
        mu1: np.ndarray
        s1: np.ndarray
        mu1, s1 = self.gp_c1.predict(x, return_std=True)  # type: ignore
        s1 = s1.reshape(-1, 1)
        p1: float = float(norm.cdf(-mu1 / (s1 + 1e-10))[0, 0])
        mu2: np.ndarray
        s2: np.ndarray
        mu2, s2 = self.gp_c2.predict(x, return_std=True)  # type: ignore
        s2 = s2.reshape(-1, 1)
        p2: float = float(norm.cdf(-mu2 / (s2 + 1e-10))[0, 0])
        return p1 * p2

    def cei(self, x: np.ndarray, y_best: float) -> float:
        """условное ожидаемое улучшение"""
        return self.ei(x, y_best) * self.pof(x)

    def acq(self, n_iter: int = 100) -> np.ndarray:
        """выбор следующей точки"""
        best_x: Optional[np.ndarray] = None
        best_val: float = -np.inf
        y_best: float = float(np.min(self.y))
        for _ in range(n_iter):
            x: np.ndarray = np.array([
                np.random.uniform(*self.bounds[0]),
                np.random.uniform(*self.bounds[1])
            ])
            val: float = self.cei(x, y_best)
            if val > best_val:
                best_val = val
                best_x = x
        if best_x is None:
            best_x = np.array([
                np.random.uniform(*self.bounds[0]),
                np.random.uniform(*self.bounds[1])
            ])
        return best_x

    def run(self, iters: int = 50) -> Tuple[Optional[np.ndarray], float, List[float]]:
        """запуск оптимизации"""
        self.init()
        best_f: float = float('inf')
        best_x: Optional[np.ndarray] = None
        first: Optional[int] = None

        for t in range(iters):
            self.fit()
            x_next: np.ndarray = self.acq()
            f_next: float = mass(x_next)
            c_next: np.ndarray = constraints(x_next)

            self.X = np.vstack([self.X, x_next])
            self.y = np.append(self.y, f_next)
            self.c1 = np.append(self.c1, c_next[0])
            self.c2 = np.append(self.c2, c_next[1])

            if check_feasibility(x_next):
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()
                if first is None:
                    first = t + self.n_init

            self.hist.append(best_f if best_f != float('inf') else np.nan)
            self.viol.append(violation(x_next))

        self.first = first
        return best_x, best_f, self.hist


# метод 1: безусловный
def run_unconstrained(n_runs: int = 10, n_iter: int = 50) -> List[Dict[str, Any]]:
    """безусловная оптимизация (игнорирует ограничения)"""
    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        class UnconstrainedBO(BO):
            def cei(self, x: np.ndarray, y_best: float) -> float:
                return self.ei(x, y_best)
        b: UnconstrainedBO = UnconstrainedBO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return results


# метод 2: штрафной
def run_penalty(n_runs: int = 10, n_iter: int = 50) -> List[Dict[str, Any]]:
    """штрафной метод"""
    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        np.random.seed(42 + run)

        def objective(x: np.ndarray) -> float:
            c: np.ndarray = constraints(x)
            penalty: float = 1e4 * float(np.sum(np.maximum(0, c)**2))
            return mass(x) + penalty

        class PenaltyBO(BO):
            def __init__(self, bounds: np.ndarray, n_init: int = 10):
                super().__init__(bounds, n_init)
                self.y = []

            def init(self) -> None:
                X: np.ndarray = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(objective(x))
                    c: np.ndarray = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)

            def cei(self, x: np.ndarray, y_best: float) -> float:
                return self.ei(x, y_best)

        b: PenaltyBO = PenaltyBO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return results


# метод 3: CEI
def run_cei(n_runs: int = 10, n_iter: int = 50) -> List[Dict[str, Any]]:
    """cei метод"""
    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        b: BO = BO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return results


# метод 4: лагранж
def run_lagrange(n_runs: int = 10, n_iter: int = 50) -> List[Dict[str, Any]]:
    """метод лагранжа"""
    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        np.random.seed(42 + run)

        def objective(x: np.ndarray) -> float:
            c: np.ndarray = constraints(x)
            L: float = mass(x)
            if c[0] > 0:
                L += 1000 * c[0] + 1000 * c[0]**2
            if c[1] > 0:
                L += 1000 * c[1] + 1000 * c[1]**2
            return L

        class LagrangeBO(BO):
            def __init__(self, bounds: np.ndarray, n_init: int = 10):
                super().__init__(bounds, n_init)
                self.y = []

            def init(self) -> None:
                X: np.ndarray = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(objective(x))
                    c: np.ndarray = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)

            def cei(self, x: np.ndarray, y_best: float) -> float:
                return self.ei(x, y_best)

        b: LagrangeBO = LagrangeBO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return results


# метод 5: барьерный
def run_barrier(n_runs: int = 10, n_iter: int = 50) -> List[Dict[str, Any]]:
    """барьерный метод"""
    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        HUGE: float = 1e10

        def objective(x: np.ndarray) -> float:
            if not check_feasibility(x):
                return HUGE + mass(x)
            c: np.ndarray = constraints(x)
            barrier: float = 0.0
            barrier -= 100 * np.log(-c[0] + 1e-10)
            barrier -= 100 * np.log(-c[1] + 1e-10)
            return mass(x) + barrier

        class BarrierBO(BO):
            def __init__(self, bounds: np.ndarray, n_init: int = 10):
                super().__init__(bounds, n_init)
                self.y = []

            def init(self) -> None:
                X: np.ndarray = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(objective(x))
                    c: np.ndarray = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)

            def cei(self, x: np.ndarray, y_best: float) -> float:
                return self.ei(x, y_best)

        b: BarrierBO = BarrierBO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return results


print("сравнение методов")
print(f"давление {P/1e6} мпа, напр {sigma_max/1e6} мпа, объем {V_min} м3")
print(f"теор масса {m_teor:.0f} кг")
print("10 запусков по 50 итер, нач выборка 10\n")

methods: Dict[str, Callable] = {
    "безусл": run_unconstrained,
    "штраф": run_penalty,
    "cei": run_cei,
    "лагранж": run_lagrange,
    "барьер": run_barrier
}

all_res: Dict[str, List[Dict[str, Any]]] = {}

for name, func in methods.items():
    print(f">{name}")
    r: List[Dict[str, Any]] = func(n_runs=10, n_iter=50)
    all_res[name] = r

    ok: List[Dict[str, Any]] = [x for x in r if x['ok']]
    cnt: int = len(ok)

    if cnt > 0:
        m: List[float] = [x['f'] for x in ok]
        fr: List[int] = [x['first'] for x in ok if x['first'] is not None]
        print(f"  ok {cnt}/10")
        print(f"  ср масса {np.mean(m):.0f} кг")
        print(f"  стд {np.std(m):.0f} кг")
        print(f"  мин {np.min(m):.0f} кг")
        print(f"  макс {np.max(m):.0f} кг")
        if fr:
            print(f"  первое {np.mean(fr):.1f} итер")
    else:
        print(f"  ok 0/10")
    print()

print("\n" + "="*60)
print("сводка")
print("="*60)

table_data: List[Dict[str, str]] = []
for name, r in all_res.items():
    ok: List[Dict[str, Any]] = [x for x in r if x['ok']]
    m: List[float] = [x['f'] for x in ok]
    fr: List[int] = [x['first'] for x in ok if x['first'] is not None]
    table_data.append({
        "метод": name,
        "ok": f"{len(ok)}/10",
        "ср масса": f"{np.mean(m):.0f}" if m else "-",
        "стд": f"{np.std(m):.0f}" if m else "-",
        "мин": f"{np.min(m):.0f}" if m else "-",
        "первое": f"{np.mean(fr):.1f}" if fr else "-"
    })

print(pd.DataFrame(table_data).to_string(index=False))

# графики
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# график сходимости
for name, r in all_res.items():
    histories: List[List[float]] = []
    for x in r:
        if any(not np.isnan(y) for y in x['hist']):
            histories.append(x['hist'][:50])
    if histories:
        arr: np.ndarray = np.array(histories)
        mu: np.ndarray = np.nanmean(arr, axis=0)
        s: np.ndarray = np.nanstd(arr, axis=0)
        ax[0,0].plot(range(len(mu)), mu, label=name)
        ax[0,0].fill_between(range(len(mu)), mu-s, mu+s, alpha=0.2)

ax[0,0].axhline(y=m_teor, color='k', ls='--')
ax[0,0].set_xlabel("итер")
ax[0,0].set_ylabel("масса")
ax[0,0].legend()
ax[0,0].grid()

# столбцы масса
names_list: List[str] = []
means_list: List[float] = []
stds_list: List[float] = []
colors_list: List[str] = []

for name, r in all_res.items():
    ok = [x for x in r if x['ok']]
    names_list.append(name)
    if ok:
        m_vals = [x['f'] for x in ok]
        means_list.append(np.mean(m_vals))
        stds_list.append(np.std(m_vals))
        colors_list.append('g')
    else:
        means_list.append(0)
        stds_list.append(0)
        colors_list.append('r')

ax[0,1].bar(names_list, means_list, yerr=stds_list, capsize=5, color=colors_list)
ax[0,1].axhline(y=m_teor, color='k', ls='--')
ax[0,1].set_ylabel("масса")
ax[0,1].tick_params(axis='x', rotation=30)
ax[0,1].grid(axis='y')

# первая допустимая
first_data: List[Dict[str, Any]] = []
for name, r in all_res.items():
    times = [x['first'] for x in r if x['first'] is not None]
    if times:
        first_data.append({'n': name, 'm': np.mean(times), 's': np.std(times)})

if first_data:
    first_names = [x['n'] for x in first_data]
    first_means = [x['m'] for x in first_data]
    first_stds = [x['s'] for x in first_data]
    ax[1,0].bar(first_names, first_means, yerr=first_stds, capsize=5, color='b')
    ax[1,0].set_ylabel("итер")
    ax[1,0].tick_params(axis='x', rotation=30)
    ax[1,0].grid(axis='y')

# процент успеха
percent: List[float] = [len([x for x in r if x['ok']]) * 10 for r in all_res.values()]
ax[1,1].bar(names_list, percent, color='purple')
ax[1,1].set_ylabel("%")
ax[1,1].tick_params(axis='x', rotation=30)
ax[1,1].set_ylim([0,100])
ax[1,1].grid(axis='y')

plt.tight_layout()
plt.show()
