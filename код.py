import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Задача
# ------------------------------------------------------------
P = 4e6
sigma_max = 9.5e6
rho = 7800
V_min = 0.4

def mass(x):
    R, t = x
    return rho * (4/3 * np.pi * ((R + t)**3 - R**3))

def volume(x):
    R, _ = x
    return 4/3 * np.pi * R**3

def stress(x):
    R, t = x
    return P * R / (2 * t)

def constraints(x):
    g1 = V_min - volume(x)
    g2 = stress(x) - sigma_max
    return np.array([g1, g2])

def is_ok(x):
    return np.all(constraints(x) <= 1e-6)

def how_bad(x):
    return np.sum(np.maximum(0, constraints(x)))

R_star = (V_min * 3/(4*np.pi))**(1/3)
t_star = P * R_star / (2 * sigma_max)
m_star = mass([R_star, t_star])

bounds = np.array([[0.4, 0.6], [0.05, 0.15]])


# ------------------------------------------------------------
# Байесовская оптимизация
# ------------------------------------------------------------
class BO:
    def __init__(self, bounds, n_start=10):
        self.bounds = bounds
        self.n_start = n_start

        self.X = []
        self.F = []
        self.G1 = []
        self.G2 = []

        self.best_m = 1e9
        self.best_x = None
        self.first_good = None
        self.track = []
        self.viol = []

        self.gp_f = None
        self.gp_g1 = None
        self.gp_g2 = None

    def lhs_sample(self, n):
        sampler = LatinHypercube(d=2)
        s = sampler.random(n=n)
        X = np.zeros((n, 2))
        for i in range(2):
            X[:, i] = s[:, i] * (self.bounds[i,1] - self.bounds[i,0]) + self.bounds[i,0]
        return X

    def init_data(self):
        X0 = self.lhs_sample(self.n_start)

        for x in X0:
            self.X.append(x)
            self.F.append(mass(x))
            g = constraints(x)
            self.G1.append(g[0])
            self.G2.append(g[1])
            self.viol.append(how_bad(x))

            if is_ok(x):
                if mass(x) < self.best_m:
                    self.best_m = mass(x)
                    self.best_x = x.copy()
                if self.first_good is None:
                    self.first_good = len(self.X) - 1

        self.X = np.array(self.X)
        self.F = np.array(self.F)
        self.G1 = np.array(self.G1)
        self.G2 = np.array(self.G2)

        if self.best_m < 1e9:
            self.track.append(self.best_m)
        else:
            self.track.append(np.nan)

    def train_gp(self):
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.1, nu=2.5) + WhiteKernel(1e-6)

        self.gp_f = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_f.fit(self.X.reshape(-1,2), self.F)

        self.gp_g1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_g1.fit(self.X.reshape(-1,2), self.G1)

        self.gp_g2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_g2.fit(self.X.reshape(-1,2), self.G2)

    def ei(self, x, best_val):
        x = x.reshape(1,-1)
        mu, sigma = self.gp_f.predict(x, return_std=True)
        sigma = sigma[0]

        if sigma < 1e-10:
            return 0.0

        gamma = (best_val - mu) / sigma
        return sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

    def p_ok(self, x):
        x = x.reshape(1,-1)
        mu1, s1 = self.gp_g1.predict(x, return_std=True)
        mu2, s2 = self.gp_g2.predict(x, return_std=True)

        p1 = norm.cdf(-mu1 / (s1[0] + 1e-8))
        p2 = norm.cdf(-mu2 / (s2[0] + 1e-8))
        return p1 * p2

    def cei(self, x, best_val):
        return self.ei(x, best_val) * self.p_ok(x)

    def next_point(self, n_tries=30):
        best_x = None
        best_val = -1e9
        current_best = self.best_m if self.best_m < 1e9 else 1e9

        for _ in range(n_tries):
            x0 = np.array([
                np.random.uniform(*self.bounds[0]),
                np.random.uniform(*self.bounds[1])
            ])

            try:
                res = minimize(
                    lambda x: -self.cei(x, current_best),
                    x0,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 100}
                )

                if res.success and -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x
            except:
                continue

        if best_x is None:
            best_x = self.lhs_sample(1)[0]

        return best_x

    def run(self, steps=50):
        self.init_data()

        for _ in range(steps):
            self.train_gp()
            x_new = self.next_point()

            f_new = mass(x_new)
            g_new = constraints(x_new)

            self.X = np.vstack([self.X, x_new])
            self.F = np.append(self.F, f_new)
            self.G1 = np.append(self.G1, g_new[0])
            self.G2 = np.append(self.G2, g_new[1])
            self.viol.append(how_bad(x_new))

            if is_ok(x_new) and f_new < self.best_m:
                self.best_m = f_new
                self.best_x = x_new.copy()
                if self.first_good is None:
                    self.first_good = len(self.X) - 1

            self.track.append(self.best_m if self.best_m < 1e9 else np.nan)

        return self.best_x, self.best_m, self.track


# ------------------------------------------------------------
# Методы
# ------------------------------------------------------------
def run_ignore(n_runs=10, n_iter=50):
    results = []
    for i in range(n_runs):
        np.random.seed(42 + i)

        class IgnoreBO(BO):
            def cei(self, x, best_val):
                return self.ei(x, best_val)

        opt = IgnoreBO(bounds, n_start=10)
        x, m, hist = opt.run(n_iter)
        results.append({
            'run': i, 'x': x, 'mass': m if x is not None else np.nan,
            'ok': is_ok(x) if x is not None else False,
            'hist': hist, 'first': opt.first_good
        })
    return results

def run_penalty(n_runs=10, n_iter=50):
    results = []
    for i in range(n_runs):
        np.random.seed(42 + i)

        def penalty_func(x):
            return mass(x) + 1e4 * how_bad(x)**2

        class PenaltyBO(BO):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.F = []

            def init_data(self):
                X0 = self.lhs_sample(self.n_start)
                for x in X0:
                    self.X.append(x)
                    self.F.append(penalty_func(x))
                    g = constraints(x)
                    self.G1.append(g[0])
                    self.G2.append(g[1])
                    self.viol.append(how_bad(x))

                    if is_ok(x):
                        if mass(x) < self.best_m:
                            self.best_m = mass(x)
                            self.best_x = x.copy()
                        if self.first_good is None:
                            self.first_good = len(self.X) - 1

                self.X = np.array(self.X)
                self.F = np.array(self.F)
                self.G1 = np.array(self.G1)
                self.G2 = np.array(self.G2)
                self.track.append(self.best_m if self.best_m < 1e9 else np.nan)

        opt = PenaltyBO(bounds, n_start=10)
        x, m, hist = opt.run(n_iter)
        results.append({
            'run': i, 'x': x, 'mass': mass(x) if x is not None else np.nan,
            'ok': is_ok(x) if x is not None else False,
            'hist': hist, 'first': opt.first_good
        })
    return results

def run_cei(n_runs=10, n_iter=50):
    results = []
    for i in range(n_runs):
        np.random.seed(42 + i)
        opt = BO(bounds, n_start=10)
        x, m, hist = opt.run(n_iter)
        results.append({
            'run': i, 'x': x, 'mass': m if x is not None else np.nan,
            'ok': is_ok(x) if x is not None else False,
            'hist': hist, 'first': opt.first_good
        })
    return results

def run_lagrange(n_runs=10, n_iter=50):
    results = []
    for i in range(n_runs):
        np.random.seed(42 + i)

        def lagrange(x):
            g = constraints(x)
            L = mass(x)
            if g[0] > 0:
                L += 1000 * g[0] + 500 * g[0]**2
            if g[1] > 0:
                L += 1000 * g[1] + 500 * g[1]**2
            return L

        class LagrangeBO(BO):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.F = []

            def init_data(self):
                X0 = self.lhs_sample(self.n_start)
                for x in X0:
                    self.X.append(x)
                    self.F.append(lagrange(x))
                    g = constraints(x)
                    self.G1.append(g[0])
                    self.G2.append(g[1])
                    self.viol.append(how_bad(x))

                    if is_ok(x):
                        if mass(x) < self.best_m:
                            self.best_m = mass(x)
                            self.best_x = x.copy()
                        if self.first_good is None:
                            self.first_good = len(self.X) - 1

                self.X = np.array(self.X)
                self.F = np.array(self.F)
                self.G1 = np.array(self.G1)
                self.G2 = np.array(self.G2)
                self.track.append(self.best_m if self.best_m < 1e9 else np.nan)

        opt = LagrangeBO(bounds, n_start=10)
        x, m, hist = opt.run(n_iter)
        results.append({
            'run': i, 'x': x, 'mass': mass(x) if x is not None else np.nan,
            'ok': is_ok(x) if x is not None else False,
            'hist': hist, 'first': opt.first_good
        })
    return results

def run_barrier(n_runs=10, n_iter=50):
    results = []
    for i in range(n_runs):
        np.random.seed(42 + i)

        def barrier(x):
            if not is_ok(x):
                return 1e10
            g = constraints(x)
            return mass(x) - 100 * np.log(-g[0] + 1e-8) - 100 * np.log(-g[1] + 1e-8)

        class BarrierBO(BO):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.F = []

            def init_data(self):
                X0 = self.lhs_sample(self.n_start)
                for x in X0:
                    self.X.append(x)
                    self.F.append(barrier(x))
                    g = constraints(x)
                    self.G1.append(g[0])
                    self.G2.append(g[1])
                    self.viol.append(how_bad(x))

                    if is_ok(x):
                        if mass(x) < self.best_m:
                            self.best_m = mass(x)
                            self.best_x = x.copy()
                        if self.first_good is None:
                            self.first_good = len(self.X) - 1

                self.X = np.array(self.X)
                self.F = np.array(self.F)
                self.G1 = np.array(self.G1)
                self.G2 = np.array(self.G2)
                self.track.append(self.best_m if self.best_m < 1e9 else np.nan)

        opt = BarrierBO(bounds, n_start=10)
        x, m, hist = opt.run(n_iter)
        results.append({
            'run': i, 'x': x, 'mass': mass(x) if x is not None else np.nan,
            'ok': is_ok(x) if x is not None else False,
            'hist': hist, 'first': opt.first_good
        })
    return results


# ------------------------------------------------------------
# Запуск
# ------------------------------------------------------------
print("="*70)
print("СРАВНЕНИЕ МЕТОДОВ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ")
print("="*70)
print(f"Давление: {P/1e6} МПа")
print(f"Напряжение: {sigma_max/1e6} МПа")
print(f"Объем: {V_min} м3")
print(f"Теория: масса = {m_star:.0f} кг, R = {R_star:.3f} м, t = {t_star:.4f} м")
print("="*70)

methods = {
    "Без ограничений": run_ignore,
    "Штрафной": run_penalty,
    "CEI": run_cei,
    "Лагранж": run_lagrange,
    "Барьерный": run_barrier
}

all_res = {}

for name, func in methods.items():
    print(f"\n{name}:")
    res = func(n_runs=10, n_iter=50)
    all_res[name] = res

    ok = [r for r in res if r['ok']]
    print(f"  Успех: {len(ok)}/10")

    if ok:
        m = [r['mass'] for r in ok]
        print(f"  Масса: {np.mean(m):.0f} ± {np.std(m):.0f} кг")
        print(f"  Лучшая: {np.min(m):.0f} кг")
        print(f"  От теории: {(np.mean(m)-m_star)/m_star*100:+.1f}%")

        first = [r['first'] for r in ok if r['first'] is not None]
        if first:
            print(f"  Первое: {np.mean(first):.1f} итер")


# ------------------------------------------------------------
# Графики
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Сравнение методов байесовской оптимизации', fontsize=14, fontweight='bold')

# 1. Сходимость
ax = axes[0,0]
for name, res in all_res.items():
    hist = []
    for r in res:
        if r['hist']:
            hist.append(r['hist'][:51])
    if hist:
        h = np.array(hist)
        mean = np.nanmean(h, axis=0)
        std = np.nanstd(h, axis=0)
        ax.plot(mean, label=name, lw=2)
        ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
ax.axhline(m_star, color='k', ls='--', label='Теория')
ax.set_xlabel('Итерация')
ax.set_ylabel('Масса, кг')
ax.set_title('Сходимость')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 2. Boxplot
ax = axes[0,1]
data = []
labels = []
for name, res in all_res.items():
    m = [r['mass'] for r in res if r['ok']]
    if m:
        data.append(m)
        labels.append(name)
bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
ax.axhline(m_star, color='r', ls='--', label='Теория')
ax.set_ylabel('Масса, кг')
ax.set_title('Распределение масс')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 3. Успешность
ax = axes[0,2]
success = [len([r for r in res if r['ok']]) * 10 for res in all_res.values()]
colors = ['green' if x >= 80 else 'orange' if x >= 50 else 'red' for x in success]
bars = ax.bar(labels, success, color=colors)
ax.set_ylabel('Успешность, %')
ax.set_title('Доля допустимых решений')
ax.set_ylim(0, 110)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, success):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom')

# 4. Скорость
ax = axes[1,0]
first_iters = []
for name, res in all_res.items():
    times = [r['first'] for r in res if r['first'] is not None]
    first_iters.append(np.mean(times) if times else 0)
bars = ax.bar(labels, first_iters, color='skyblue')
ax.set_ylabel('Итерация')
ax.set_title('Первое допустимое решение')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, first_iters):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')

# 5. Отклонения
ax = axes[1,1]
for name, res in all_res.items():
    m = [r['mass'] for r in res if r['ok']]
    if m:
        dev = [(x - m_star)/m_star*100 for x in m]
        ax.hist(dev, alpha=0.5, bins=15, label=name)
ax.axvline(0, color='r', ls='--', label='Оптимум')
ax.set_xlabel('Отклонение, %')
ax.set_ylabel('Частота')
ax.set_title('Отклонение от теории')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6. Лучшие результаты
ax = axes[1,2]
best = []
for name, res in all_res.items():
    m = [r['mass'] for r in res if r['ok']]
    best.append(np.min(m) if m else np.nan)
bars = ax.bar(labels, best, color='darkorange')
ax.axhline(m_star, color='g', ls='--', label='Теория')
ax.set_ylabel('Масса, кг')
ax.set_title('Лучшее решение')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, best):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Итоговая таблица
# ------------------------------------------------------------
print("\n" + "="*70)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*70)

summary = []
for name, res in all_res.items():
    ok = [r for r in res if r['ok']]
    if ok:
        m = [r['mass'] for r in ok]
        times = [r['first'] for r in ok if r['first'] is not None]
        summary.append({
            'Метод': name,
            'Успех': f"{len(ok)}/10",
            'Ср. масса': f"{np.mean(m):.0f}",
            'Стд': f"{np.std(m):.0f}",
            'Лучшая': f"{np.min(m):.0f}",
            'Откл.': f"{(np.mean(m)-m_star)/m_star*100:+.1f}%",
            'Первая': f"{np.mean(times):.1f}" if times else "-"
        })

df = pd.DataFrame(summary)
print(df.to_string(index=False))

print("\n" + "="*70)
