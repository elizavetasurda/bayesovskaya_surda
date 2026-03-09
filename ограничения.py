import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

P = 4e6
sigma_max = 9.5e6
rho = 7800
V_min = 0.4

def mass(x):
    R, t = x
    V_material = (4/3) * np.pi * ((R + t)**3 - R**3)
    return rho * V_material

def volume(x):
    R, _ = x
    return (4/3) * np.pi * R**3

def stress(x):
    R, t = x
    return P * R / (2 * t)

def constraints(x):
    R, t = x
    g1 = V_min - volume(x)
    g2 = stress(x) - sigma_max
    return np.array([g1, g2])

def check_feasibility(x, tol=1e-3):
    return np.all(constraints(x) <= tol)

def violation(x):
    c = constraints(x)
    return np.sum(np.maximum(0, c))

R_teor = (V_min * 3/(4*np.pi))**(1/3)
t_teor = P * R_teor / (2 * sigma_max)
m_teor = mass([R_teor, t_teor])

bounds = np.array([
    [0.4, 0.6],
    [0.05, 0.15]
])

class BO:
    def __init__(self, bounds, n_init=10):
        self.bounds = bounds
        self.n_init = n_init
        self.X = []
        self.y = []
        self.c1 = []
        self.c2 = []
        self.hist = []
        self.first = None
        self.viol = []

    def sample(self):
        lhc = LatinHypercube(d=2)
        s = lhc.random(n=self.n_init)
        X = np.zeros((self.n_init, 2))
        for i in range(2):
            X[:, i] = s[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        return X

    def init(self):
        X = self.sample()
        for x in X:
            self.X.append(x)
            self.y.append(mass(x))
            c = constraints(x)
            self.c1.append(c[0])
            self.c2.append(c[1])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.c1 = np.array(self.c1)
        self.c2 = np.array(self.c2)

    def fit(self):
        kernel = C(1.0) * Matern(length_scale=0.1, nu=2.5)
        self.gp_f = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_f.fit(self.X, self.y)
        self.gp_c1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_c1.fit(self.X, self.c1)
        self.gp_c2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_c2.fit(self.X, self.c2)

    def ei(self, x, y_best):
        x = x.reshape(1, -1)
        mu, s = self.gp_f.predict(x, return_std=True)
        s = s.reshape(-1, 1)
        if s < 1e-10:
            return 0.0
        gamma = (y_best - mu) / s
        return (s * (gamma * norm.cdf(gamma) + norm.pdf(gamma)))[0, 0]

    def pof(self, x):
        x = x.reshape(1, -1)
        mu1, s1 = self.gp_c1.predict(x, return_std=True)
        s1 = s1.reshape(-1, 1)
        p1 = norm.cdf(-mu1 / (s1 + 1e-10))[0, 0]
        mu2, s2 = self.gp_c2.predict(x, return_std=True)
        s2 = s2.reshape(-1, 1)
        p2 = norm.cdf(-mu2 / (s2 + 1e-10))[0, 0]
        return p1 * p2

    def cei(self, x, y_best):
        return self.ei(x, y_best) * self.pof(x)

    def acq(self, n_iter=100):
        best_x = None
        best_val = -np.inf
        y_best = np.min(self.y)
        for _ in range(n_iter):
            x = np.array([
                np.random.uniform(*self.bounds[0]),
                np.random.uniform(*self.bounds[1])
            ])
            val = self.cei(x, y_best)
            if val > best_val:
                best_val = val
                best_x = x
        return best_x

    def run(self, iters=50):
        self.init()
        best_f = float('inf')
        best_x = None
        first = None

        for t in range(iters):
            self.fit()
            x_next = self.acq()
            f_next = mass(x_next)
            c_next = constraints(x_next)

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

def run1(n_runs=10, n_iter=50):
    res = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        class Tmp(BO):
            def cei(self, x, y_best):
                return self.ei(x, y_best)
        b = Tmp(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        res.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return res

def run2(n_runs=10, n_iter=50):
    res = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        def obj(x):
            c = constraints(x)
            p = 1e4 * np.sum(np.maximum(0, c)**2)
            return mass(x) + p

        class Tmp(BO):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init(self):
                X = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        b = Tmp(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        res.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return res

def run3(n_runs=10, n_iter=50):
    res = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        b = BO(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        res.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return res

def run4(n_runs=10, n_iter=50):
    res = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        def obj(x):
            c = constraints(x)
            L = mass(x)
            if c[0] > 0:
                L += 1000 * c[0] + 1000 * c[0]**2
            if c[1] > 0:
                L += 1000 * c[1] + 1000 * c[1]**2
            return L

        class Tmp(BO):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init(self):
                X = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        b = Tmp(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        res.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return res

def run5(n_runs=10, n_iter=50):
    res = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        HUGE = 1e10
        def obj(x):
            if not check_feasibility(x):
                return HUGE + mass(x)
            c = constraints(x)
            b = 0
            b -= 100 * np.log(-c[0] + 1e-10)
            b -= 100 * np.log(-c[1] + 1e-10)
            return mass(x) + b

        class Tmp(BO):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init(self):
                X = self.sample()
                for x in X:
                    self.X.append(x)
                    self.y.append(obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        b = Tmp(bounds, n_init=10)
        x, f, h = b.run(n_iter)
        res.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'ok': check_feasibility(x) if x is not None else False,
            'hist': h,
            'first': b.first
        })
    return res

print("сравнение методов")
print(f"давление {P/1e6} мпа, напр {sigma_max/1e6} мпа, объем {V_min} м3")
print(f"теор масса {m_teor:.0f} кг")
print("10 запусков по 50 итер, нач выборка 10\n")

methods = {
    "безусл": run1,
    "штраф": run2,
    "cei": run3,
    "лагранж": run4,
    "барьер": run5
}

all_res = {}

for name, f in methods.items():
    print(f">{name}")
    r = f(n_runs=10, n_iter=50)
    all_res[name] = r

    ok = [x for x in r if x['ok']]
    cnt = len(ok)

    if cnt > 0:
        m = [x['f'] for x in ok]
        fr = [x['first'] for x in ok if x['first'] is not None]
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

tab = []
for name, r in all_res.items():
    ok = [x for x in r if x['ok']]
    m = [x['f'] for x in ok]
    fr = [x['first'] for x in ok if x['first'] is not None]
    tab.append({
        "метод": name,
        "ok": f"{len(ok)}/10",
        "ср масса": f"{np.mean(m):.0f}" if m else "-",
        "стд": f"{np.std(m):.0f}" if m else "-",
        "мин": f"{np.min(m):.0f}" if m else "-",
        "первое": f"{np.mean(fr):.1f}" if fr else "-"
    })

print(pd.DataFrame(tab).to_string(index=False))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

for name, r in all_res.items():
    hh = []
    for x in r:
        if any(not np.isnan(y) for y in x['hist']):
            hh.append(x['hist'][:50])
    if hh:
        a = np.array(hh)
        mu = np.nanmean(a, axis=0)
        s = np.nanstd(a, axis=0)
        ax[0,0].plot(range(len(mu)), mu, label=name)
        ax[0,0].fill_between(range(len(mu)), mu-s, mu+s, alpha=0.2)

ax[0,0].axhline(y=m_teor, color='k', ls='--')
ax[0,0].set_xlabel("итер")
ax[0,0].set_ylabel("масса")
ax[0,0].legend()
ax[0,0].grid()

names = []
means = []
stds = []
cols = []

for name, r in all_res.items():
    ok = [x for x in r if x['ok']]
    if ok:
        m = [x['f'] for x in ok]
        names.append(name)
        means.append(np.mean(m))
        stds.append(np.std(m))
        cols.append('g')
    else:
        names.append(name)
        means.append(0)
        stds.append(0)
        cols.append('r')

ax[0,1].bar(names, means, yerr=stds, capsize=5, color=cols)
ax[0,1].axhline(y=m_teor, color='k', ls='--')
ax[0,1].set_ylabel("масса")
ax[0,1].tick_params(axis='x', rotation=30)
ax[0,1].grid(axis='y')

fd = []
for name, r in all_res.items():
    tms = [x['first'] for x in r if x['first'] is not None]
    if tms:
        fd.append({'n': name, 'm': np.mean(tms), 's': np.std(tms)})

if fd:
    nms = [x['n'] for x in fd]
    mns = [x['m'] for x in fd]
    ss = [x['s'] for x in fd]
    ax[1,0].bar(nms, mns, yerr=ss, capsize=5, color='b')
    ax[1,0].set_ylabel("итер")
    ax[1,0].tick_params(axis='x', rotation=30)
    ax[1,0].grid(axis='y')

pct = [len([x for x in r if x['ok']])*10 for r in all_res.values()]
ax[1,1].bar(names, pct, color='purple')
ax[1,1].set_ylabel("%")
ax[1,1].tick_params(axis='x', rotation=30)
ax[1,1].set_ylim([0,100])
ax[1,1].grid(axis='y')

plt.tight_layout()
plt.show()
