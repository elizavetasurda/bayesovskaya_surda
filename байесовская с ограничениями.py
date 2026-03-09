import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------
#параметры задачи
#------------------------------------------------------------
P = 4e6
sigma_max = 9.5e6
rho = 7800
V_min = 0.4

#------------------------------------------------------------
#функция массы сосуда
#------------------------------------------------------------
def mass(x):
    R, t = x
    V_material = (4/3) * np.pi * ((R + t)**3 - R**3)
    return rho * V_material

#------------------------------------------------------------
#функция объема
#------------------------------------------------------------
def volume(x):
    R, _ = x
    return (4/3) * np.pi * R**3

#------------------------------------------------------------
#функция напряжения
#------------------------------------------------------------
def stress(x):
    R, t = x
    return P * R / (2 * t)

#------------------------------------------------------------
#ограничения
#------------------------------------------------------------
def constraints(x):
    R, t = x
    g1 = V_min - volume(x)
    g2 = stress(x) - sigma_max
    return np.array([g1, g2])

#------------------------------------------------------------
#проверка допустимости
#------------------------------------------------------------
def check_feasibility(x, tol=1e-3):
    return np.all(constraints(x) <= tol)

#------------------------------------------------------------
#степень нарушения
#------------------------------------------------------------
def violation(x):
    c = constraints(x)
    return np.sum(np.maximum(0, c))

#------------------------------------------------------------
#теоретический расчет
#------------------------------------------------------------
R_teor = (V_min * 3/(4*np.pi))**(1/3)
t_teor = P * R_teor / (2 * sigma_max)
m_teor = mass([R_teor, t_teor])

#------------------------------------------------------------
#границы поиска
#------------------------------------------------------------
bounds = np.array([
    [0.4, 0.6],    #радиус (м)
    [0.05, 0.15]   #толщина (м)
])

#------------------------------------------------------------
#КЛАСС ДЛЯ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ С ОГРАНИЧЕНИЯМИ
#------------------------------------------------------------
class ConstrainedBayesianOptimization:
    def __init__(self, bounds, n_init=10):
        self.bounds = bounds
        self.n_init = n_init
        self.X = []
        self.y = []
        self.c1 = []
        self.c2 = []
        self.history_best_feasible = []
        self.history_first_feasible = None
        self.history_violation = []

    def latin_hypercube_sample(self):
        lhc = LatinHypercube(d=2)
        sample = lhc.random(n=self.n_init)
        X_init = np.zeros((self.n_init, 2))
        for i in range(2):
            X_init[:, i] = sample[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        return X_init

    def init_sample(self):
        X_init = self.latin_hypercube_sample()
        for x in X_init:
            self.X.append(x)
            self.y.append(mass(x))
            c = constraints(x)
            self.c1.append(c[0])
            self.c2.append(c[1])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.c1 = np.array(self.c1)
        self.c2 = np.array(self.c2)

    def fit_gp_models(self):
        kernel = C(1.0) * Matern(length_scale=0.1, nu=2.5)
        self.gp_f = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_f.fit(self.X, self.y)
        self.gp_c1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_c1.fit(self.X, self.c1)
        self.gp_c2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self.gp_c2.fit(self.X, self.c2)

    def ei(self, x, y_best):
        x = x.reshape(1, -1)
        mu, sigma = self.gp_f.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)
        if sigma < 1e-10:
            return 0.0
        gamma = (y_best - mu) / sigma
        return (sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma)))[0, 0]

    def pof(self, x):
        x = x.reshape(1, -1)
        mu1, sigma1 = self.gp_c1.predict(x, return_std=True)
        sigma1 = sigma1.reshape(-1, 1)
        pof1 = norm.cdf(-mu1 / (sigma1 + 1e-10))[0, 0]
        mu2, sigma2 = self.gp_c2.predict(x, return_std=True)
        sigma2 = sigma2.reshape(-1, 1)
        pof2 = norm.cdf(-mu2 / (sigma2 + 1e-10))[0, 0]
        return pof1 * pof2

    def cei(self, x, y_best):
        return self.ei(x, y_best) * self.pof(x)

    def optimize_acquisition(self, n_iter=100):
        best_x = None
        best_acq = -np.inf
        y_best = np.min(self.y)
        for _ in range(n_iter):
            x = np.array([
                np.random.uniform(*self.bounds[0]),
                np.random.uniform(*self.bounds[1])
            ])
            acq = self.cei(x, y_best)
            if acq > best_acq:
                best_acq = acq
                best_x = x
        return best_x

    def run(self, n_iterations=50):
        self.init_sample()
        best_feasible_f = float('inf')
        best_feasible_x = None
        first_feasible_iter = None

        for t in range(n_iterations):
            self.fit_gp_models()
            x_next = self.optimize_acquisition()
            f_next = mass(x_next)
            c_next = constraints(x_next)

            self.X = np.vstack([self.X, x_next])
            self.y = np.append(self.y, f_next)
            self.c1 = np.append(self.c1, c_next[0])
            self.c2 = np.append(self.c2, c_next[1])

            if check_feasibility(x_next):
                if f_next < best_feasible_f:
                    best_feasible_f = f_next
                    best_feasible_x = x_next.copy()
                if first_feasible_iter is None:
                    first_feasible_iter = t + self.n_init

            self.history_best_feasible.append(
                best_feasible_f if best_feasible_f != float('inf') else np.nan
            )
            self.history_violation.append(violation(x_next))

        self.first_feasible_iter = first_feasible_iter
        return best_feasible_x, best_feasible_f, self.history_best_feasible

#------------------------------------------------------------
#МЕТОДЫ ОПТИМИЗАЦИИ
#------------------------------------------------------------
def run_unconstrained(n_runs=10, n_iter=50):
    results = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        class UnconstrainedBO(ConstrainedBayesianOptimization):
            def cei(self, x, y_best):
                return self.ei(x, y_best)
        bo = UnconstrainedBO(bounds, n_init=10)
        x, f, history = bo.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'feasible': check_feasibility(x) if x is not None else False,
            'history': history,
            'first_feasible': bo.first_feasible_iter
        })
    return results

def run_penalty(n_runs=10, n_iter=50):
    results = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        def penalty_obj(x):
            c = constraints(x)
            penalty = 1e4 * np.sum(np.maximum(0, c)**2)
            return mass(x) + penalty

        class PenaltyBO(ConstrainedBayesianOptimization):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init_sample(self):
                X_init = self.latin_hypercube_sample()
                for x in X_init:
                    self.X.append(x)
                    self.y.append(penalty_obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        bo = PenaltyBO(bounds, n_init=10)
        x, f, history = bo.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'feasible': check_feasibility(x) if x is not None else False,
            'history': history,
            'first_feasible': bo.first_feasible_iter
        })
    return results

def run_cei(n_runs=10, n_iter=50):
    results = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        bo = ConstrainedBayesianOptimization(bounds, n_init=10)
        x, f, history = bo.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': f if x is not None else np.nan,
            'feasible': check_feasibility(x) if x is not None else False,
            'history': history,
            'first_feasible': bo.first_feasible_iter
        })
    return results

def run_lagrange(n_runs=10, n_iter=50):
    results = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        def lagrange_obj(x):
            c = constraints(x)
            L = mass(x)
            if c[0] > 0:
                L += 1000 * c[0] + 1000 * c[0]**2
            if c[1] > 0:
                L += 1000 * c[1] + 1000 * c[1]**2
            return L

        class LagrangeBO(ConstrainedBayesianOptimization):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init_sample(self):
                X_init = self.latin_hypercube_sample()
                for x in X_init:
                    self.X.append(x)
                    self.y.append(lagrange_obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        bo = LagrangeBO(bounds, n_init=10)
        x, f, history = bo.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'feasible': check_feasibility(x) if x is not None else False,
            'history': history,
            'first_feasible': bo.first_feasible_iter
        })
    return results

def run_barrier(n_runs=10, n_iter=50):
    results = []
    for run in range(n_runs):
        np.random.seed(42 + run)
        HUGE = 1e10
        def barrier_obj(x):
            if not check_feasibility(x):
                return HUGE + mass(x)
            c = constraints(x)
            barrier = 0
            barrier -= 100 * np.log(-c[0] + 1e-10)
            barrier -= 100 * np.log(-c[1] + 1e-10)
            return mass(x) + barrier

        class BarrierBO(ConstrainedBayesianOptimization):
            def __init__(self, bounds, n_init=10):
                super().__init__(bounds, n_init)
                self.y = []
            def init_sample(self):
                X_init = self.latin_hypercube_sample()
                for x in X_init:
                    self.X.append(x)
                    self.y.append(barrier_obj(x))
                    c = constraints(x)
                    self.c1.append(c[0])
                    self.c2.append(c[1])
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.c1 = np.array(self.c1)
                self.c2 = np.array(self.c2)
            def cei(self, x, y_best):
                return self.ei(x, y_best)

        bo = BarrierBO(bounds, n_init=10)
        x, f, history = bo.run(n_iter)
        results.append({
            'run': run,
            'x': x,
            'f': mass(x) if x is not None else np.nan,
            'feasible': check_feasibility(x) if x is not None else False,
            'history': history,
            'first_feasible': bo.first_feasible_iter
        })
    return results

#------------------------------------------------------------
#ЗАПУСК ЭКСПЕРИМЕНТА
#------------------------------------------------------------
print("="*80)
print("ЭКСПЕРИМЕНТАЛЬНОЕ СРАВНЕНИЕ МЕТОДОВ")
print("="*80)
print(f"давление: {P/1e6} мпа")
print(f"допустимое напряжение: {sigma_max/1e6} мпа")
print(f"минимальный объем: {V_min} м³")
print(f"теоретическая масса: {m_teor:.0f} кг")
print("\nпараметры эксперимента:")
print("• количество запусков: 10")
print("• итераций на запуск: 50")
print("• начальная выборка: 10 точек (латинский гиперкуб)")
print()

methods = {
    "безусловный": run_unconstrained,
    "штрафной": run_penalty,
    "cei": run_cei,
    "лагранж": run_lagrange,
    "барьерный": run_barrier
}

all_results = {}

for name, method in methods.items():
    print(f"► {name}")
    results = method(n_runs=10, n_iter=50)
    all_results[name] = results

    feasible_runs = [r for r in results if r['feasible']]
    feasible_count = len(feasible_runs)

    if feasible_count > 0:
        masses = [r['f'] for r in feasible_runs]
        first_feasible = [r['first_feasible'] for r in feasible_runs if r['first_feasible'] is not None]

        print(f"  допустимых решений: {feasible_count}/10")
        print(f"  средняя масса: {np.mean(masses):.0f} кг")
        print(f"  стандартное отклонение: {np.std(masses):.0f} кг")
        print(f"  лучшая масса: {np.min(masses):.0f} кг")
        print(f"  худшая масса: {np.max(masses):.0f} кг")
        if first_feasible:
            print(f"  первое допустимое: {np.mean(first_feasible):.1f} итер.")
    else:
        print(f"  допустимых решений: 0/10")
    print()

#------------------------------------------------------------
#ИТОГОВАЯ ТАБЛИЦА
#------------------------------------------------------------
print("\n" + "="*100)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*100)

summary = []
for name, results in all_results.items():
    feasible_runs = [r for r in results if r['feasible']]
    masses = [r['f'] for r in feasible_runs]
    first_feasible = [r['first_feasible'] for r in feasible_runs if r['first_feasible'] is not None]

    summary.append({
        "метод": name,
        "допустимые": f"{len(feasible_runs)}/10",
        "средняя масса": f"{np.mean(masses):.0f}" if masses else "—",
        "стд": f"{np.std(masses):.0f}" if masses else "—",
        "лучшая масса": f"{np.min(masses):.0f}" if masses else "—",
        "первое допустимое": f"{np.mean(first_feasible):.1f}" if first_feasible else "—"
    })

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

#------------------------------------------------------------
#ГРАФИКИ
#------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

#график сходимости
ax1 = axes[0, 0]
for name, results in all_results.items():
    histories = []
    for r in results:
        if any(not np.isnan(x) for x in r['history']):
            hist = r['history'][:50]
            histories.append(hist)

    if histories:
        hist_array = np.array(histories)
        mean_hist = np.nanmean(hist_array, axis=0)
        std_hist = np.nanstd(hist_array, axis=0)
        iterations = range(len(mean_hist))
        ax1.plot(iterations, mean_hist, label=name, linewidth=2)
        ax1.fill_between(iterations, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

ax1.axhline(y=m_teor, color='black', linestyle='--', label='теория')
ax1.set_xlabel("итерация")
ax1.set_ylabel("лучшая допустимая масса (кг)")
ax1.set_title("сходимость методов (среднее ± стд)")
ax1.legend()
ax1.grid(True, alpha=0.3)

#сравнение масс
ax2 = axes[0, 1]
methods_list = []
mean_masses = []
std_masses = []
colors = []

for name, results in all_results.items():
    feasible_runs = [r for r in results if r['feasible']]
    if feasible_runs:
        masses = [r['f'] for r in feasible_runs]
        methods_list.append(name)
        mean_masses.append(np.mean(masses))
        std_masses.append(np.std(masses))
        colors.append('green')
    else:
        methods_list.append(name)
        mean_masses.append(0)
        std_masses.append(0)
        colors.append('red')

bars = ax2.bar(methods_list, mean_masses, yerr=std_masses, capsize=5, color=colors, alpha=0.7)
ax2.axhline(y=m_teor, color='black', linestyle='--', label='теория')
ax2.set_ylabel("масса (кг)")
ax2.set_title("сравнение масс (среднее ± стд)")
ax2.tick_params(axis='x', rotation=30)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

#первое допустимое решение
ax3 = axes[1, 0]
first_data = []
for name, results in all_results.items():
    first_times = [r['first_feasible'] for r in results if r['first_feasible'] is not None]
    if first_times:
        first_data.append({'method': name, 'mean': np.mean(first_times), 'std': np.std(first_times)})

if first_data:
    names = [d['method'] for d in first_data]
    means = [d['mean'] for d in first_data]
    stds = [d['std'] for d in first_data]
    ax3.bar(names, means, yerr=stds, capsize=5, color='blue', alpha=0.7)
    ax3.set_ylabel("итерация")
    ax3.set_title("первое допустимое решение (среднее ± стд)")
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.3, axis='y')

#процент допустимых решений
ax4 = axes[1, 1]
percentages = [len([r for r in results if r['feasible']]) * 10 for results in all_results.values()]
ax4.bar(methods_list, percentages, color='purple', alpha=0.7)
ax4.set_ylabel("процент допустимых решений (%)")
ax4.set_title("успешность методов")
ax4.tick_params(axis='x', rotation=30)
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

#------------------------------------------------------------
#ВЫВОДЫ
#------------------------------------------------------------
print("\n" + "="*100)
print("АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТА")
print("="*100)

#сбор данных для анализа
analysis = {}
for name, results in all_results.items():
    feasible_runs = [r for r in results if r['feasible']]
    if feasible_runs:
        masses = [r['f'] for r in feasible_runs]
        first_times = [r['first_feasible'] for r in feasible_runs if r['first_feasible'] is not None]
        analysis[name] = {
            'count': len(feasible_runs),
            'mean_mass': np.mean(masses),
            'std_mass': np.std(masses),
            'min_mass': np.min(masses),
            'max_mass': np.max(masses),
            'first_time': np.mean(first_times) if first_times else None,
            'cv': (np.std(masses) / np.mean(masses)) * 100
        }

#1. общая статистика
print("\n1. ОБЩАЯ СТАТИСТИКА")
print("-" * 80)
print(f"{'метод':<12} {'успех':<10} {'средняя масса':<15} {'стд':<8} {'лучшая масса':<13} {'худшая масса':<13} {'CV,%':<8} {'скорость':<10}")
print("-" * 80)

for name, data in analysis.items():
    print(f"{name:<12} {data['count']}/10 ({data['count']*10:>3.0f}%)  "
          f"{data['mean_mass']:>6.0f} кг     {data['std_mass']:>5.0f}     "
          f"{data['min_mass']:>5.0f} кг     {data['max_mass']:>5.0f} кг     "
          f"{data['cv']:>5.1f}    {data['first_time']:>5.1f} итер.")

#2. рейтинг по качеству
print("\n2. РЕЙТИНГ ПО КАЧЕСТВУ (лучшая масса)")
print("-" * 50)

ranking_quality = sorted(analysis.items(), key=lambda x: x[1]['min_mass'])
for i, (name, data) in enumerate(ranking_quality, 1):
    deviation = ((data['min_mass'] - m_teor) / m_teor) * 100
    print(f"   {i}. {name:<12} {data['min_mass']:>5.0f} кг  "
          f"(отклонение от теории: {deviation:+.1f}%)")

#3. рейтинг по средней массе
print("\n3. РЕЙТИНГ ПО СРЕДНЕЙ МАССЕ")
print("-" * 50)

ranking_mean = sorted(analysis.items(), key=lambda x: x[1]['mean_mass'])
for i, (name, data) in enumerate(ranking_mean, 1):
    deviation = ((data['mean_mass'] - m_teor) / m_teor) * 100
    print(f"   {i}. {name:<12} {data['mean_mass']:>5.0f} кг  "
          f"(отклонение: {deviation:+.1f}%)")

#4. рейтинг по скорости
print("\n4. РЕЙТИНГ ПО СКОРОСТИ (первое допустимое решение)")
print("-" * 50)

ranking_speed = sorted([(name, data['first_time']) for name, data in analysis.items()
                        if data['first_time'] is not None], key=lambda x: x[1])
for i, (name, time) in enumerate(ranking_speed, 1):
    print(f"   {i}. {name:<12} {time:>5.1f} итераций")

#5. рейтинг по стабильности
print("\n5. РЕЙТИНГ ПО СТАБИЛЬНОСТИ (меньше CV = стабильнее)")
print("-" * 50)

ranking_stability = sorted(analysis.items(), key=lambda x: x[1]['cv'])
for i, (name, data) in enumerate(ranking_stability, 1):
    print(f"   {i}. {name:<12} CV = {data['cv']:>5.1f}%  "
          f"(±{data['std_mass']:.0f} кг)")

#6. сравнение с теорией
print("\n6. СРАВНЕНИЕ С ТЕОРЕТИЧЕСКИМ ОПТИМУМОМ")
print("-" * 50)
print(f"   теоретическая масса (сфера): {m_teor:.0f} кг")

best_mean = min(analysis.items(), key=lambda x: x[1]['mean_mass'])
best_mean_dev = ((best_mean[1]['mean_mass'] - m_teor) / m_teor) * 100
print(f"   лучшая средняя масса: {best_mean[0]} = {best_mean[1]['mean_mass']:.0f} кг "
      f"({best_mean_dev:+.1f}% от теории)")

best_min = min(analysis.items(), key=lambda x: x[1]['min_mass'])
best_min_dev = ((best_min[1]['min_mass'] - m_teor) / m_teor) * 100
print(f"   лучшая минимальная масса: {best_min[0]} = {best_min[1]['min_mass']:.0f} кг "
      f"({best_min_dev:+.1f}% от теории)")

#7. итоговые выводы
print("\n7. ИТОГОВЫЕ ВЫВОДЫ")
print("-" * 50)

print("   на основе проведенного эксперимента можно сделать следующие выводы:\n")

#лучший по минимальной массе
best_min_method = min(analysis.items(), key=lambda x: x[1]['min_mass'])
print(f"   • лучший результат по минимальной массе показал метод «{best_min_method[0]}»")
print(f"     (масса {best_min_method[1]['min_mass']:.0f} кг, "
      f"отклонение от теории {((best_min_method[1]['min_mass'] - m_teor)/m_teor*100):+.1f}%)")

#лучший по средней массе
best_mean_method = min(analysis.items(), key=lambda x: x[1]['mean_mass'])
print(f"\n   • лучший результат по средней массе показал метод «{best_mean_method[0]}»")
print(f"     (средняя масса {best_mean_method[1]['mean_mass']:.0f} кг, "
      f"стд {best_mean_method[1]['std_mass']:.0f} кг)")

#самый быстрый
fastest_method = min(ranking_speed, key=lambda x: x[1])
print(f"\n   • самый быстрый метод — «{fastest_method[0]}»")
print(f"     (первое допустимое решение в среднем на {fastest_method[1]:.1f} итерации)")

#самый стабильный
most_stable = min(ranking_stability, key=lambda x: x[1]['cv'])
print(f"\n   • самый стабильный метод — «{most_stable[0]}»")
print(f"     (коэффициент вариации {most_stable[1]['cv']:.1f}%, "
      f"стд {most_stable[1]['std_mass']:.0f} кг)")

#общий анализ
print("\n   • обобщая результаты:")

#сравнение штрафного и лагранжа
penalty_data = analysis.get('штрафной', {})
lagrange_data = analysis.get('лагранж', {})

if penalty_data and lagrange_data:
    if penalty_data['min_mass'] < lagrange_data['min_mass']:
        print(f"     - штрафной метод показал лучшую минимальную массу "
              f"({penalty_data['min_mass']:.0f} vs {lagrange_data['min_mass']:.0f} кг)")
    else:
        print(f"     - метод лагранжа показал лучшую минимальную массу "
              f"({lagrange_data['min_mass']:.0f} vs {penalty_data['min_mass']:.0f} кг)")

#сравнение cei с теорией
cei_data = analysis.get('cei', {})
if cei_data:
    print(f"     - метод CEI показал среднюю массу {cei_data['mean_mass']:.0f} кг "
          f"(отклонение {((cei_data['mean_mass']-m_teor)/m_teor*100):+.1f}%)")

#рекомендации
print("\n   • рекомендации по выбору метода:")

if best_min_method[0] == "штрафной":
    print("     - для получения минимальной массы рекомендуется штрафной метод")
    print(f"       (лучшая масса {best_min_method[1]['min_mass']:.0f} кг)")
elif best_min_method[0] == "лагранж":
    print("     - для получения минимальной массы рекомендуется метод лагранжа")
    print(f"       (лучшая масса {best_min_method[1]['min_mass']:.0f} кг)")
else:
    print(f"     - для получения минимальной массы рекомендуется метод {best_min_method[0]}")
    print(f"       (лучшая масса {best_min_method[1]['min_mass']:.0f} кг)")

print("\n     - если важна скорость получения результата, выбирайте")
print(f"       метод «{fastest_method[0]}» (в среднем {fastest_method[1]:.1f} итераций)")

print("\n     - если важна стабильность результатов, выбирайте")
print(f"       метод «{most_stable[0]}» (CV = {most_stable[1]['cv']:.1f}%)")

print("\n     - штрафной метод является хорошим компромиссным вариантом")
print("       (хорошее соотношение массы, скорости и стабильности)")
