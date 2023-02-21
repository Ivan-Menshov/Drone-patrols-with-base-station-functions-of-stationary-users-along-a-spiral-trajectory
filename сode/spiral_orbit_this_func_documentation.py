# Используемые пакеты и методы
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve

# Параметры модели для численного анализа
N = 100  # число пользователей
a_ravn, b_ravn = [0, 1]  # Координаты границ области моделирования
R = 0.1  # радиус покрытия дрона в км
drone_speed = 5 * 10 ** -3  # скорость дрона в метрах
time_model = 1000  # Время моделирования


def plot_drone_round(x, y, r):
    """
        Функция для отрисвки области покрытия дрона \n

        Параметры:
                    x (float): первая координата центра окружности \n
                    y (float): второе координата центра окружности \n
                    r (float): радиус области покрытия \n
    """

    angles = np.linspace(0, 2 * np.pi, 100)
    x_cir = x + r * np.cos(angles)
    y_cir = y + r * np.sin(angles)
    plt.plot(x_cir, y_cir, color='green')


def plot_line(a, b, n, iterations):
    """
        Функция отрисовки спиральной траетории

        Параметры:
                    a (float): растояния между ветками \n
                    b (float): коэфициент cмещения спирали \n
                    n (float): угол поворота спирали \n
                    iterations (float): точность (чем больше, тем точнее отображается линия) \n
    """
    angles = np.linspace(0, n * np.pi, iterations)
    x_cir = np.abs((a / (2 * np.pi) * angles)) * np.cos(angles) + b
    y_cir = np.abs((a / (2 * np.pi) * angles)) * np.sin(angles) + b
    plt.plot(x_cir, y_cir)


def f(x, a):
    """
        Функция  для нахождения  длины спирали

        Параметры:
                    x (float): угол поворота спирали \n
                    a (float): растояния между ветками \n

        Возвращаемое значение:
                    (float): длина спирали\n
    """
    f1 = a * np.cos(x) / (2 * np.pi) - a * x * np.sin(x) / (2 * np.pi)
    f2 = a * np.sin(x) / (2 * np.pi) + a * x * np.cos(x) / (2 * np.pi)
    return np.sqrt(f1 ** 2 + f2 ** 2)


def eqt1(x, a, phi, vd):
    """
         Функция  необходимая для нахождения угла phi_i+1 (движение от 0*pi до n*pi)

         Параметры:
                     x (float): искомый угол phi_i+1 \n
                     a (float): растояния между ветками \n
                     phi (float): угол phi\n
                     vd (float): скорость дрона \n

         Возвращаемое значение:
                     Уравнение
     """
    return (4 * np.pi * vd / a) + np.log(np.sqrt(phi ** 2 + 1) + phi) + phi * np.sqrt(phi ** 2 + 1) - x * np.sqrt(
        x ** 2 + 1) - np.log(np.sqrt(x ** 2 + 1) + x)


def eqt2(x, a, phi, vd):
    """
           Функция  необходимая для нахождения phi_i+1 (движение от n*pi до 0*pi)

           Параметры:
                       x (float): искомый угол phi_i+1 \n
                       a (float): растояния между ветками \n
                       phi (float): угол phi\n
                       vd (float): скорость дрона \n

           Возвращаемое значение:
                       Уравнение
       """
    return -(4 * np.pi * vd / a) - np.log(np.sqrt(x ** 2 + 1) + x) - x * np.sqrt(x ** 2 + 1) + phi * np.sqrt(
        phi ** 2 + 1) + np.log(np.sqrt(phi ** 2 + 1) + phi)


def distance(x1, y1, x2, y2):
    """
           Функция для расчета растояния между дроном и пользователем

           Параметры:
                       x1 (float): x координата первого пользователя \n
                       y1 (float): y координата первого пользователя  \n
                       x2 (float): x координата второго пользователя\n
                       y2 (float): y координата второго пользователя \n

           Возвращаемое значение:
                       (float): растояние между дроном и пользователем
       """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Задаем координаты N пользователей (равномерное распределение [a_ravn,b_ravn])
x_user = np.random.uniform(a_ravn, b_ravn, N)
y_user = np.random.uniform(a_ravn, b_ravn, N)

# Вычисляем параметры параметры спирали
a = 2 * R
b = b_ravn / 2
n = (b * 1 - R) / a * 2

# Вычисляем длину спирали
traektory_len, err = integrate.quad(f, 0, n * np.pi, args=a)

# Блок для нахождения углов phi_i на i - ом такте
phi = [0]
S = 0
working_mode = 0
residual_distance = 0
for i in range(time_model):
    if working_mode == 0:
        x_sol = fsolve(eqt1, args=(a, phi[-1], drone_speed), x0=0)
        phi.append(x_sol[0])
        v, err = integrate.quad(f, phi[-2], phi[-1], args=a)
        S += v
        if S > traektory_len:
            phi[-1] = n * np.pi
            residual_distance = S - traektory_len
            x_sol = fsolve(eqt2, args=(a, phi[-1], residual_distance), x0=0)
            v, err = integrate.quad(f, x_sol, phi[-2], args=a)
            phi[-1] = x_sol[0]
            S = traektory_len - v
            working_mode = 1
    if working_mode == 1:
        x_sol = fsolve(eqt2, args=(a, phi[-1], drone_speed), x0=0)
        phi.append(x_sol[0])
        v, err = integrate.quad(f, phi[-1], phi[-2], args=a)
        S -= v
        if S < 0:
            phi[-1] = 0 * np.pi
            residual_distance = 0 - S
            x_sol = fsolve(eqt1, args=(a, phi[-1], residual_distance), x0=0)
            v, err = integrate.quad(f, phi[-1], x_sol, args=a)
            phi[-1] = x_sol[0]
            S = +v
            working_mode = 0

# Записываем массив  углов phi_i на i - ом такте
angles = np.array(phi)

# Массивы для расчета длины кривой(спирали)
X = (a / (2 * np.pi) * angles) * np.cos(angles) + b
Y = (a / (2 * np.pi) * angles) * np.sin(angles) + b

# блок для вычисления и заполнения массива покрытых пользователей в момент времени i
covered_users = []  # задаем список, который будет хранить количство покрытых пользователей в момент времни i
sir_users = np.array([0] * N)  # задаем список, который будет вычислять SIR i - ого пользовате
for j in range(len(X)):
    covered = 0
    for i in range(len(x_user)):
        if distance(x_user[i], y_user[i], X[j], Y[j]) <= R:  # проверяем условие обслуживания
            covered += 1
            sir_users[i] += 1
    covered_users.append(covered)

# Блок построения 1-ого графика
fig1 = plt.figure(figsize=(15, 6))
ax1 = plt.subplot()
ax1.bar([i for i in range(1, 101)], sir_users)
plt.xlabel('n')
plt.ylabel('t (n)')
plt.title('Время (в тактах) нахождения пользователя n в зоне прямой видимости БПЛА в течение времени моделирования T')
fig1.savefig('1.png')

# Блок построения 2-ого графика
fig2 = plt.figure(figsize=(15, 6))
ax2 = plt.subplot()
ax2.bar([i for i in range(1, 101)], sir_users / time_model)
plt.xlabel('n')
plt.ylabel('p_cov (n)')
plt.title('Вероятность покрытия пользователя n в течение времени моделирования')
fig2.savefig('2.png')

# Блок построения 3-ого графика
covered_users_copy = np.copy(covered_users)
covered_users_copy = [i / N for i in covered_users_copy]
fig3 = plt.figure(figsize=(15, 6))
ax3 = plt.subplot()
ax3.plot(range(len(X)), covered_users_copy)
plt.xlabel('i')
plt.ylabel('P cov,i')
plt.title('Вероятность покрытия зоны обслуживания на такте [t_i, t_(i+1) ) - доля пользователей, находящихся в зоне '
          'прямой видимости БПЛА на такте [t_i, t_(i+1) )')
fig3.savefig('3.png')

# Блок построения визуализации имитационной моделе на i-ом такте
moment = int(input(f'Выбирете момент времени от 0 до {len(X) - 1} : '))
fig4 = plt.figure(dpi=100, figsize=(8, 8))
ax4 = plt.subplot()
plot_line(a, b, n, 2000)
ax4.plot(X[moment], Y[moment], 'o', color='black')
plot_drone_round(X[moment], Y[moment], R)
covered = 0
for i in range(len(x_user)):
    if distance(x_user[i], y_user[i], X[moment], Y[moment]) <= R:
        ax4.plot(x_user[i], y_user[i], '.', color='green')
        covered += 1
    else:
        ax4.plot(x_user[i], y_user[i], '.', color='red')
plt.xlim(a_ravn, b_ravn)
plt.ylim(a_ravn, b_ravn)
plt.title('Time: ' + str(moment))
fig4.savefig('4.png')

# Блок для создания файлов со статистикой
user_id = np.array([i for i in range(1, 101)])
user_stat = {
    "Пользователь": user_id,
    "t (n)": sir_users,
    "p_cov (n)": sir_users / time_model,
}
data1 = pd.DataFrame(user_stat)
data1.to_csv('ВВХ_1.csv', sep=';', index=False)

time_id = np.array([i for i in range(0, time_model + 2)])
user_stat1 = {
    "Такт времени": time_id,
    "P cov,i": covered_users_copy
}
data2 = pd.DataFrame(user_stat1)
data2.to_csv('ВВХ_2.csv', sep=';', index=False)
