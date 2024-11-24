import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.bodies import Earth
from astropy.constants import G, M_earth
from astroquery.jplhorizons import Horizons
from catboost import CatBoostRegressor
import joblib
from scipy.optimize import minimize
from datetime import timedelta
import os

# Список небесных тел для полёта к ним
horizons_id_map = {
    'mercury': '199',   
    'venus': '299',    
    'mars': '499',     
    'jupiter': '599',  
    'saturn': '699',   
    'uranus': '799',    
    'neptune': '899',  
    'pluto': '999',    
    # Спутники
    'moon': '301',      
    'phobos': '401',    
    'deimos': '402',    
    'io': '501',       
    'europa': '502',    
    'ganymede': '503',  
    'callisto': '504',    
    'dione': '604',    
    'rhea': '605',     
    'titan': '606',     
    'iapetus': '608',
    # Астероиды
    'ceres': '1',      
    'pallas': '2',     
    'juno': '3'
}

# Небесные тела, учитываемые при гравитационном взаимодействии
body_ids = {
    'Sun': '10',
    'Mercury': '199',
    'Venus': '299',  
    'Earth': '399',
    'Mars': '499',
    'Moon': '301',
    'Jupiter': '599',
    'Saturn': '699',
    'Uranus': '799',
    'Neptune': '899',
    'Pluto': '999',
}

mu_values = {
    'Sun': 132712440041.9394,      # Гравитационный параметр для Солнца (км^3/с^2)
    'Mercury': 22032.090000,       # Гравитационный параметр для Меркурия (км^3/с^2)
    'Venus': 324858.592000,        # Гравитационный параметр для Венеры (км^3/с^2)
    'Earth': 398600.436233,        # Гравитационный параметр для Земли (км^3/с^2)
    'Mars': 42828.375214,          # Гравитационный параметр для Марса (км^3/с^2)
    'Jupiter': 126712764.800000,   # Гравитационный параметр для Юпитера (км^3/с^2)
    'Saturn': 37940585.200000,     # Гравитационный параметр для Сатурна (км^3/с^2)
    'Uranus': 5794548.600000,      # Гравитационный параметр для Урана (км^3/с^2)
    'Neptune': 6836535.000000,     # Гравитационный параметр для Нептуна (км^3/с^2)
    'Pluto': 977.000000,           # Гравитационный параметр для Плутона (км^3/с^2)
    'Moon': 4902.800076            # Гравитационный параметр для Луны (км^3/с^2)
}

# Константы
J2, J3, J4 = 1.08263e-3, -2.52e-6, -1.61e-6

# ======== Вспомогательные функции ========

def get_float_input(prompt, min_val, max_val):
    """
    Функция для получения от пользователя вещественного числа в заданном диапазоне.

    Параметры:
    prompt (str): Сообщение, которое будет выведено пользователю при запросе ввода.
    min_val (float): Минимально допустимое значение для введённого числа.
    max_val (float): Максимально допустимое значение для введённого числа.

    Возвращает:
    float: Введённое пользователем число, если оно корректно и находится в пределах от min_val до max_val.

    При ошибке ввода (нечисловое значение или значение вне диапазона) пользователю будет предложено повторить ввод.
    """
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Значение должно быть в пределах от {min_val} до {max_val}. Попробуйте снова.")
        except ValueError:
            print("Некорректный ввод. Попробуйте снова.")

def get_r1_geo():
    """
    Получение геоцентрического начального радиус-вектора космического аппарата с учётом различных параметров орбиты.

    Параметры:
    Нет параметров (ввод данных происходит через input).

    Возвращает:
    astropy.units.Quantity: Геоцентрический радиус-вектор космического аппарата в километрах.

    Запрашиваемые параметры:
    - altitude (h): Высота орбиты в километрах.
    - semi_major_axis (a): Большая полуось орбиты в километрах.
    - inclination (i): Наклонение орбиты в градусах.
    - omega (Ω): Долгота восходящего узла в градусах.
    - eccentricity (e): Эксцентриситет орбиты (безразмерное число).
    - arg_periapsis (ω): Аргумент перицентра орбиты в градусах.
    - mean_anomaly (M): Средняя аномалия орбиты в градусах.

    Пример:
    r1_geo = get_r1_geo()
    """
    
    altitude = get_float_input("Высота h [160; 2000] км: ", 160, 2000) * u.km
    semi_major_axis = (Earth.R.to(u.km) + altitude)
    print(f"Большая полуось a {semi_major_axis} км")

    inclination = get_float_input("Наклонение i [0; 180]°: ", 0, 180) * u.deg 

    omega = get_float_input("Долгота восходящего узла Ω [0; 360]°: ", 0, 360) * u.deg
    eccentricity = get_float_input("Эксцентриситет e [0; 0.1]: ", 0, 0.1)

    arg_periapsis = get_float_input("Аргумент перицентра ω [0; 360]°: ", 0, 360) * u.deg
    mean_anomaly = get_float_input("Средняя аномалия M [0; 360]°: ", 0, 360) * u.deg

    mean_anomaly_rad = mean_anomaly.to(u.rad).value
    eccentric_anomaly = solve_kepler(mean_anomaly_rad, eccentricity)

    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2)
    )

    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))

    inclination_rad = inclination.to(u.rad).value
    omega_rad = omega.to(u.rad).value
    arg_periapsis_rad = arg_periapsis.to(u.rad).value
    true_anomaly_rad = true_anomaly

    # Позиции в пространстве
    x = r * (np.cos(omega_rad) * np.cos(arg_periapsis_rad + true_anomaly_rad) - 
             np.sin(omega_rad) * np.sin(arg_periapsis_rad + true_anomaly_rad) * np.cos(inclination_rad))
    y = r * (np.sin(omega_rad) * np.cos(arg_periapsis_rad + true_anomaly_rad) + 
             np.cos(omega_rad) * np.sin(arg_periapsis_rad + true_anomaly_rad) * np.cos(inclination_rad))
    z = r * np.sin(arg_periapsis_rad + true_anomaly_rad) * np.sin(inclination_rad)

    r1_geo = np.array([x.value, y.value, z.value]) * u.km

    return (
        r1_geo
    )

def solve_kepler(M, e, tolerance=1e-6):
    """
    Решение уравнения Кеплера методом Ньютона-Рафсона для нахождения эксцентричной аномалии (E).

    Параметры:
    M (float): Средняя аномалия (в радианах).
    e (float): Эксцентриситет орбиты.
    tolerance (float): Точность решения для метода Ньютона-Рафсона (по умолчанию 1e-6).

    Возвращает:
    float: Эксцентричная аномалия (E) в радианах.

    Пример:
    E = solve_kepler(M=0.5, e=0.1)
    """
    E = M  # Начальное приближение
    while True:
        delta = E - e * np.sin(E) - M
        E -= delta / (1 - e * np.cos(E))
        if abs(delta) < tolerance:
            break
    return E

def get_target_position(target_body, time):
    """
    Вычисляет положение целевого небесного тела в определённый момент времени с использованием библиотеки Horizons.

    Параметры:
    target_body (str): Название целевого небесного тела (например, 'Earth', 'Mars').
    time (astropy.time.Time): Время (объект Time), для которого нужно вычислить положение.

    Возвращает:
    astropy.units.Quantity: Вектор положения целевого тела в километрах, либо None, если произошла ошибка.

    Пример:
    target_position = get_target_position('Earth', Time('2024-01-01'))
    """
    try:
        body_id = horizons_id_map[target_body]
        obj = Horizons(id=body_id, location='500@0', epochs=time.jd)
        eph = obj.vectors()
        r2 = np.array([eph['x'][0], eph['y'][0], eph['z'][0]]) * u.au
        r2 = r2.to(u.km)
        return r2
    except Exception as e:
        return None

def gravitational_effects_with_J2_J3_J4(r_vec):
    """
    Вычисляет гравитационные воздействия от Земли на объект в орбите, с учётом эффектов J2, J3, J4.

    Параметры:
    r_vec (astropy.units.Quantity): Вектор положения объекта относительно Земли (в километрах).

    Возвращает:
    astropy.units.Quantity: Вектор ускорений, вызванных гравитационными воздействиями (J2, J3, J4), в единицах км/с^2.

    Пример:
    gravitational_effects = gravitational_effects_with_J2_J3_J4(r1)
    """
    mu_earth = (G.to(u.km**3/(u.kg * u.s**2)) * M_earth).value
    
    r = np.linalg.norm(r_vec.value)
    x, y, z = r_vec.to(u.km).value  
    
    # Эффект J2
    z2 = z**2
    r2 = r**2
    r5 = r**5
    factor_J2 = (3 / 2) * J2 * (mu_earth / r5) * Earth.R.to(u.km).value**2
    ax_J2 = factor_J2 * x * (5 * z2 / r2 - 1)
    ay_J2 = factor_J2 * y * (5 * z2 / r2 - 1)
    az_J2 = factor_J2 * z * (5 * z2 / r2 - 3)

    # Эффект J3
    z3 = z**3
    factor_J3 = (1 / 2) * J3 * (mu_earth / r**7) * Earth.R.to(u.km).value**3
    ax_J3 = factor_J3 * x * (10 * z3 / r2 - 15 * z / r)
    ay_J3 = factor_J3 * y * (10 * z3 / r2 - 15 * z / r)
    az_J3 = factor_J3 * (4 * z3 / r2 - 3 * z)

    # Эффект J4
    z4 = z**4
    factor_J4 = (5 / 8) * J4 * (mu_earth / r**7) * Earth.R.to(u.km).value**4
    ax_J4 = factor_J4 * x * (35 * z4 / r2 - 30 * z2 / r + 3)
    ay_J4 = factor_J4 * y * (35 * z4 / r2 - 30 * z2 / r + 3)
    az_J4 = factor_J4 * z * (35 * z4 / r2 - 42 * z2 / r + 9)

    # Суммарные гравитационные воздействия
    ax_total = ax_J2 + ax_J3 + ax_J4
    ay_total = ay_J2 + ay_J3 + ay_J4
    az_total = az_J2 + az_J3 + az_J4

    summary_J2_J3_J4 = np.array([ax_total, ay_total, az_total])

    return summary_J2_J3_J4 * u.km / u.s**2

def gravitational_effects(launch_date, arrival_date, r1, r2):
    """
    Вычисляет гравитационные воздействия от различных небесных тел на космический аппарат на старте и момент прибытия.

    Параметры:
    launch_date (astropy.time.Time): Время старта миссии (объект Time).
    arrival_date (astropy.time.Time): Время прибытия (объект Time).
    r1 (astropy.units.Quantity): Начальный радиус-вектор космического аппарата в километрах.
    r2 (astropy.units.Quantity): Конечный радиус-вектор космического аппарата в километрах.

    Возвращает:
    tuple: Кортеж из двух словарей — `effects_start` и `effects_end`:
        - effects_start: Воздействия на старте (словарь, где ключи — имена тел, значения — величины воздействия).
        - effects_end: Воздействия в момент прибытия (словарь, где ключи — имена тел, значения — величины воздействия).

    Пример:
    effects_start, effects_end = gravitational_effects(launch_date, arrival_date, r1, r2)
    """
    bodies = ['Sun', 'Venus', 'Earth', 'Mars', 'Moon', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    effects_start = {}
    effects_end = {}

    with solar_system_ephemeris.set('jpl'):
        for body_name in bodies:
            body_id = body_ids.get(body_name)  # Получаем id для тела

            if body_id is None:
                print(f"Ошибка: не найден идентификатор для {body_name}")
                continue

            try:
                # Использование Horizons для получения эпемерид для старта и прибытия
                obj_start = Horizons(id=body_id, location='500@0', epochs=launch_date.jd)
                eph_start = obj_start.vectors()

                obj_end = Horizons(id=body_id, location='500@0', epochs=arrival_date.jd)
                eph_end = obj_end.vectors()

                r_body_start = np.array([eph_start['x'][0], eph_start['y'][0], eph_start['z'][0]]) * u.au
                r_body_end = np.array([eph_end['x'][0], eph_end['y'][0], eph_end['z'][0]]) * u.au
                
                mu_body = mu_values.get(body_name, 0) * u.km**3 / u.s**2  # Гравитационный параметр

                # Для старта
                r_vec_start = r1 - r_body_start
                distance_start = np.linalg.norm(r_vec_start)
                if distance_start > 0:
                    effect_magnitude_start = mu_body / distance_start**2  # Гравитационное влияние на старте
                    unit_vector_start = r_vec_start / distance_start  # Направление воздействия
                    effect_vector_start = effect_magnitude_start * unit_vector_start  # Вектор воздействия 
                    effects_start[body_name] = {
                        'effect_start': effect_magnitude_start.value,
                        'x_effect_start': effect_vector_start[0].value,
                        'y_effect_start': effect_vector_start[1].value,
                        'z_effect_start': effect_vector_start[2].value,
                        }
                else:
                    effects_start[body_name] = {'effect_start': 0, 'x_effect_start': 0, 'y_effect_start': 0, 'z_effect_start': 0}

                # Для прибытия
                r_vec_end = r2 - r_body_end
                distance_end = np.linalg.norm(r_vec_end)
                if distance_end > 0:
                    effect_magnitude_end = mu_body / distance_end**2  # Гравитационное влияние на момент прибытия
                    unit_vector_end = r_vec_end / distance_end  # Направление воздействия
                    effect_vector_end = effect_magnitude_end * unit_vector_end  # Вектор воздействия
                    effects_end[body_name] = {
                        'effect_end': effect_magnitude_end.value,
                        'x_effect_end': effect_vector_end[0].value,
                        'y_effect_end': effect_vector_end[1].value,
                        'z_effect_end': effect_vector_end[2].value
                        }
                else:
                    effects_end[body_name] = {'effect_end': 0, 'x_effect_end': 0, 'y_effect_end': 0, 'z_effect_end': 0}

            except Exception as e:
                print(f"Ошибка при получении данных для {body_name}: {e}")

    return effects_start, effects_end

# Целевая функция для оптимизации
# Целевая функция для минимизации delta-v
def objective(x):
    """
    Целевая функция для оптимизации параметров движения космического аппарата.
    
    Аргументы:
    x (array-like): Вектор оптимизируемых параметров:
        x[0:3] - Коррекции начальных скоростей (v1_transfer).
        x[3:6] - Коррекции конечных скоростей (v2_transfer).
        x[6]   - Время полёта (tof_days).

    Возвращает:
    float: Сумма характеристической скорости (delta-v) и штраф за слишком большое время полёта.
    """

    v1_corr = x[:3]  # Коррекция начальных скоростей
    v2_corr = x[3:6]  # Коррекция конечных скоростей
    tof_days = x[6]  # Время полета
    delta_v = np.linalg.norm(v1_corr) + np.linalg.norm(v2_corr)
    
    # Добавляем штраф за слишком большие tof_days (по желанию)
    penalty = 0.01 * tof_days  
    return delta_v + penalty

# Ограничения: минимальное значение delta-v
def delta_v_constraint(x):
    """
    Ограничение на максимальное значение характеристической скорости delta-v.
    
    Аргументы:
    x (array-like): Вектор оптимизируемых параметров:
        x[0:3] - Коррекции начальных скоростей (v1_transfer).
        x[3:6] - Коррекции конечных скоростей (v2_transfer).
    
    Возвращает:
    float: Разница между заданным пределом delta-v и вычисленным значением.
           Должно быть >= 0 для выполнения ограничения.
    """
    delta_v = np.linalg.norm(x[:3]) + np.linalg.norm(x[3:6])
    return 10.0 - delta_v  # Предел в 10 км/с

models = CatBoostRegressor()

def save_predictions_to_file(predictions, v1_corr_opt, v2_corr_opt, tof_opt, delta_v_opt, optimized_launch_date, 
                             save_dir="..\\examples"):
    """
    Сохраняет предсказанные значения и оптимальные параметры в текстовый файл.

    Аргументы:
    predictions (dict): Словарь с предсказанными значениями. Ключи и значения могут включать:
        - 'v1_transfer_x_km_s', 'v1_transfer_y_km_s', 'v1_transfer_z_km_s': компоненты скорости на старте.
        - 'v2_transfer_x_km_s', 'v2_transfer_y_km_s', 'v2_transfer_z_km_s': компоненты скорости на финише.
        - 'tof_days_d', 'delta_v_km_s', 'days_to_launch': время полета, характеристическая скорость и дни до старта.
        - 'v1_corr_opt', 'v2_corr_opt': оптимальные коррекции скоростей.
        - 'tof_opt', 'delta_v_opt': оптимальное время полета и характеристическая скорость.
        - 'optimized_launch_date': оптимальная дата старта в формате datetime.
    
    file_name (str, опционально): Имя файла, в который будут записаны данные.

    Описание:
    Функция открывает файл с указанным именем и записывает в него значения, переданные в 
    словаре `predictions`, в удобочитаемом формате. Каждый элемент словаря будет выведен в виде строки 
    с использованием форматированных строк (f-строк). Если файл с таким именем уже существует, он будет 
    перезаписан.
    """
    file_path = os.path.join(save_dir, "Result_Trajectory_Parameters.txt")
    with open(file_path, 'w') as f:
        # Записываем предсказанные значения
        f.write("Predicted Values:\n")
        f.write(f"v1_transfer_x_km_s: {predictions['v1_transfer_x_km_s']}\n")
        f.write(f"v1_transfer_y_km_s: {predictions['v1_transfer_y_km_s']}\n")
        f.write(f"v1_transfer_z_km_s: {predictions['v1_transfer_z_km_s']}\n")
        f.write(f"v2_transfer_x_km_s: {predictions['v2_transfer_x_km_s']}\n")
        f.write(f"v2_transfer_y_km_s: {predictions['v2_transfer_y_km_s']}\n")
        f.write(f"v2_transfer_z_km_s: {predictions['v2_transfer_z_km_s']}\n")
        f.write(f"tof_days_d: {predictions['tof_days_d']}\n")
        f.write(f"delta_v_km_s: {predictions['delta_v_km_s']}\n")
        f.write(f"days_to_launch: {predictions['days_to_launch']}\n")

        f.write("\n-------------------------------------------\n")

        # Записываем оптимальные коррекции скоростей
        f.write("Оптимальные коррекции скоростей (v1): {}\n".format(v1_corr_opt))
        f.write("Оптимальные коррекции скоростей (v2): {}\n".format(v2_corr_opt))
        
        # Записываем оптимальное время полёта и характеристическую скорость
        f.write("Оптимальное время полёта: {} дней\n".format(tof_opt))
        f.write("Оптимальная характеристическая скорость (delta-v): {} км/с\n".format(delta_v_opt))
        
        # Записываем оптимальную дату старта
        f.write("Оптимальная дата старта: {}\n".format(optimized_launch_date.strftime('%Y-%m-%d')))


# ======== Основной код ========
def main():
    """
    Основная функция для расчёта и коррекции радиус-векторов, скоростей и временных параметров космического аппарата. 
    Функция объединяет расчёт начальных и конечных позиций, гравитационных воздействий, предсказание моделей, 
    а также оптимизацию параметров для минимизации характеристической скорости (delta-v).

    Функциональность:
    1. Пользовательский ввод для выбора целевого небесного тела, времени полёта и параметров орбиты.
    2. Расчёт начальной и конечной позиций аппарата с учётом гравитационных эффектов.
    3. Применение корректировок на основе гравитационных возмущений (включая эффекты J2, J3, J4).
    4. Предсказание параметров движения с использованием моделей машинного обучения.
    5. Оптимизация параметров для минимизации delta-v с учётом ограничений на скорости и время полёта.
    6. Вывод оптимальных параметров, включая дату старта и временные характеристики.

    Входные параметры:
    - Ввод данных осуществляется через интерактивный пользовательский интерфейс:
        * Небесное тело назначения.
        * Время полёта.
        * Орбитальные параметры аппарата.

    Возвращаемое значение:
    - Выводит скорректированные параметры:
        * Скорости на начальном и конечном этапах (v1 и v2).
        * Оптимизированное время полёта.
        * Минимальную характеристическую скорость (delta-v).
        * Оптимальную дату старта.

    Шаги выполнения:
    1. Запрос целевого тела, времени полёта и параметров орбиты.
    2. Расчёт радиус-векторов начальной и конечной позиции аппарата.
    3. Учет гравитационных возмущений (гравитационные эффекты и возмущения J2, J3, J4).
    4. Предсказание начальных параметров с помощью предобученных моделей.
    5. Оптимизация с использованием начального приближения, заданного предсказанными параметрами.
    6. Вывод оптимизированных параметров полёта.

    Пример использования:
        main()
    """
     
    # Ввод данных пользователя
    target_body = input(
    "Write target body from the list:\n"
    "Planets:\n"
    "  mercury, venus, mars, jupiter, saturn, uranus, neptune, pluto\n"
    "Moons:\n"
    "  moon, phobos, deimos, io, europa, ganymede, callisto, dione, rhea, titan, iapetus\n"
    "Asteroids:\n"
    "  ceres, pallas, juno\n"
    "Your choise: "
    ).strip().lower()

    launch_date = Time("2025-01-01", scale="tdb")
    earth_position = get_body_barycentric_posvel("earth", launch_date)
    
    r1_geo = get_r1_geo()
    r1 = r1_geo + earth_position[0].xyz.to(u.km)
    flight_time = float(input("Время полёта (дни): ")) * u.day
    arrival_date = launch_date + flight_time

    r2 = get_target_position(target_body, arrival_date)

    effects_start, effects_end = gravitational_effects(launch_date, arrival_date, r1, r2)
    summary_J2_J3_J4 = gravitational_effects_with_J2_J3_J4(r1)
        
    total_gravity_effect_start = np.array([0.0, 0.0, 0.0])
    total_gravity_effect_end = np.array([0.0, 0.0, 0.0])

    for init_effects in effects_start.values():
        total_gravity_effect_start += np.array([init_effects.get('x_effect_start', 0), 
                                                init_effects.get('y_effect_start', 0), 
                                                init_effects.get('z_effect_start', 0)])
            
    for fin_effects in effects_end.values():
        total_gravity_effect_end += np.array([fin_effects.get('x_effect_end', 0), 
                                            fin_effects.get('y_effect_end', 0), 
                                            fin_effects.get('z_effect_end', 0)])
        
    # Применяем коррекцию
    r1_with_gravity_corrected = r1 + total_gravity_effect_start * u.km
    r1_corrected = r1_with_gravity_corrected + summary_J2_J3_J4.value * u.km
    r2_corrected = r2 + total_gravity_effect_end * u.km

    r1_total_corrected_x_km = r1_corrected[0].value if hasattr(r1_corrected[0], 'value') else r1_corrected[0]
    r1_total_corrected_y_km = r1_corrected[1].value if hasattr(r1_corrected[1], 'value') else r1_corrected[1]
    r1_total_corrected_z_km = r1_corrected[2].value if hasattr(r1_corrected[2], 'value') else r1_corrected[2]

    r2_total_corrected_x_km = r2_corrected[0].value if hasattr(r2_corrected[0], 'value') else r2_corrected[0]
    r2_total_corrected_y_km = r2_corrected[1].value if hasattr(r2_corrected[1], 'value') else r2_corrected[1]
    r2_total_corrected_z_km = r2_corrected[2].value if hasattr(r2_corrected[2], 'value') else r2_corrected[2]

    print(f"Corrected initial position: {r1_corrected.value} km")
    print(f"r1_total_corrected_x_km : {r1_total_corrected_x_km} km")
    print(f"r1_total_corrected_y_km : {r1_total_corrected_y_km} km")
    print(f"r1_total_corrected_z_km : {r1_total_corrected_z_km} km")
    print("------------------------------------------------------")
    print(f"Corrected final position: {r2_corrected.value} km")
    print(f"r2_total_corrected_x_km : {r2_total_corrected_x_km} km")
    print(f"r2_total_corrected_y_km : {r2_total_corrected_y_km} km")
    print(f"r2_total_corrected_z_km : {r2_total_corrected_z_km} km")

    # Определение целевых переменных
    target_columns = [
        'v1_transfer_x_km_s', 'v1_transfer_y_km_s', 'v1_transfer_z_km_s',
        'v2_transfer_x_km_s', 'v2_transfer_y_km_s', 'v2_transfer_z_km_s',
        'tof_days_d', 'delta_v_km_s', 'days_to_launch'
    ]

    user_input = pd.DataFrame({
    'target_body_name': [target_body],
    'r1_total_corrected_x_km': [r1_total_corrected_x_km],
    'r1_total_corrected_y_km': [r1_total_corrected_y_km],
    'r1_total_corrected_z_km': [r1_total_corrected_z_km],
    'r2_total_corrected_x_km': [r2_total_corrected_x_km],
    'r2_total_corrected_y_km': [r2_total_corrected_y_km],
    'r2_total_corrected_z_km': [r2_total_corrected_z_km]
    })

    loaded_models = {}
    for target in target_columns:
        model = CatBoostRegressor()
        model.load_model(f'model_{target}.cbm')  # Загружаем каждую модель
        loaded_models[target] = model

    scalers = joblib.load('scalers.pkl')

    print("Все модели и скейлеры успешно загружены.")

    # Получаем предсказания
    predictions = {}

    for target in target_columns:
        model = loaded_models[target]  # Загружаем модель
        y_pred = model.predict(user_input)  # Получаем предсказание

        # Денормализация
        if target in scalers:  # Проверяем, что scalers - это словарь
            y_pred_denormalized = scalers[target].inverse_transform(y_pred.reshape(-1, 1)).flatten()[0]
        else:
            y_pred_denormalized = y_pred[0]

        predictions[target] = y_pred_denormalized

    v1_transfer_x_km_s = predictions['v1_transfer_x_km_s']
    v1_transfer_y_km_s = predictions['v1_transfer_y_km_s']
    v1_transfer_z_km_s = predictions['v1_transfer_z_km_s']
    v2_transfer_x_km_s = predictions['v2_transfer_x_km_s']
    v2_transfer_y_km_s = predictions['v2_transfer_y_km_s']
    v2_transfer_z_km_s = predictions['v2_transfer_z_km_s']
    tof_days_d = predictions['tof_days_d']
    delta_v_km_s = predictions['delta_v_km_s']
    days_to_launch = predictions['days_to_launch']

    print("Predicted Values:")
    print(f"v1_transfer_x_km_s: {v1_transfer_x_km_s}")
    print(f"v1_transfer_y_km_s: {v1_transfer_y_km_s}")
    print(f"v1_transfer_z_km_s: {v1_transfer_z_km_s}")
    print(f"v2_transfer_x_km_s: {v2_transfer_x_km_s}")
    print(f"v2_transfer_y_km_s: {v2_transfer_y_km_s}")
    print(f"v2_transfer_z_km_s: {v2_transfer_z_km_s}")
    print(f"tof_days_d: {tof_days_d}")
    print(f"delta_v_km_s: {delta_v_km_s}")
    print(f"days_to_launch: {days_to_launch}")

    # Начальное приближение
    x0 = [
        v1_transfer_x_km_s,
        v1_transfer_y_km_s,
        v1_transfer_z_km_s,
        v2_transfer_x_km_s,
        v2_transfer_y_km_s,
        v2_transfer_z_km_s,
        tof_days_d
    ]

    # Пределы для оптимизации
    bounds = [
        (-5, 5),  # Коррекции v1_transfer_x
        (-5, 5),  # Коррекции v1_transfer_y
        (-5, 5),  # Коррекции v1_transfer_z
        (-5, 5),  # Коррекции v2_transfer_x
        (-5, 5),  # Коррекции v2_transfer_y
        (-5, 5),  # Коррекции v2_transfer_z
        (240, 1000),  # Время полёта в днях
    ]

    # Ограничение в виде словаря
    constraints = {
        'type': 'ineq',  # Неравенство: должно быть >= 0
        'fun': delta_v_constraint
    }

    # Оптимизация
    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    # Оптимальные значения
    v1_corr_opt = result.x[:3]
    v2_corr_opt = result.x[3:6]
    tof_opt = result.x[6]
    delta_v_opt = objective(result.x)

    # Вычисление оптимальной даты старта
    optimized_launch_date = launch_date + timedelta(days=(tof_opt - tof_days_d))

    # Вывод результатов
    print("Optimal v1 correction:", v1_corr_opt)
    print("Optimal v2 correction:", v2_corr_opt)
    print("Optimal time of flight:", tof_opt, "days")
    print("Optimal delta-V", delta_v_opt, "km/s")
    print(f"Optimal launch time: {optimized_launch_date.strftime('%Y-%m-%d')}")

    save_predictions_to_file(predictions, v1_corr_opt, v2_corr_opt, tof_opt, delta_v_opt, optimized_launch_date)


# Точка входа в программу
if __name__ == "__main__":
    main()