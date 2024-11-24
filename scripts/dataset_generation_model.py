from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from poliastro.bodies import Sun, Earth
import numpy as np
import pandas as pd
from astropy.constants import G, M_earth
from astroquery.jplhorizons import Horizons
import random
import datetime
import os
from poliastro.iod import izzo

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
    'juno': '3',       
    'vesta': '4',      
    'hygiea': '10',    
}

# Планеты для расчета гравитационных возмущений
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

def select_random_target(horizons_id_map):
    """
    Функция для выбора случайного тела из словаря, содержащего карту идентификаторов небесных тел.

    Параметры:
    horizons_id_map : dict
        Словарь, где ключи - это имена небесных тел, а значения - их идентификаторы.

    Возвращает:
    str
        Случайно выбранное имя небесного тела из словаря horizons_id_map.
    """
    body_name = random.choice(list(horizons_id_map.keys()))
    print (body_name)
    return body_name

# Константы для Земли
J2 = 1.08263e-3  # в безразмерных единицах
J3 = -2.52e-6
J4 = -1.61e-6

# Параметры миссии
target_body = "mars"
for_range = 100;  # не должно быть больше arrival_window
initial_launch_date = Time("2025-01-01", scale="tdb")  # Начальная дата для поиска
arrival_window = 300 
search_step = 50  # если будет долго считаться ставь 10 или 15

print(datetime.datetime.now().time(), f"Run | {target_body} | forrange:{for_range} | step:{search_step} | arrival_window:{arrival_window}")

def random_position_on_leo():
    """
    Функция для генерации случайной орбитальной позиции на низкой околоземной орбите (LEO).

    Возвращает:
    tuple
        Кортеж, содержащий:
        - r1_geo (np.array) : позиция в пространстве (x, y, z) в километрах.
        - altitude (float) : высота орбиты в километрах.
        - semi_major_axis (float) : большая полуось орбиты в километрах.
        - eccentricity (float) : эксцентриситет орбиты.
        - arg_periapsis (astropy.units.Quantity) : аргумент перицентра в угловых градусах.
        - omega (astropy.units.Quantity) : долгота восходящего узла в угловых градусах.
        - inclination (astropy.units.Quantity) : наклонение орбиты в угловых градусах.
        - mean_anomaly (astropy.units.Quantity) : средняя аномалия орбиты в угловых градусах.
        - eccentric_anomaly (float) : эксцентрическая аномалия.
        - true_anomaly (float) : истинная аномалия орбиты.
    """
    altitude = np.random.uniform(160, 2000) * u.km
    semi_major_axis = (Earth.R.to(u.km) + altitude)

    inclination = np.random.uniform(0, 180) * u.deg  
    omega = np.random.uniform(0, 360) * u.deg
    eccentricity = np.random.uniform(0, 0.1)

    arg_periapsis = np.random.uniform(0, 360) * u.deg
    mean_anomaly = np.random.uniform(0, 360) * u.deg

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
        r1_geo,  
        altitude,                     # Высота в км
        semi_major_axis,              # Большая полуось
        eccentricity,                 # Эксцентриситет
        arg_periapsis,                # Аргумент перицентра
        omega,                        # Долгота восходящего узла
        inclination,                  # Наклонение
        mean_anomaly,                 # Средняя аномалия
        eccentric_anomaly,            # Эксцентрическая аномалия
        true_anomaly                  # Истинная аномалия
    )

def solve_kepler(M, e, tolerance=1e-6):
    """
    Решение уравнения Кеплера для нахождения эксцентрической аномалии (E) по средней аномалии (M) и эксцентриситету (e) 
    с использованием метода Ньютона-Рафсона.
    
    Параметры:
    M : float
        Средняя аномалия (угловое расстояние, соответствующее времени, с учетом эксцентриситета).
    
    e : float
        Эксцентриситет орбиты.
    
    tolerance : float, optional
        Точность решения. По умолчанию равна 1e-6.
    
    Возвращает:
    E : float
        Эксцентрическая аномалия, решенная с заданной точностью.
    """
    E = M  # Начальное приближение для эксцентрической аномалии (E)
    while True:
        # Расчет разности между левой и правой частями уравнения Кеплера
        delta = E - e * np.sin(E) - M
        E -= delta / (1 - e * np.cos(E))
        # Прерываем цикл, когда разность между левой и правой частью уравнения становится меньше заданной точности
        if abs(delta) < tolerance:
            break
    return E

def get_target_position(body_name, time):
    """
    Получение текущей позиции и скорости целевого небесного тела с помощью API Horizons.
    
    Параметры:
    body_name : str
        Имя целевого небесного тела.
    
    time : astropy.time.Time
        Время, для которого требуется получить позицию и скорость.
    
    Возвращает:
    r2 : np.array
        Позиция целевого тела в км (x, y, z).
    
    v2_target : np.array
        Скорость целевого тела в км/с (vx, vy, vz).
    
    Если возникла ошибка, возвращает (None, None).
    """
    try:
        # Получаем идентификатор тела по его имени
        body_id = horizons_id_map[body_name]
        # Используем API Horizons для получения позиции и скорости
        obj = Horizons(id=body_id, location='500@0', epochs=time.jd)
        eph = obj.vectors()
        r2 = np.array([eph['x'][0], eph['y'][0], eph['z'][0]]) * u.au
        r2 = r2.to(u.km)
        v2_target = np.array([eph['vx'][0], eph['vy'][0], eph['vz'][0]]) * (u.au / u.day)
        v2_target = v2_target.to(u.km / u.s)
        return r2, v2_target
    except Exception as e:
        return None, None

def gravitational_effects_with_J2_J3_J4(r_vec):
    """
    Вычисление гравитационных воздействий Земли на объект на орбите с учетом возмущений J2, J3 и J4.
    
    Параметры:
    r_vec : astropy.units.Quantity
        Вектор положения объекта (x, y, z) в пространстве.
    
    Возвращает:
    ax_J2, ay_J2, az_J2 : float
        Компоненты ускорения от возмущения J2 в направлении x, y и z.
    
    a_J2_magnitude : float
        Модуль ускорения от возмущения J2.
    
    ax_J3, ay_J3, az_J3 : float
        Компоненты ускорения от возмущения J3 в направлениях x, y и z.
    
    a_J3_magnitude : float
        Модуль ускорения от возмущения J3.
    
    ax_J4, ay_J4, az_J4 : float
        Компоненты ускорения от возмущения J4 в направлениях x, y и z.
    
    a_J4_magnitude : float
        Модуль ускорения от возмущения J4.
    
    summary_J2_J3_J4 : astropy.units.Quantity
        Суммарное ускорение от всех трех возмущений в км/с^2.
    """

    # Определяем гравитационную постоянную и массу Земли
    mu_earth = (G.to(u.km**3/(u.kg * u.s**2)) * M_earth).value
    
    # Расстояние от центра Земли до объекта
    r = np.linalg.norm(r_vec.value)
    x, y, z = r_vec.to(u.km).value  
    
    # Рассчитываем гравитационное возмущение от J2
    z2 = z**2
    r2 = r**2
    r5 = r**5
    factor_J2 = (3 / 2) * J2 * (mu_earth / r5) * Earth.R.to(u.km).value**2
    ax_J2 = factor_J2 * x * (5 * z2 / r2 - 1)
    ay_J2 = factor_J2 * y * (5 * z2 / r2 - 1)
    az_J2 = factor_J2 * z * (5 * z2 / r2 - 3)
    a_J2_magnitude = np.sqrt(ax_J2**2 + ay_J2**2 + az_J2**2)

    # Рассчитываем гравитационное возмущение от J3
    z3 = z**3
    factor_J3 = (1 / 2) * J3 * (mu_earth / r**7) * Earth.R.to(u.km).value**3
    ax_J3 = factor_J3 * x * (10 * z3 / r2 - 15 * z / r)
    ay_J3 = factor_J3 * y * (10 * z3 / r2 - 15 * z / r)
    az_J3 = factor_J3 * (4 * z3 / r2 - 3 * z)
    a_J3_magnitude = np.sqrt(ax_J3**2 + ay_J3**2 + az_J3**2)

    # Рассчитываем гравитационное возмущение от J4
    z4 = z**4
    factor_J4 = (5 / 8) * J4 * (mu_earth / r**7) * Earth.R.to(u.km).value**4
    ax_J4 = factor_J4 * x * (35 * z4 / r2 - 30 * z2 / r + 3)
    ay_J4 = factor_J4 * y * (35 * z4 / r2 - 30 * z2 / r + 3)
    az_J4 = factor_J4 * z * (35 * z4 / r2 - 42 * z2 / r + 9)
    a_J4_magnitude = np.sqrt(ax_J4**2 + ay_J4**2 + az_J4**2)

    # Суммарные гравитационные воздействия
    ax_total = ax_J2 + ax_J3 + ax_J4
    ay_total = ay_J2 + ay_J3 + ay_J4
    az_total = az_J2 + az_J3 + az_J4

    summary_J2_J3_J4 = np.array([ax_total, ay_total, az_total])

    return ax_J2, ay_J2, az_J2, a_J2_magnitude, ax_J3, ay_J3, az_J3, a_J3_magnitude, ax_J4, ay_J4, az_J4, a_J4_magnitude, summary_J2_J3_J4 * u.km / u.s**2

# Функция для расчета гравитационного воздействия
def gravitational_effects(launch_date, arrival_date, r1, r2):
    """
    Рассчитывает гравитационные воздействия от планет и других тел солнечной системы на объект в момент старта и прибытия.
    Воздействия рассчитываются с использованием эпемерид для даты старта и прибытия.
    
    Параметры:
    launch_date : astropy.time.Time
        Дата старта (в формате astropy).
    
    arrival_date : astropy.time.Time
        Дата прибытия (в формате astropy).
    
    r1 : astropy.units.Quantity
        Начальная позиция объекта (в километрах).
    
    r2 : astropy.units.Quantity
        Конечная позиция объекта (в километрах).
    
    Возвращает:
    effects_start : dict
        Словарь с гравитационными воздействиями на объект в момент старта (для каждого тела).
    
    effects_end : dict
        Словарь с гравитационными воздействиями на объект в момент прибытия (для каждого тела).
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

# Функция для расчета эллиптической траектории с задачей Ламберта
def calculate_elliptical_trajectory(r1, r2, tof):
    """
    Рассчитывает эллиптическую траекторию с использованием метода Ламберта для заданных начальных и конечных точек.
    
    Параметры:
    r1 : astropy.units.Quantity
        Начальная позиция объекта (в километрах).
    
    r2 : astropy.units.Quantity
        Конечная позиция объекта (в километрах).
    
    tof : float
        Время полета (в днях).
    
    Возвращает:
    v1_transfer : astropy.units.Quantity
        Начальная скорость для перехода (в км/с).
    
    v2_transfer : astropy.units.Quantity
        Конечная скорость для перехода (в км/с).
    
    Возвращает (None, None), если произошла ошибка.
    """
    try:
        (v1_transfer, v2_transfer), = izzo.lambert(Sun.k.to(u.km**3 / u.s**2), r1, r2, tof)
        return v1_transfer, v2_transfer
    except Exception:
        return None, None

# Поиск оптимальных дат и сохранение данных
def generate_dataset(num_samples):
    """
    Генерирует набор данных для орбитальных параметров с расчетом гравитационных эффектов
    и минимизации маневра delta-v для заданного количества выборок.

    Параметры:
    num_samples (int): Количество выборок, которые необходимо сгенерировать.

    Возвращает:
    pd.DataFrame: DataFrame, содержащий сгенерированные данные для каждого расчета, включая
    орбитальные параметры и коррекцию с учетом гравитационных эффектов.
    """
    data = []
    min_delta_v_elliptical = np.inf
    best_params_elliptical = None

    for i in range(num_samples):

        for launch_offset in range(0, arrival_window, search_step):
            launch_date = initial_launch_date + TimeDelta(launch_offset * u.day)

            earth_position = get_body_barycentric_posvel("earth", launch_date)
    
            # Генерируем позицию спутника в геоцентрической системе
            r1_geo, altitude, semi_major_axis, eccentricity, arg_periapsis, omega, inclination, mean_anomaly, eccentric_anomaly, true_anomaly = random_position_on_leo()
    
            r1 = r1_geo + earth_position[0].xyz.to(u.km)
            v1_earth = earth_position[1].xyz.to(u.km / u.s)
            v1_earth_magnitude = np.linalg.norm(v1_earth.to(u.km / u.s))

            eccentric_anomaly_rad = eccentric_anomaly * u.rad
            true_anomaly_rad = true_anomaly * u.rad
            eccentric_anomaly_deg = eccentric_anomaly_rad.to(u.deg)
            true_anomaly_deg = true_anomaly_rad.to(u.deg)

            for arrival_offset in range(for_range, arrival_window, search_step):
                arrival_date = launch_date + TimeDelta(arrival_offset * u.day)

                r2, v2_target = get_target_position(target_body, arrival_date)
                if r2 is None:
                    print(f"Ошибка: не удалось получить позицию для {target_body} на дату {arrival_date}")
                    continue
        
                tof = (arrival_date - launch_date).to(u.day)
                v2_target_body_magnitude = np.linalg.norm(v2_target.to(u.km / u.s))
        
                effects_start, effects_end = gravitational_effects(launch_date, arrival_date, r1, r2)
                ax_J2, ay_J2, az_J2, a_J2_magnitude, ax_J3, ay_J3, az_J3, a_J3_magnitude, ax_J4, ay_J4, az_J4, a_J4_magnitude, summary_J2_J3_J4 = gravitational_effects_with_J2_J3_J4(r1)
        
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

                # Расчет траектории
                v1_transfer, v2_transfer = calculate_elliptical_trajectory(r1_corrected, r2_corrected, tof)

                if v1_transfer is not None and v2_transfer is not None:
                    delta_v_start = np.linalg.norm(v1_transfer - v1_earth)
                    delta_v_arrival = np.linalg.norm(v2_transfer - v2_target)
                    total_delta_v = delta_v_start + delta_v_arrival
                    print("delta_v_start", delta_v_start)
                    print("v1_transfer", v1_transfer)
                    print("v1_earth", v1_earth)
                    print("v2_transfer", v2_transfer)
                    
                    print("v2_target", v2_target)
                    print("delta_v_arrival", delta_v_arrival)
                    
                    print("total_delta_v", total_delta_v)

                    delta_v_variation = np.random.uniform(-0.1, 0.1)  # Погрешность от -10% до 10%
                    total_delta_v2 = total_delta_v + total_delta_v * delta_v_variation

                    tof_variation = np.random.uniform(-0.1, 0.1)  # Погрешность от -10% до 10%
                    tof2 = tof + tof * tof_variation

                    delta_v_variation2 = np.random.uniform(-0.1, 0.1)  # Погрешность от -10% до 10%
                    total_delta_v3 = total_delta_v + total_delta_v * delta_v_variation2

                    tof_variation2 = np.random.uniform(-0.1, 0.1)  # Погрешность от -10% до 10%
                    tof3 = tof + tof * tof_variation2
            
                    if total_delta_v < min_delta_v_elliptical:
                        min_delta_v_elliptical = total_delta_v
                        best_params_elliptical = {
                            "target_body_name": target_body,
                            "launch_date_YYYY-MM-DD": launch_date.iso,
                            "arrival_date_YYYY-MM-DD": arrival_date.iso,
                            "tof_days_d": tof.value,
                            "delta_v_km_s": total_delta_v.value,
                            "r1_x_geocentric_km": r1_geo[0],
                            "r1_y_geocentric_km": r1_geo[1],
                            "r1_z_geocentric_km": r1_geo[2],
                            "orbit_altitude_km": altitude.value,
                            "semi_major_axis_km": semi_major_axis,
                            "eccentricity": eccentricity,
                            "arg_periapsis_deg": arg_periapsis.value,                                    
                            "raan_deg": omega.value,
                            "inclination_deg": inclination.value,
                            "mean_anomaly_deg": mean_anomaly.value,
                            "eccentric_anomaly_deg": eccentric_anomaly_deg.value,
                            "true_anomaly_deg": true_anomaly_deg.value,
                            # Невозмущенные векторы
                            "r1_x_km": r1[0].value,
                            "r1_y_km": r1[1].value,
                            "r1_z_km": r1[2].value,
                            "v1_x_km_s": v1_earth[0].to(u.km / u.s).value,
                            "v1_y_km_s": v1_earth[1].to(u.km / u.s).value,
                            "v1_z_km_s": v1_earth[2].to(u.km / u.s).value,
                            "v1_magnitude_km_s": v1_earth_magnitude.value,
                            "r2_x_km": r2[0].value,
                            "r2_y_km": r2[1].value,
                            "r2_z_km": r2[2].value,
                            "v2_x_km_s": v2_target[0].to(u.km / u.s).value,
                            "v2_y_km_s": v2_target[1].to(u.km / u.s).value,
                            "v2_z_km_s": v2_target[2].to(u.km / u.s).value,
                            "v2_magnitude_km_s": v2_target_body_magnitude.value,
                            # Возмущенные векторы
                            "r1_with_gravity_corrected_x_km": r1_with_gravity_corrected[0].value,
                            "r1_with_gravity_corrected_y_km": r1_with_gravity_corrected[1].value,
                            "r1_with_gravity_corrected_z_km": r1_with_gravity_corrected[2].value,
                            "r1_total_corrected_x_km": r1_corrected[0].value,
                            "r1_total_corrected_y_km": r1_corrected[1].value,
                            "r1_total_corrected_z_km": r1_corrected[2].value,
                            "v1_transfer_x_km_s": v1_transfer[0].value,
                            "v1_transfer_y_km_s": v1_transfer[1].value,
                            "v1_transfer_z_km_s": v1_transfer[2].value,
                            "r2_total_corrected_x_km": r2_corrected[0].value,
                            "r2_total_corrected_y_km": r2_corrected[1].value,
                            "r2_total_corrected_z_km": r2_corrected[2].value,
                            "v2_transfer_x_km_s": v2_transfer[0].value,
                            "v2_transfer_y_km_s": v2_transfer[1].value,
                            "v2_transfer_z_km_s": v2_transfer[2].value,
                            # Разницы векторов
                            "delta_r1_with_gravity_x_km": r1_with_gravity_corrected[0].value - r1[0].value,
                            "delta_r1_with_gravity_y_km": r1_with_gravity_corrected[1].value - r1[1].value,
                            "delta_r1_with_gravity_z_km": r1_with_gravity_corrected[2].value - r1[2].value,
                            "delta_r1_x_km": r1_corrected[0].value - r1[0].value,
                            "delta_r1_y_km": r1_corrected[1].value - r1[1].value,
                            "delta_r1_z_km": r1_corrected[2].value - r1[2].value,
                            "delta_r2_x_km": r2_corrected[0].value - r2[0].value,
                            "delta_r2_y_km": r2_corrected[1].value - r2[1].value,
                            "delta_r2_z_km": r2_corrected[2].value - r2[2].value,
                            # Возмущения
                            "ax_J2_km_s2": ax_J2,
                            "ay_J2_km_s2": ay_J2,
                            "az_J2_km_s2": az_J2,
                            "a_J2_magnitude_km_s2": a_J2_magnitude,
                            "ax_J3_km_s2": ax_J3,
                            "ay_J3_km_s2": ay_J3,
                            "az_J3_km_s2": az_J3,
                            "a_J3_magnitude_km_s2": a_J3_magnitude,
                            "ax_J4_km_s2": ax_J4,
                            "ay_J4_km_s2": ay_J4,
                            "az_J4_km_s2": az_J4,
                            "a_J4_magnitude_km_s2": a_J4_magnitude,
                            **{f"{body_name}_effect_start_km_s2": init_effects.get('effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_x_effect_start_km_s2": init_effects.get('x_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_y_effect_start_km_s2": init_effects.get('y_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_z_effect_start_km_s2": init_effects.get('z_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_effect_end_km_s2": fin_effects.get('effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_x_effect_end_km_s2": fin_effects.get('x_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_y_effect_end_km_s2": fin_effects.get('y_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_z_effect_end_km_s2": fin_effects.get('z_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                        }
                        data.append(best_params_elliptical)

                        almost_optimal_params_elliptical = {
                            # Параметры почти оптимального решения
                            "target_body_name": target_body,
                            "launch_date_YYYY-MM-DD": launch_date.iso,
                            "arrival_date_YYYY-MM-DD": arrival_date.iso,
                            "tof_days_d": tof2.value,
                            "delta_v_km_s": total_delta_v2.value,
                            "r1_x_geocentric_km": r1_geo[0],
                            "r1_y_geocentric_km": r1_geo[1],
                            "r1_z_geocentric_km": r1_geo[2],
                            "orbit_altitude_km": altitude.value,
                            "semi_major_axis_km": semi_major_axis,
                            "eccentricity": eccentricity,
                            "arg_periapsis_deg": arg_periapsis.value,                                    
                            "raan_deg": omega.value,
                            "inclination_deg": inclination.value,
                            "mean_anomaly_deg": mean_anomaly.value,
                            "eccentric_anomaly_deg": eccentric_anomaly_deg.value,
                            "true_anomaly_deg": true_anomaly_deg.value,
                            # Невозмущенные векторы
                            "r1_x_km": r1[0].value,
                            "r1_y_km": r1[1].value,
                            "r1_z_km": r1[2].value,
                            "v1_x_km_s": v1_earth[0].to(u.km / u.s).value,
                            "v1_y_km_s": v1_earth[1].to(u.km / u.s).value,
                            "v1_z_km_s": v1_earth[2].to(u.km / u.s).value,
                            "v1_magnitude_km_s": v1_earth_magnitude.value,
                            "r2_x_km": r2[0].value,
                            "r2_y_km": r2[1].value,
                            "r2_z_km": r2[2].value,
                            "v2_x_km_s": v2_target[0].to(u.km / u.s).value,
                            "v2_y_km_s": v2_target[1].to(u.km / u.s).value,
                            "v2_z_km_s": v2_target[2].to(u.km / u.s).value,
                            "v2_magnitude_km_s": v2_target_body_magnitude.value,
                            # Возмущенные векторы
                            "r1_with_gravity_corrected_x_km": r1_with_gravity_corrected[0].value,
                            "r1_with_gravity_corrected_y_km": r1_with_gravity_corrected[1].value,
                            "r1_with_gravity_corrected_z_km": r1_with_gravity_corrected[2].value,
                            "r1_total_corrected_x_km": r1_corrected[0].value,
                            "r1_total_corrected_y_km": r1_corrected[1].value,
                            "r1_total_corrected_z_km": r1_corrected[2].value,
                            "v1_transfer_x_km_s": v1_transfer[0].value,
                            "v1_transfer_y_km_s": v1_transfer[1].value,
                            "v1_transfer_z_km_s": v1_transfer[2].value,
                            "r2_total_corrected_x_km": r2_corrected[0].value,
                            "r2_total_corrected_y_km": r2_corrected[1].value,
                            "r2_total_corrected_z_km": r2_corrected[2].value,
                            "v2_transfer_x_km_s": v2_transfer[0].value,
                            "v2_transfer_y_km_s": v2_transfer[1].value,
                            "v2_transfer_z_km_s": v2_transfer[2].value,
                            # Разницы векторов
                            "delta_r1_with_gravity_x_km": r1_with_gravity_corrected[0].value - r1[0].value,
                            "delta_r1_with_gravity_y_km": r1_with_gravity_corrected[1].value - r1[1].value,
                            "delta_r1_with_gravity_z_km": r1_with_gravity_corrected[2].value - r1[2].value,
                            "delta_r1_x_km": r1_corrected[0].value - r1[0].value,
                            "delta_r1_y_km": r1_corrected[1].value - r1[1].value,
                            "delta_r1_z_km": r1_corrected[2].value - r1[2].value,
                            "delta_r2_x_km": r2_corrected[0].value - r2[0].value,
                            "delta_r2_y_km": r2_corrected[1].value - r2[1].value,
                            "delta_r2_z_km": r2_corrected[2].value - r2[2].value,
                            # Возмущения
                            "ax_J2_km_s2": ax_J2,
                            "ay_J2_km_s2": ay_J2,
                            "az_J2_km_s2": az_J2,
                            "a_J2_magnitude_km_s2": a_J2_magnitude,
                            "ax_J3_km_s2": ax_J3,
                            "ay_J3_km_s2": ay_J3,
                            "az_J3_km_s2": az_J3,
                            "a_J3_magnitude_km_s2": a_J3_magnitude,
                            "ax_J4_km_s2": ax_J4,
                            "ay_J4_km_s2": ay_J4,
                            "az_J4_km_s2": az_J4,
                            "a_J4_magnitude_km_s2": a_J4_magnitude,
                            **{f"{body_name}_effect_start_km_s2": init_effects.get('effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_x_effect_start_km_s2": init_effects.get('x_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_y_effect_start_km_s2": init_effects.get('y_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_z_effect_start_km_s2": init_effects.get('z_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_effect_end_km_s2": fin_effects.get('effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_x_effect_end_km_s2": fin_effects.get('x_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_y_effect_end_km_s2": fin_effects.get('y_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_z_effect_end_km_s2": fin_effects.get('z_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                        }
                        data.append(almost_optimal_params_elliptical)

                        almost_optimal_params_elliptical2 = {
                            # Параметры почти оптимального решения
                            "target_body_name": target_body,
                            "launch_date_YYYY-MM-DD": launch_date.iso,
                            "arrival_date_YYYY-MM-DD": arrival_date.iso,
                            "tof_days_d": tof3.value,
                            "delta_v_km_s": total_delta_v3.value,
                            "r1_x_geocentric_km": r1_geo[0],
                            "r1_y_geocentric_km": r1_geo[1],
                            "r1_z_geocentric_km": r1_geo[2],
                            "orbit_altitude_km": altitude.value,
                            "semi_major_axis_km": semi_major_axis,
                            "eccentricity": eccentricity,
                            "arg_periapsis_deg": arg_periapsis.value,                                    
                            "raan_deg": omega.value,
                            "inclination_deg": inclination.value,
                            "mean_anomaly_deg": mean_anomaly.value,
                            "eccentric_anomaly_deg": eccentric_anomaly_deg.value,
                            "true_anomaly_deg": true_anomaly_deg.value,
                            # Невозмущенные векторы
                            "r1_x_km": r1[0].value,
                            "r1_y_km": r1[1].value,
                            "r1_z_km": r1[2].value,
                            "v1_x_km_s": v1_earth[0].to(u.km / u.s).value,
                            "v1_y_km_s": v1_earth[1].to(u.km / u.s).value,
                            "v1_z_km_s": v1_earth[2].to(u.km / u.s).value,
                            "v1_magnitude_km_s": v1_earth_magnitude.value,
                            "r2_x_km": r2[0].value,
                            "r2_y_km": r2[1].value,
                            "r2_z_km": r2[2].value,
                            "v2_x_km_s": v2_target[0].to(u.km / u.s).value,
                            "v2_y_km_s": v2_target[1].to(u.km / u.s).value,
                            "v2_z_km_s": v2_target[2].to(u.km / u.s).value,
                            "v2_magnitude_km_s": v2_target_body_magnitude.value,
                            # Возмущенные векторы
                            "r1_with_gravity_corrected_x_km": r1_with_gravity_corrected[0].value,
                            "r1_with_gravity_corrected_y_km": r1_with_gravity_corrected[1].value,
                            "r1_with_gravity_corrected_z_km": r1_with_gravity_corrected[2].value,
                            "r1_total_corrected_x_km": r1_corrected[0].value,
                            "r1_total_corrected_y_km": r1_corrected[1].value,
                            "r1_total_corrected_z_km": r1_corrected[2].value,
                            "v1_transfer_x_km_s": v1_transfer[0].value,
                            "v1_transfer_y_km_s": v1_transfer[1].value,
                            "v1_transfer_z_km_s": v1_transfer[2].value,
                            "r2_total_corrected_x_km": r2_corrected[0].value,
                            "r2_total_corrected_y_km": r2_corrected[1].value,
                            "r2_total_corrected_z_km": r2_corrected[2].value,
                            "v2_transfer_x_km_s": v2_transfer[0].value,
                            "v2_transfer_y_km_s": v2_transfer[1].value,
                            "v2_transfer_z_km_s": v2_transfer[2].value,
                            # Разницы векторов
                            "delta_r1_with_gravity_x_km": r1_with_gravity_corrected[0].value - r1[0].value,
                            "delta_r1_with_gravity_y_km": r1_with_gravity_corrected[1].value - r1[1].value,
                            "delta_r1_with_gravity_z_km": r1_with_gravity_corrected[2].value - r1[2].value,
                            "delta_r1_x_km": r1_corrected[0].value - r1[0].value,
                            "delta_r1_y_km": r1_corrected[1].value - r1[1].value,
                            "delta_r1_z_km": r1_corrected[2].value - r1[2].value,
                            "delta_r2_x_km": r2_corrected[0].value - r2[0].value,
                            "delta_r2_y_km": r2_corrected[1].value - r2[1].value,
                            "delta_r2_z_km": r2_corrected[2].value - r2[2].value,
                            # Возмущения
                            "ax_J2_km_s2": ax_J2,
                            "ay_J2_km_s2": ay_J2,
                            "az_J2_km_s2": az_J2,
                            "a_J2_magnitude_km_s2": a_J2_magnitude,
                            "ax_J3_km_s2": ax_J3,
                            "ay_J3_km_s2": ay_J3,
                            "az_J3_km_s2": az_J3,
                            "a_J3_magnitude_km_s2": a_J3_magnitude,
                            "ax_J4_km_s2": ax_J4,
                            "ay_J4_km_s2": ay_J4,
                            "az_J4_km_s2": az_J4,
                            "a_J4_magnitude_km_s2": a_J4_magnitude,
                            **{f"{body_name}_effect_start_km_s2": init_effects.get('effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_x_effect_start_km_s2": init_effects.get('x_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_y_effect_start_km_s2": init_effects.get('y_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_z_effect_start_km_s2": init_effects.get('z_effect_start', 0) for body_name, init_effects in effects_start.items()},
                            **{f"{body_name}_effect_end_km_s2": fin_effects.get('effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_x_effect_end_km_s2": fin_effects.get('x_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_y_effect_end_km_s2": fin_effects.get('y_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                            **{f"{body_name}_z_effect_end_km_s2": fin_effects.get('z_effect_end', 0) for body_name, fin_effects in effects_end.items()},
                        }
                        data.append(almost_optimal_params_elliptical2)

        current_time = datetime.datetime.now().time()
        print(current_time, f" Generated trajectory sample {i}")

    return pd.DataFrame(data)

def main():
    # Относительный путь к исходному файлу данных
    dataset = generate_dataset(num_samples=2)
    save_dir="..\\data"
    file_path = os.path.join(save_dir, f"model_{target_body}_{for_range}_step{search_step}_arrivalWindow{arrival_window}.csv")
    dataset.to_csv(file_path, index=False, sep=';', encoding='utf-8')

    print(f"Расчеты завершены и сохранены в файл model_{target_body}_{for_range}_step{search_step}_arrivalWindow{arrival_window}.csv")

# Точка входа в программу
if __name__ == "__main__":
    main()
