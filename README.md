# Visual Localization (Позиціювання дрона без GPS)

Цей ROS 2 пакет забезпечує відометрію (visual odometry) та визначення абсолютних координат дрона на основі порівняння поточного відеопотоку зі заздалегідь підготовленим ортофотопланом (наприклад, картою з Mavic).
Це дозволяє реалізувати GNSS-denied навігацію (навігацію в умовах відсутності або придушення GPS/GNSS) на основі комп'ютерного зору (Map Matching).

## 🚀 Основний принцип роботи

1. Вузол підписується на топік зображення камери спрямованої вниз (напр., `/camera/image_raw`).
2. За допомогою OpenCV алгоритму (наразі ORB) виділяються ключові точки з кадру та з великої карти.
3. Алгоритм співставляє точки, знаходить матрицю перетворення (гомографію) і обчислює:
   - Де знаходиться центр поточного кадру на великій карті (Координати X, Y).
   - Кут повороту дрона відносно карти (Yaw).
4. Результат публікується як повідомлення `geometry_msgs/PoseWithCovarianceStamped` у топік `/visual_pose`.
5. Похибка (коваріація) розраховується динамічно: чим більше знайдено якісних спільних точок (inliers), тим більша довіра до координат.
6. Отримані дані можна подавати у фільтр Калмана (наприклад, `robot_localization` EKF Node) разом із даними IMU з польотного контролера (через MAVROS / micro-ROS).

## 📁 Структура пакету

- `visual_localization/map_matcher.py`: Основна логіка OpenCV (не залежить від ROS).
- `visual_localization/localization_node.py`: ROS 2 вузол, який працює як міст між ROS топіками та `MapMatcher`.
- `config/ekf_visual.yaml`: Приклад конфігурації для злиття сенсорів (Sensor Fusion) у `robot_localization`.
- `test/`: Модульні Pytest та ROS 2 тести.

## ⚙️ Встановлення

Для роботи пакету необхідний Companion Computer (наприклад, NVIDIA Jetson, Raspberry Pi 5) зі встановленим ROS 2 (Humble / Iron / Jazzy).

1. Клонуйте або перемістіть пакет у ваш робочий простір (наприклад, `~/ros2_ws/src/`).
2. Встановіть Python залежності:
```bash
pip install opencv-python numpy
```
3. Зберіть пакет:
```bash
cd ~/ros2_ws
colcon build --packages-select visual_localization
source install/setup.bash
```

## 🏃 Запуск

Для запуску ноди вкажіть шлях до файлу з вашим ортофотопланом (kartою) як параметр `map_path`:

```bash
ros2 run visual_localization localization_node --ros-args -p map_path:="/абсолютний/шлях/до/карти.png"
```

## 📡 Топіки

### Підписується на (Subscribes to):
- `/camera/image_raw` (`sensor_msgs/Image`): Зображення з бортової камери дрона.

### Публікує (Publishes):
- `/visual_pose` (`geometry_msgs/PoseWithCovarianceStamped`): Абсолютна позиція дрона на карті (X, Y) та орієнтація (Yaw) у вигляді кватерніону. Форм-фактор підходить для ROS пакету `robot_localization`.

## 🛠 Подальший розвиток та інтеграція нейромереж
Алгоритм OpenCV ORB підходить для швидкого тестування, але є чутливим до змін освітлення чи використання тепловізійної камери. 
Ви можете легко інтегрувати Deep Learning алгоритми (наприклад, **SuperPoint / SuperGlue** або **LoFTR**) замість ORB.
Для цього просто замініть вміст функції `match(self, frame_img)` у файлі `map_matcher.py` на ваш PyTorch/TensorRT код (сам ROS 2 інтерфейс переписувати не потрібно).
