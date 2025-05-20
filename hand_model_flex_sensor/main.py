import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from scelet import hand_structure
from flex_sensor_hand import FlexSensorHand
from leacky_relu import MLP

def mirror_hand_structure(hand_struct):
    left_hand = copy.deepcopy(hand_struct)

    def mirror_node(node):
        #Инвертирование координаты по оси x
        node["pos"][0] = -node["pos"][0]

        # Инвертирование оси y
        if node["rot_axis"] is not None:
            if node["rot_axis"][1] != 0:
                node["rot_axis"][1] = -node["rot_axis"][1]

        # Рекурсия для дочерних суставов
        for child in node["children"].values():
            mirror_node(child)

    mirror_node(left_hand["wrist"])
    return left_hand

def generate_hand_movements_with_sensor_effects():
    movements = []
    R_min = 25000
    R_max = 125000
    R_range = R_max - R_min

    # Примерные параметры датчика
    nonlinearity = 0.03  # 3% нелинейность
    hysteresis = 0.07  # 7% гистерезис

    # Функция для внесения нелинейности и гистерезиса
    def apply_sensor_effects(resistance, is_increasing):
        # Нормализованное значение от 0 до 1
        norm_val = (resistance - R_min) / R_range

        # Нелинейное преобразование (квадратичная нелинейность)
        nonlinear_term = nonlinearity * (norm_val - 0.5) ** 2

        # Гистерезис
        hysteresis_term = hysteresis * norm_val if is_increasing else 0

        # Финальное преобразованное значение
        transformed_val = norm_val + nonlinear_term - hysteresis_term

        # Возвращаем к исходному диапазону
        return int(R_min + transformed_val * R_range)

    # 1. Сжатие в кулак и обратно
    # Сгибание (движение вверх)
    for i in range(R_min, R_max + 1, 2000):
        step = max(min((i - R_min), R_max - R_min), 0)

        # Гистерезис (движение вверх)
        th = apply_sensor_effects(R_min + step, True)
        idx = apply_sensor_effects(R_min + step, True)
        mid = apply_sensor_effects(R_min + step, True)
        ri = apply_sensor_effects(R_min + step, True)
        pi = apply_sensor_effects(R_min + step, True)

        movements.append({
            "thumb": th,
            "index": idx,
            "middle": mid,
            "ring": ri,
            "pinky": pi
        })

    # Гистерезис (движение вниз)
    for i in range(R_max, R_min - 1, -2000):
        step = max(min((i - R_min), R_max - R_min), 0)

        # Применяем эффекты датчика (движение вниз - разгибание)
        th = apply_sensor_effects(R_min + step, False)
        idx = apply_sensor_effects(R_min + step, False)
        mid = apply_sensor_effects(R_min + step, False)
        ri = apply_sensor_effects(R_min + step, False)
        pi = apply_sensor_effects(R_min + step, False)

        movements.append({
            "thumb": th,
            "index": idx,
            "middle": mid,
            "ring": ri,
            "pinky": pi
        })

    # 2. Жест "Заяц"
    for i in range(R_min, R_max + 1, 2000):
        step = max(min((i - R_min), R_max - R_min), 0)
        movements.append({
            "thumb": apply_sensor_effects(R_min + step, True),
            "index": R_min,  # Остается прямым
            "middle": R_min,  # Остается прямым
            "ring": apply_sensor_effects(R_min + step, True),
            "pinky": apply_sensor_effects(R_min + step, True)
        })

    for i in range(R_max, R_min - 1, -2000):
        step = max(min((i - R_min), R_max - R_min), 0)
        movements.append({
            "thumb": apply_sensor_effects(R_min + step, False),
            "index": R_min,
            "middle": R_min,
            "ring": apply_sensor_effects(R_min + step, False),
            "pinky": apply_sensor_effects(R_min + step, False)
        })

    # 3. Жест "Коза"
    for i in range(R_min, R_max + 1, 2000):
        step = max(min((i - R_min), R_max - R_min), 0)
        movements.append({
            "thumb": R_min,
            "index": R_min,
            "middle": apply_sensor_effects(R_min + step, True),
            "ring": apply_sensor_effects(R_min + step, True),
            "pinky": R_min
        })

    for i in range(R_max, R_min - 1, -2000):
        step = max(min((i - R_min), R_max - R_min), 0)
        movements.append({
            "thumb": R_min,
            "index": R_min,
            "middle": apply_sensor_effects(R_min + step, False),
            "ring": apply_sensor_effects(R_min + step, False),
            "pinky": R_min
        })

    # 4. Сгибание среднего пальца к большому
    for i in range(R_min, R_max + 1, 2000):
        middle_step = max(min((i - R_min), 75950), 0)  # Средний до 100950
        thumb_step = max(min((i - R_min) * 0.7, 70000), 0)  # Большой до 95000

        movements.append({
            "thumb": apply_sensor_effects(R_min + thumb_step, True),
            "index": R_min,
            "middle": apply_sensor_effects(R_min + middle_step, True),
            "ring": R_min,
            "pinky": R_min
        })

    for i in range(R_max, R_min - 1, -2000):
        middle_step = max(min((i - R_min), 75950), 0)
        thumb_step = max(min((i - R_min) * 0.7, 70000), 0)

        movements.append({
            "thumb": apply_sensor_effects(R_min + thumb_step, False),
            "index": R_min,
            "middle": apply_sensor_effects(R_min + middle_step, False),
            "ring": R_min,
            "pinky": R_min
        })

    return movements

def prepare_training_data(angles_dict):
    joint_names = list(angles_dict.keys())
    angles_data = list(zip(*[angles_dict[joint] for joint in joint_names]))

    # Входные и выходные значения
    X = []
    y = []

    for angle_set in angles_data:
        # Вход
        right_hand_angles = np.array(angle_set)
        X.append(right_hand_angles)

        # Выход
        mirrored_angles = []

        for i, joint_name in enumerate(joint_names):
            angle = angle_set[i]
            # Инвертируем
            if "thumb" in joint_name or joint_name.startswith("index") or joint_name.startswith(
                    "middle") or joint_name.startswith("ring") or joint_name.startswith("pinky"):
                mirrored_angles.append(angle)
            else:
                mirrored_angles.append(angle)

        y.append(np.array(mirrored_angles))

    return np.array(X), np.array(y)

def create_animation(filename="hand_mirroring_with_sensor_effects.gif"):
    # Создаем два подграфика
    fig = plt.figure(figsize=(14, 8))

    #  левая рука слева, правая рука справа
    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')

    # Инициализируем левую руку
    left_hand_structure = mirror_hand_structure(hand_structure)
    left_hand = FlexSensorHand(fig, ax_left, left_hand_structure, is_left=True)

    # Инициализируем правую руку
    right_hand = FlexSensorHand(fig, ax_right, hand_structure, is_left=False)

    # Генерируем движения с учетом нелинейности и гистерезиса
    movements = generate_hand_movements_with_sensor_effects()

    # Записываем углы суставов правой руки
    for movement in movements:
        right_hand.set_flex_resistance(
            th=movement["thumb"],
            idx=movement["index"],
            mid=movement["middle"],
            ri=movement["ring"],
            pi=movement["pinky"]
        )

    # Сохраняем данные углов в CSV
    right_hand.csv_save_angles("hand_data_with_sensor_effects")

    # Обучаем MLP на записанных данных
    print("Обучение MLP-модели с учетом нелинейности и гистерезиса (Leaky ReLU)...")
    X, y = prepare_training_data(right_hand.angles)

    # Разделяем данные на обучающую и тестовую выборки (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация для обучающих данных
    X_train_mean, X_train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train_std[X_train_std == 0] = 1  # Избегаем деления на ноль
    X_train_normalized = (X_train - X_train_mean) / X_train_std

    y_train_mean, y_train_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_std[y_train_std == 0] = 1  # Избегаем деления на ноль
    y_train_normalized = (y_train - y_train_mean) / y_train_std

    # Для тестовых данных
    X_test_normalized = (X_test - X_train_mean) / X_train_std
    y_test_normalized = (y_test - y_train_mean) / y_train_std

    # Задаем параметры многослойного перцептрона
    mlp = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=[5],
        output_size=y_train.shape[1],
        alpha=0.01
    )

    # Записываем ошибки при обучении
    train_losses, val_losses = mlp.train(
        X_train_normalized,
        y_train_normalized,
        X_test_normalized,
        y_test_normalized,
        epochs=1000,
        learning_rate=0.01  # Уменьшаем скорость обучения для Leaky ReLU
    )

    # Строим графики
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Test MSE')
    plt.title('MSE при обучении и тестировании (Leaky ReLU)')
    plt.xlabel('Эпоха')
    plt.ylabel('Среднеквадратичная ошибка (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_test_mse_comparison_leaky_relu.png')
    plt.close()
    print("График MSE для обучающей и тестовой выборок сохранен.")

    # Анимируем
    def animate(i):
        # Очищаем подписи на фигуре, чтобы избежать наложения
        for txt in fig.texts:
            txt.remove()

        # Очищаем подграфики
        ax_left.clear()
        ax_right.clear()

        # Правая рука использует оригинальные данные движения
        movement = movements[i]
        flex_values = right_hand.set_flex_resistance(
            th=movement["thumb"],
            idx=movement["index"],
            mid=movement["middle"],
            ri=movement["ring"],
            pi=movement["pinky"]
        )
        right_hand.draw()

        # Левая рука использует предсказание MLP
        # Извлекаем углы из текущего положения правой руки
        current_angles = []
        for joint_name in right_hand.angles.keys():
            if i < len(right_hand.angles[joint_name]):
                current_angles.append(right_hand.angles[joint_name][i])
            else:
                current_angles.append(0)  # Значение по умолчанию, если недоступно

        # Нормализуем вход для предсказания (используя параметры обучающей выборки)
        input_normalized = (np.array([current_angles]) - X_train_mean) / X_train_std

        # Предсказываем и денормализуем
        predicted_normalized = mlp.predict(input_normalized)
        predicted_angles = predicted_normalized * y_train_std + y_train_mean

        # Устанавливаем предсказанные углы суставов для левой руки
        for j, joint_name in enumerate(left_hand.angles.keys()):
            if "thumb" in joint_name:
                flex_tendon = "thumb"
            elif "index" in joint_name:
                flex_tendon = "index"
            elif "middle" in joint_name:
                flex_tendon = "middle"
            elif "ring" in joint_name:
                flex_tendon = "ring"
            elif "pinky" in joint_name:
                flex_tendon = "pinky"
            else:
                flex_tendon = "wrist"

        # Устанавливаем значения flex_resistance на основе предсказанных значений
        flex_values = left_hand.set_flex_resistance(
            th=movement["thumb"],
            idx=movement["index"],
            mid=movement["middle"],
            ri=movement["ring"],
            pi=movement["pinky"]
        )
        left_hand.draw()

        # Определяем текущий жест
        gesture_name = ""
        frame_count = len(movements) // 5  # Примерное количество кадров на жест (включая возврат)

        if i < frame_count:
            gesture_name = "Сжатие в кулак"
        elif i < frame_count * 2:
            gesture_name = "'Заяц'"
        elif i < frame_count * 3:
            gesture_name = "'Коза'"
        elif i < frame_count * 4:
            gesture_name = "Сгибание среднего к большому"
        else:
            gesture_name = "Сгибание среднего к большому"

        plt.figtext(0.5, 0.35, f"Текущий жест: {gesture_name}", ha='center', fontsize=16)

        return ax_left, ax_right

    # Создаем анимацию
    anim = FuncAnimation(fig, animate, frames=len(movements), interval=100, blit=False)

    # Сохраняем анимацию
    print(f"Сохраняем анимацию в {filename}...")
    anim.save(filename, writer='pillow', fps=10)

    # Показываем итоговые метрики
    print("\nИтоговые метрики:")
    print(f"Ошибка на обучающей выборке (MSE): {train_losses[-1]:.6f}")
    print(f"Ошибка на тестовой выборке (MSE): {val_losses[-1]:.6f}")

    return mlp, train_losses, val_losses
def main():
    print("Starting hand mirroring simulation with MLP using Leaky ReLU activation...")

    print("Генерация данных...")

    # Создание и обучение модели
    mlp, train_losses, val_losses = create_animation()

    print("Завершение!")

    # Характеристики сети
    print(f"Входной слой: {mlp.input_size}")
    print(f"Скрытые слои: {mlp.hidden_sizes}")
    print(f"Выходной слой: {mlp.output_size}")
    print(f"Количество парметров: {sum([w.size + b.size for w, b in zip(mlp.weights, mlp.biases)])}")
    print(f"альфа параметр функции активации: {mlp.alpha}")

    # Дополнительная визуализация для сравнения ошибок
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, 'b-', label='Обучающая выборка')
    plt.plot(val_losses, 'r-', label='Тестовая выборка')
    plt.title('MSE с Leaky ReLU активацией')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # График для детализации последних эпох
    plt.subplot(2, 1, 2)
    last_epochs = 100  # Последние 100 эпох
    start_idx = max(0, len(train_losses) - last_epochs)
    plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 'b-', label='Обучающая выборка')
    plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 'r-', label='Тестовая выборка')
    plt.title(f'MSE за последние {last_epochs} эпох')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('train_test_mse_detailed_leaky_relu.png')
    print("Подробный график MSE сохранен в train_test_mse_detailed_leaky_relu.png")

    # Анализ переобучения
    train_final_mse = train_losses[-1]
    val_final_mse = val_losses[-1]
    overfitting_ratio = val_final_mse / train_final_mse

    print(f"Финальная MSE на обучающей выборке: {train_final_mse:.6f}")
    print(f"Финальная MSE на тестовой выборке: {val_final_mse:.6f}")
    print(f"Отношение Test MSE / Train MSE: {overfitting_ratio:.2f}")

    # Сохранение результатов в CSV
    results_data = {
        'epoch': list(range(len(train_losses))),
        'train_mse': train_losses,
        'test_mse': val_losses
    }

    with open('training_results_leaky_relu.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_mse', 'test_mse'])
        for i in range(len(train_losses)):
            writer.writerow([i, train_losses[i], val_losses[i]])

    print("Результаты обучения сохранены в training_results_leaky_relu.csv")

if __name__ == "__main__":
    main()