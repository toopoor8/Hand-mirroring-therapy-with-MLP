import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
from scelet import hand_structure

class FlexSensorHand:
    def __init__(self, fig=None, ax=None, hand_structure=hand_structure, is_left=False):
        self.base_structure = copy.deepcopy(hand_structure)
        self.hand_structure = copy.deepcopy(self.base_structure)
        self.is_left = is_left

        if fig is None or ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig = fig
            self.ax = ax

        self.angles = {}

    def _rotate_joint(self, point_pos, parent_pos, axis, degrees):
        # Перевод градусов в радианы
        theta = -1 * np.radians(degrees)

        #единичный вектор оси вращения
        k = np.array(axis) / np.linalg.norm(axis)

        v = point_pos - parent_pos

        # Флормула родригера
        v_rot = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

        # Сместить на положение родителя
        return v_rot + parent_pos

    def _new_position(self, node, parent_node, sensor_flexion_values, child_name=None):
        """Рекрсивное обновление суставов.
        Дочерние суставы изменяют положение относительно родителей.
        """
        current_pos = np.array(node["pos"])
        node["orig_pos"] = copy.deepcopy(node["pos"])

        if parent_node is not None:
            tendon_name = parent_node["tendon"]
            if tendon_name:
                flexion_value = sensor_flexion_values[tendon_name]
                rotation_angle = flexion_value * parent_node["max_angle"] + parent_node["angle"]

                new_pos = self._rotate_joint(
                    current_pos,
                    np.array(parent_node["orig_pos"]),
                    parent_node["rot_axis"],
                    rotation_angle
                )

                new_pos = new_pos + np.array(parent_node["pos"]) - np.array(parent_node["orig_pos"])
                node["pos"] = list(new_pos)
                node["angle"] = rotation_angle

                if child_name is not None:
                    if not self.angles.get(child_name):
                        self.angles[child_name] = []

                    self.angles[child_name].append(rotation_angle)

        for child_name, child_node in node["children"].items():
            self._new_position(child_node, node, sensor_flexion_values, child_name)

    def set_flex_resistance(self, th=0, idx=0, mid=0, ri=0, pi=0):
        """Установление нормированных коэффициентов сгиба в зависимости от сопротивления"""
        R_min = 25000
        R_max = 125000
        self.hand_structure = copy.deepcopy(self.base_structure)
        wr = 0

        flex_sensor_dict = {
            "wrist": wr,
            "thumb": (th - R_min) / 100000,
            "index": (idx - R_min) / 100000,
            "middle": (mid - R_min) / 100000,
            "ring": (ri - R_min) / 100000,
            "pinky": (pi - R_min) / 100000
        }

        self._new_position(self.hand_structure['wrist'], None, flex_sensor_dict)
        return flex_sensor_dict

    def draw(self):
        """Рекурсивная визуализация"""

        def plot_node_and_edges(ax, node, parent_pos):
            current_pos = np.array(node["pos"])

            # Левая рука розовая правая голубая
            point_color = 'deeppink' if self.is_left else 'blue'
            line_color = 'deeppink' if self.is_left else 'blue'
            line_width = 1.5

            ax.scatter(*current_pos, color=point_color, s=25)

            if parent_pos is not None:
                ax.plot3D(*zip(parent_pos, current_pos), color=line_color, linewidth=line_width)

            for child_node in node["children"].values():
                plot_node_and_edges(ax, child_node, current_pos)

        # Отрисовка руки
        node = self.hand_structure["wrist"]
        plot_node_and_edges(self.ax, node, None)

        # Настройка осей
        self.ax.set_xlim3d(-1.5, 1.5)
        self.ax.set_ylim3d(-1.5, 1.5)
        self.ax.set_zlim3d(-1.5, 1.5)

        if self.is_left:
            view_angle = -50  # Ракурс левой руки
            title = "Левая рука (MLP)"
        else:
            view_angle = -130  # Ракурс правой руки
            title = "Правая рука"

        self.ax.view_init(30, view_angle)
        self.ax.set_axis_off()

        # Надписи рук
        self.ax.set_title(title, color='black', fontsize=20, fontweight='bold')

    def csv_save_angles(self, filename="hand_data"):
        # Сохранение углов в csv
        with open(f"./{filename}.csv", "w", newline='') as file:
            writer = csv.writer(file, delimiter=',')

            writer.writerow(list(self.angles.keys()))
            writer.writerows([list(pair) for pair in zip(*list(self.angles.values()))])
