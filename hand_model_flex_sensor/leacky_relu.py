import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, alpha=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.alpha = alpha  # Параметр для Leaky ReLU

        # Инициализация весов и смещения
        self.weights = []
        self.biases = []

        # Инициализация для ReLU
        # Входной слой для первого скрытого слоя
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2/input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        # Для остальных скрытых слоев
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i + 1]) * np.sqrt(2/hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i + 1])))

        # Последний скрытый слой
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2 /hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))

    def leaky_relu(self, x):
        """Функция активации Leaky ReLU с параметром alpha"""
        return np.maximum(self.alpha * x, x)

    def leaky_relu_derivative(self, x):
        """Производная функции Leaky ReLU"""
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]

        # Прямое распространение через все слои кроме последнего
        for i in range(len(self.weights) - 1):
            layer_input = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            layer_output = self.leaky_relu(layer_input)  # Используем Leaky ReLU
            self.layer_outputs.append(layer_output)

        # Для выходного слоя используем линейную активацию
        layer_input = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(layer_input)
        self.layer_outputs.append(layer_input)  # Линейная активация

        return self.layer_outputs[-1]

    def backward(self, X, y, learning_rate):
        n = X.shape[0]

        # Ошибка выходного слоя
        output_delta = self.layer_outputs[-1] - y

        # Обратное распростронение ошибки
        deltas = [output_delta]

        # Обратное распространение через скрытые слои
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.leaky_relu_derivative(self.layer_inputs[ i -1])
            deltas.insert(0, delta)

        # Обновить веса и смещения
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.layer_outputs[i].T, deltas[i]) / n
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0) / n

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, learning_rate=0.1):
        """
        Обучение модели с отслеживанием ошибки на обучающей и валидационной выборках
        """
        train_losses = []
        val_losses = [] if X_val is not None else None

        for epoch in range(epochs):
            # Прямое распространение на обучающей выборке
            train_output = self.forward(X_train)

            # Потери
            train_loss = np.mean(np.square(train_output - y_train))
            train_losses.append(train_loss)

            # Обратное распространение (обновление весов только на обучающей выборке)
            self.backward(X_train, y_train, learning_rate)

            # Потери валидационной выборки
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = np.mean(np.square(val_output - y_val))
                val_losses.append(val_loss)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")

        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)
