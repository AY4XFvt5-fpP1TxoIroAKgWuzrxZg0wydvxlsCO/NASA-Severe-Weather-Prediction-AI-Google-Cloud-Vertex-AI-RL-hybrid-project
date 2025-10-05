import tensorflow as tf
import os, json

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # Example shapes and synthetic data
    X = tf.random.normal((100, 24, 8))
    y = tf.random.normal((100, 1))
    model = build_model(X.shape[1:])
    model.fit(X, y, epochs=10)
    os.makedirs("/app/output", exist_ok=True)
    model.save("/app/output/lstm_model")
    with open("/app/output/lstm_metrics.json","w") as f:
        json.dump({"mae": float(model.history.history["mae"][-1])}, f)
