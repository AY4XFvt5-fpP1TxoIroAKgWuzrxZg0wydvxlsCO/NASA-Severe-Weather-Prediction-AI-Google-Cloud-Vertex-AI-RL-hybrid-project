import tensorflow as tf

def build_model(seq_len=5, feature_dim=9, lstm_units=128):
    """Multi-task LSTM model: predicts storm intensity & steering angle."""
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), name="sequence_input")

    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=False)
    )(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    wind_out = tf.keras.layers.Dense(1, name="next_max_wind")(x)
    steer_out = tf.keras.layers.Dense(1, name="next_steering_angle")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[wind_out, steer_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "next_max_wind": "mse",
            "next_steering_angle": "mse"
        },
        loss_weights={"next_max_wind": 1.0, "next_steering_angle": 0.5},
        metrics={"next_max_wind": "mae", "next_steering_angle": "mae"}
    )
    return model
