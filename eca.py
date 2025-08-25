
# ✅ Efficient Channel Attention (ECA)

def eca_block(input_tensor, k_size=3):

    channels = input_tensor.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input_tensor)

    avg_pool = Lambda(lambda x: tf.expand_dims(x, axis=-1))(avg_pool)

    conv = tf.keras.layers.Conv1D(1, kernel_size=k_size, padding="same", activation="sigmoid")(avg_pool)

    conv = Lambda(lambda x: tf.squeeze(x, axis=-1))(conv)

    conv = Lambda(lambda x: tf.reshape(x, [-1, 1, 1, channels]))(conv)

    return Multiply()([input_tensor, conv])
