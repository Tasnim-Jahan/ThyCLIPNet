

# ✅ ASPP Block

def aspp_block(inputs, num_filters=256):

    conv1 = Conv2D(num_filters, (1, 1), padding="same", activation="relu")(inputs)

    conv3_1 = Conv2D(num_filters, (3, 3), dilation_rate=6, padding="same", activation="relu")(inputs)

    conv3_2 = Conv2D(num_filters, (3, 3), dilation_rate=12, padding="same", activation="relu")(inputs)

    conv3_3 = Conv2D(num_filters, (3, 3), dilation_rate=18, padding="same", activation="relu")(inputs)

    avg_pool = GlobalAveragePooling2D()(inputs)

    avg_pool = Reshape((1, 1, inputs.shape[-1]))(avg_pool)

    avg_pool = Conv2D(num_filters, (1, 1), padding="same", activation="relu")(avg_pool)

    avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation="bilinear")(avg_pool)

    output = Concatenate()([conv1, conv3_1, conv3_2, conv3_3, avg_pool])

    return Conv2D(num_filters, (1, 1), padding="same", activation="relu")(output)
