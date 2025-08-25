
# ✅ lightweight_cbam_block

def lightweight_cbam_block(input_tensor, reduction_ratio=16):




    avg_pool = GlobalAveragePooling2D()(input_tensor)

    dense1 = Dense(input_tensor.shape[-1] // reduction_ratio, activation="relu")(avg_pool)

    dense2 = Dense(input_tensor.shape[-1], activation="sigmoid")(dense1)

    channel_attention = Multiply()([input_tensor, dense2])

    avg_pool_spatial = GlobalAveragePooling2D()(channel_attention)

    max_pool_spatial = GlobalMaxPooling2D()(channel_attention)

    spatial_attention = Concatenate()([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Reshape((1, 1, spatial_attention.shape[-1]))(spatial_attention)

    spatial_attention = Conv2D(1, (7, 7), activation="sigmoid", padding="same")(spatial_attention)




    return Multiply()([channel_attention, spatial_attention])
