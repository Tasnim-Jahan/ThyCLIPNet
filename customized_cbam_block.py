# -*- coding: utf-8 -*-


# ✅ customized_cbam_block

def customized_cbam_block(input_tensor):

    channel_avg = GlobalAveragePooling2D()(input_tensor)

    channel_max = GlobalMaxPooling2D()(input_tensor)

    shared_dense = Dense(input_tensor.shape[-1] // 8, activation="relu")

    channel_weights = Dense(input_tensor.shape[-1], activation="sigmoid")(shared_dense(channel_avg) + shared_dense(channel_max))

    channel_attention = Multiply()([input_tensor, channel_weights])









    avg_pool_spatial = GlobalAveragePooling2D()(channel_attention)

    max_pool_spatial = GlobalMaxPooling2D()(channel_attention)

    spatial_attention = Concatenate()([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Reshape((1, 1, spatial_attention.shape[-1]))(spatial_attention)

    spatial_attention = Conv2D(1, (7, 7), activation="sigmoid", padding="same")(spatial_attention)



    return Multiply()([channel_attention, spatial_attention])
