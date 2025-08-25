

# ✅ Train the Model

history = model.fit(

    train_dataset_combined,

    validation_data=valid_dataset_combined,

    epochs=150,

    callbacks=callbacks,

    batch_size=batch_size,

    verbose=1
)
