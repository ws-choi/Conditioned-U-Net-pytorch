def dense_block(
    x, n_neurons, input_dim, initializer, activation='relu'
):
    for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
        extra = i != 0
        x = Dense(n, input_dim=d, activation=activation,
                  kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x
