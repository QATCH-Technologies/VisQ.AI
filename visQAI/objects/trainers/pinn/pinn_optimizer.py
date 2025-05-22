from keras_tuner import BayesianOptimization
import tensorflow as tf


class PINNBayes(BayesianOptimization):
    def __init__(
        self,
        loss_fn,
        constraints,
        data_weight=1.0,
        metrics=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.constraints = constraints
        self.data_weight = data_weight
        self.metrics = metrics or []

    def run_trial(
        self,
        trial,
        X,
        y,
        X_val,
        y_val,
        batch_size=32,
        epochs=10,
        **kwargs
    ):
        # Build and compile model
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])

        # Prepare datasets
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(buffer_size=len(X))
            .batch(batch_size)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(batch_size)
        )
        train_metric_objs = {m.name: m for m in [tf.keras.metrics.Mean(name='train_loss')] +
                             [getattr(tf.keras.metrics, m)() for m in self.metrics if isinstance(m, str)]}
        val_metric_objs = {m.name: m for m in [tf.keras.metrics.Mean(name='val_loss')] +
                           [getattr(tf.keras.metrics, m)() for m in self.metrics if isinstance(m, str)]}
        for epoch in range(epochs):
            for m in train_metric_objs.values():
                m.reset_state()      # singular! there is no reset_states()
            for m in val_metric_objs.values():
                m.reset_state()

            for x_batch, y_batch in train_ds:
                with tf.GradientTape() as tape:
                    total_loss, data_loss, phys_loss = self.loss_fn(
                        model, x_batch, y_batch,
                        constraints=self.constraints,
                        data_weight=self.data_weight
                    )
                grads = tape.gradient(total_loss, model.trainable_variables)
                tf.keras.optimizers.get(optimizer).apply_gradients(
                    zip(grads, model.trainable_variables))
                train_metric_objs['train_loss'].update_state(total_loss)
            for m in val_metric_objs.values():
                if hasattr(m, 'reset_state'):
                    m.reset_state()
                elif hasattr(m, 'reset_states'):
                    m.reset_states()
            for x_batch, y_batch in val_ds:
                total_loss, *_ = self.loss_fn(
                    model, x_batch, y_batch,
                    constraints=self.constraints,
                    data_weight=self.data_weight
                )
                val_metric_objs['val_loss'].update_state(total_loss)

            logs = {name: m.result().numpy() for name, m in {
                **train_metric_objs, **val_metric_objs}.items()}
            self.oracle.update_trial(trial.trial_id, metrics=logs)
        return logs['val_loss']
