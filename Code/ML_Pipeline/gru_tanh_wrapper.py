"""
GRU-tanh wrapper
================
Builds on the existing KerasRNNWrapper but uses tanh activation on the
GRU layer instead of ReLU, while keeping everything else (architecture
depth, dense head, dropout, optimizer, loss, batch size, epochs) identical.

This isolates the activation change so any performance difference can
be attributed to the activation, not to other architectural variation.

Reason for the experiment: GRU spikes badly at Price horizons 0h, 48h,
72h, 96h while LSTM (same architecture except cell type) does not.
ReLU on a GRU's candidate state can lead to unbounded growth, which
LSTM's internal gating dampens. tanh is the Keras default for GRU.
"""

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping

# Import the original wrapper so we can match its interface exactly
from ML_Pipeline.model_trainer import KerasRNNWrapper


class KerasGRUtanhWrapper(KerasRNNWrapper):
    """
    GRU wrapper that uses tanh activation on the recurrent layer.
    Everything else (dropout, dense head, optimizer, loss, training loop)
    matches KerasRNNWrapper exactly so the comparison is clean.
    """
    def __init__(self, epochs=50, batch_size=64):
        # Force rnn_type='GRU' so any code paths that branch on it work
        super().__init__(rnn_type='GRU', epochs=epochs, batch_size=batch_size)

    def fit(self, X, y, validation_data=None, verbose=0):
        # Build model on first call so feature count is known
        if self.model is None:
            self.model = Sequential()
            input_shape = (X.shape[1], X.shape[2])

            # The ONLY change vs the parent class: activation='tanh' instead of 'relu'
            self.model.add(GRU(64, activation='tanh', input_shape=input_shape))

            # Rest matches KerasRNNWrapper exactly
            self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop] if validation_data else []

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks
        )

        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])

    # predict() is inherited from KerasRNNWrapper unchanged
