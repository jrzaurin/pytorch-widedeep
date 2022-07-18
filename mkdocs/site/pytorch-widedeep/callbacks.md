# Callbacks

Here are the 4 callbacks available to the user in `pytorch-widedepp`:
`LRHistory`, `ModelCheckpoint`, `EarlyStopping` and `RayTuneReporter`.

:information_source:  **NOTE**: other callbacks , like `History`, run always
 by default. In particular, the `History` callback saves the metrics in the
 `history` attribute of the `Trainer`.

::: pytorch_widedeep.callbacks.LRHistory

::: pytorch_widedeep.callbacks.ModelCheckpoint

::: pytorch_widedeep.callbacks.EarlyStopping
