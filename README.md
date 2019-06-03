Start new branch
# ML Higgs
## Multiclass classification
### Full experiments report
```https://www.overleaf.com/read/jtgybrwhkvst```
### Running training
1.Set `RHORHO_DATA` env variable
2. Set python 2 environment with Tensorflow 1.10 library
3. Run `python main.py`. You can change default behavior by using command line options.

`--restrict_most_probable_angle True/False` restricts range of most probable mixing angle from (0,2pi) to (0,pi)

`--force_download True/False` when set, it forces to download data from server

`--data_url <URL>` set url to data location (with files `rhorho_raw.data.npy`, `rhorho_raw.perm.npy`, `rhorho_raw.w_<i>.npy`)

`--reuse_weights True/False` forces to reuse calculated weights (when available)

`----normalize_weights True/False` normalize weights to make constant term equal one
### Changelog
3.06
- Added WEIGHTS_SUBSET option
- Fixed calculation of l1 and l2 metric in classification
- Fixed plotting histogram of most probable class prediction
- Allow run in different processes parallely (different training)
- Add ROC AUC calculation
- Add testing max sensitive evaluation
- Add ROC AUC testing in notebook
- Fixed plots in notebook
- Fixed calculating distance in accuracy (first and last class is the same)

### Version 2
Modifications made

- Prepered model for new weights e.g. change range from 2pi to 1pi to prevent from ambiguous mixing angle
- Preperad unweighted datasets
- Implemented huber loss with sin-cos parametrization (`tloss==parametrized_sincos`) which is now default
- Prepared `MixingAngleAnalysis` notebook with tests on unweighted data in `notebooks` directory
- Added `InitialModelTests` notebook with initial analysis 
- Added `Sin-cos training_test` notebook with benchmark to test training on simple data

### Version 1
Modified files:
- `train_rhorho.py`
- `tf_model.py`
- `data_utils.py`

Modification made:
1. Modifications of `Dataset` and `EventDatasets` classes in `data_utils.py`.
    - Changed `wa`, `wb` weights of class scalar and pseudoscalar to array of weights
    - Added `max_args` array which stores `phi` value which corresponds to maximum weight for each event
    - Added `popts` array which stores `a`, `b`, `c` parameters of weights function fitting
    ```
    self.train = Dataset(data[train_ids], weights[train_ids, :], arg_maxs[train_ids], popts[train_ids])
    self.valid = Dataset(data[valid_ids], weights[valid_ids, :], arg_maxs[valid_ids], popts[valid_ids])
    self.test = Dataset(data[test_ids], weights[test_ids, :], arg_maxs[test_ids], popts[test_ids])
    ```
2. Changes in data loading (`train_rhorho.py` file):
    - There are three data files loaded
    ```
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w = read_np(os.path.join(data_path, "rhorho_raw.w.npy"))
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    ```
    - `rhorho_raw.data.npy` is numpy file with event data (particle kinematics data) (not changed, from previous version),
    - `rhorho_raw.w.npy` is numpy file with weights for each event (0, 0.1, 0.2, ..., 0.9, 1)
    - `rhorho_raw.perm.npy` is numpy permutation file (not changed, from previous version)

3. Changes in data preprocessing (`train_rhorho.py` file):
    - If file with calculated fitting parameters does not exist (`popt.npy`), we run fitting process to weights and save the results
    ```angular2
    if not os.path.exists(os.path.join(data_path, 'popts.npy')):
        popts = np.zeros((data_len, 3))
        for i in range(data_len ):
            popt, pcov = optimize.curve_fit(weight_fun, x, w[i, :], p0=[1, 1, 1])
            popts[i] = popt
        np.save(os.path.join(data_path, 'popts.npy'), popts)
    ```
    - if flag `reuse_weigths` is not set we run process of calculating weights for each class (based on number of classes). For example if `NUM_CLASSES` parameters is set to 50, it will be calulated 50 weights for each event (for different phi)
    - Simultaneusly there is found argument (phi) which corresponds to maximum weight
    - Results are saved to file. It saves time if parameter `NUM_CLASSESS` is not changed
    ```
    if not reuse_weigths:
        for i in range(data_len):
            weights[i] = weight_fun(classes, *popts[i])
            arg_max = 0
            if weight_fun(np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = np.pi
            phi = np.arctan(popts[i][2] / popts[i][1])

            if 0 < phi < np.pi and weight_fun(phi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = phi
            if 0 < phi + np.pi < np.pi and weight_fun(phi + np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = phi + np.pi

            arg_maxs[i] = arg_max
        np.save(os.path.join(data_path, 'weigths.npy'), weights)
        np.save(os.path.join(data_path, 'arg_maxs.npy'), arg_maxs)
    ```

3. Neural network modifications (`tf_model.py` file)
    - There are three available loss function options. Option is chosen by setting parameter `tloss` of `NeuralNetwork` class. Default is `soft` which corresponds to softmax. Other options are `regr` and `popts` which corresponds to regression to maximum weight value and regression to fitting parameters respectively.
    - Softmax is generalization of previous method on more than two classes. Instead of two classes weights `wa` `wb` , there is passesed array of weights. Loss functions is softmax cross entropy which is common for classification tasks.
    ```angular2
    sx = linear(x, "regression", num_classes)
    self.preds = tf.nn.softmax(sx)
    self.p = self.preds

    labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), (1,num_classes))
    self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
    ```
    - Regr is regression to argument for maximum weight. Only this value is returned by neural network. Loss function is set to l2 metric (MSE)
    ```
    sx = linear(x, "regr", 1)
    self.sx = sx
    self.loss = loss = tf.losses.mean_squared_error(self.arg_maxs, sx[:, 0])
    ```
    - Popts is modification of regr method. In this case neural network retuns three values, which corresponds to three parameters of weights function fitting. Loss function is set to l2 metric (MSE)
    ```
    sx = linear(x, "regr", 3)
    self.sx = sx
    self.p = sx
    self.loss = loss = tf.losses.mean_squared_error(self.popts, sx)
    ```
4. Evaluation (Work in progress) (`tf_model.py` file)
    - Function `predictions` is modifed to return more predictions (based on chosen loss function option). It also return real data weights, popts, and max_args from train, valid or test dataset.
    ```angular2
    def predictions(model, dataset, at_most=None, filtered=False):
        sess = tf.get_default_session()
        x = dataset.x
        weights = dataset.weights
        filt = dataset.filt
        arg_maxs = dataset.arg_maxs
        popts = dataset.popts

        if at_most is not None:
          filt = filt[:at_most]
          x = x[:at_most]
          weights = weights[:at_most]
          arg_maxs = arg_maxs[:at_most]

        p = sess.run(model.p, {model.x: x})

        if filtered:
          p = p[filt == 1]
          x = x[filt == 1]
          weights = weights[filt == 1]
          arg_maxs = arg_maxs[filt == 1]

        return x, p, weights, arg_maxs, popts
    ```
    - Function `evaluate` is used to calulate scores or save values returned by neural network. Currently it is not generic, when I run experiments for different options I modify this function to save specific values and scores. It will be fixed soon.  
