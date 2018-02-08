from .. import utils
from .. utils import lasagne as utils_lasagne
from .. datasources import maps_labelnoise as maps

import numpy as np
import lasagne.layers as L
import theano.tensor as T
import lasagne.nonlinearities as A
import lasagne.objectives as O
import lasagne.updates as U
import lasagne.init as I
import theano
import os

from collections import defaultdict
import tqdm
from functools import partial
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from scipy.ndimage import zoom


def map_from_to(y, from_fps, to_fps):
    scale_factor = to_fps / from_fps
    return zoom(y, (scale_factor, 1), order=5, mode='constant', cval=0., prefilter=True)


def eval_framewise(predictions, targets, thresh=0.5):
    """
    author: filip (+ data-format amendments by rainer)
    """
    if predictions.shape != targets.shape:
        raise ValueError('predictions.shape {} != targets.shape {} !'.format(predictions.shape, targets.shape))

    pred = predictions > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return tp.sum(), fp.sum(), 0, fn.sum()


def prf_framewise((tp, fp, tn, fn)):
    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)

    if tp + fp == 0.:
        p = 0.
    else:
        p = tp / (tp + fp)

    if tp + fn == 0.:
        r = 0.
    else:
        r = tp / (tp + fn)

    if p + r == 0.:
        f = 0.
    else:
        f = 2 * ((p * r) / (p + r))

    if tp + fp + fn == 0.:
        a = 0.
    else:
        a = tp / (tp + fp + fn)

    return p, r, f, a


def fix_used(y_true, y_pred):
    # this just swaps positional parameters
    return prf_framewise(eval_framewise(y_pred, y_true, 0.5))


def fix_should(predictions, targets, threshold=0.5):
    pred = predictions > threshold
    targ = targets > threshold

    if predictions.shape != targets.shape:
        raise ValueError('predictions.shape {} != targets.shape {} !'.format(predictions.shape, targ))

    _tp = pred & targ
    _fp = pred ^ _tp
    _fn = targ ^ _tp

    _p = np.zeros(len(predictions))
    _r = np.zeros(len(predictions))
    _f = np.zeros(len(predictions))
    _a = np.zeros(len(predictions))
    for t in range(len(predictions)):
        tp = float(_tp[t].sum())
        fp = float(_fp[t].sum())
        fn = float(_fn[t].sum())

        if tp + fp == 0.:
            p = 0.
        else:
            p = tp / (tp + fp)

        if tp + fn == 0.:
            r = 0.
        else:
            r = tp / (tp + fn)

        if p + r == 0.:
            f = 0.
        else:
            f = 2 * ((p * r) / (p + r))

        if tp + fp + fn == 0.:
            a = 0.
        else:
            a = tp / (tp + fp + fn)

        _p[t] = p
        _r[t] = r
        _f[t] = f
        _a[t] = a

    return _p.mean(), _r.mean(), _f.mean(), _a.mean()


fix_prfs = precision_recall_fscore_support


def block(net, num_filters, pool_size, suffix):
    net = L.Conv2DLayer(net, num_filters=num_filters, filter_size=(3, 3), pad='same', nonlinearity=A.rectify, W=I.HeNormal('relu'), name='conv_0_{}'.format(suffix))
    net = L.batch_norm(net, name='bn_0_{}'.format(suffix))
    net = L.Conv2DLayer(net, num_filters=num_filters, filter_size=(3, 3), nonlinearity=A.rectify, W=I.HeNormal('relu'), name='conv_1_{}'.format(suffix))
    net = L.batch_norm(net, name='bn_1_{}'.format(suffix))
    net = L.MaxPool2DLayer(net, pool_size=pool_size, name='pool_{}'.format(suffix))
    net = L.DropoutLayer(net, p=0.25, name='do_{}'.format(suffix))
    return net


class Model(object):
    def __init__(self, hyper):
        x_in = T.ftensor4('x_in')
        y_in = T.matrix('y_in')
        w_in = T.matrix('w_in')

        learning_rate = theano.shared(np.float32(hyper['learning_rate']))

        net = L.InputLayer(hyper['input_shape'], x_in, name='x_input')
        if hyper['audio_options']['fps'] == 31.25:
            net = L.Conv2DLayer(net, num_filters=32, filter_size=(3, 3), pad='same', nonlinearity=A.rectify, name='conv_0')
            net = L.batch_norm(net, name='bn_0')
            net = L.Conv2DLayer(net, num_filters=32, filter_size=(3, 3), nonlinearity=A.rectify, name='conv_1')
            net = L.batch_norm(net, name='bn_1')
            net = L.MaxPool2DLayer(net, pool_size=(1, 2), name='pool_0')
            net = L.DropoutLayer(net, p=0.25, name='do_0')

            net = L.Conv2DLayer(net, num_filters=64, filter_size=(3, 3), nonlinearity=A.rectify, name='conv_2')
            net = L.batch_norm(net, name='bn_2')
            net = L.MaxPool2DLayer(net, pool_size=(1, 2), name='pool_1')
            net = L.DropoutLayer(net, p=0.25, name='do_1')

            net = L.DenseLayer(net, num_units=512, nonlinearity=A.rectify, name='dense_0')
            net = L.batch_norm(net, name='bn_3')
            net = L.DropoutLayer(net, p=0.5, name='do_2')

            net = L.DenseLayer(net, num_units=88, nonlinearity=A.sigmoid, name='out')
        else:
            net = block(net, 32, (2, 2), 'a')
            net = block(net, 64, (1, 2), 'b')
            net = block(net, 96, (1, 2), 'c')
            net = L.Conv2DLayer(net, num_filters=128, filter_size=(3, 3), nonlinearity=A.rectify, name='conv_conv')

            net = L.DenseLayer(net, num_units=512, nonlinearity=A.rectify, name='dense_0')
            net = L.batch_norm(net, name='bn_3')
            net = L.DropoutLayer(net, p=0.5, name='do_2')

            net = L.DenseLayer(net, num_units=88, nonlinearity=A.sigmoid, name='out')

        print 'network (n_params={})'.format(utils_lasagne.get_n_params(net))
        print utils_lasagne.to_string(net)

        # IMPORTANT_LESSON: this *needs* to be AT LEAST 1e-7 !
        _EPSILON = 1e-7
        out = L.get_output(net)
        out = T.clip(out, _EPSILON, 1.0 - _EPSILON)
        loss = T.mean(O.binary_crossentropy(out, y_in) * w_in)

        out_det = L.get_output(net, deterministic=True)
        out_det = T.clip(out_det, _EPSILON, 1.0 - _EPSILON)
        loss_det = T.mean(T.nnet.binary_crossentropy(out_det, y_in))

        params = L.get_all_params(net, trainable=True)

        updates = U.nesterov_momentum(loss, params, learning_rate, 0.9)
        names = [p.name for p in params]
        grads = [T.grad(loss, p) for p in params]
        stats = [T.mean(g) for g in grads] + [T.var(g) for g in grads]

        f_grad_stat = theano.function(
            inputs=[x_in, y_in, w_in],
            outputs=stats
        )

        f_update = theano.function(
            inputs=[x_in, y_in, w_in],
            outputs=loss,
            updates=updates
        )

        f_predict = theano.function(
            inputs=[x_in, y_in],
            outputs=[loss_det, out_det]
        )

        # print 'compiling saliencies function'
        # saliencies = []
        # for i in range(88):
        #     saliency = theano.grad(out_det[:, i].sum(), wrt=x_in)
        #     saliencies.append(saliency)

        # f_saliency = theano.function(
        #     inputs=[x_in],
        #     outputs=[out_det] + saliencies
        # )

        self.x_in = x_in
        self.y_in = y_in
        self.w_in = w_in
        self.net = net
        self.loss = loss
        self.params = params
        self.learning_rate = learning_rate
        self.f_update = f_update
        self.f_predict = f_predict
        self.f_grad_stat = f_grad_stat
        # self.f_saliency = f_saliency
        self.names = names
        self.stats = stats
        self.grads = grads
        self.synth_sample_counter = Counter()

    def update(self, hyper, state, batch):
        self.synth_sample_counter.update([f[0:8] for f in batch['f']])

        loss = self.f_update(batch['x'], batch['y'], batch['w'])
        return dict(
            loss=loss
        )

    def predict(self, batch):
        loss_det, out_det = self.f_predict(batch['x'], batch['y'])
        return dict(
            loss=loss_det,
            y_true=batch['y'],
            y_pred=out_det,
            f=batch['f'],
            w=batch['w'] if 'w' in batch else None
        )

    def after_update(self, hyper, state, update_result):
        if np.isinf(update_result['loss']):
            raise utils.StopTraining('encountered Inf ({})'.format(update_result['loss']))
        elif np.isnan(update_result['loss']):
            raise utils.StopTraining('encountered NaN ({})'.format(update_result['loss']))

        if state['i_update'] % hyper['log_frequency'] == 0:
            state['train_summary'].append(update_result)

    def after_validation(self, hyper, state, valid_accumulator):
        if state['i_epoch'] % hyper['stepwise_lr']['n_epoch'] == 0 and state['i_epoch'] > 0:
            lr = self.learning_rate.get_value()
            nlr = np.float32(lr * hyper['stepwise_lr']['factor'])
            print 'changing the learning_rate {} -> {}'.format(lr, nlr)
            self.learning_rate.set_value(nlr)

        mean_loss = np.mean(map(lambda vi: vi['loss'], valid_accumulator))
        y_pred = np.vstack(map(lambda vi: vi['y_pred'], valid_accumulator))
        y_true = np.vstack(map(lambda vi: vi['y_true'], valid_accumulator))

        p, r, f, a = prf_framewise(eval_framewise(y_pred, y_true, 0.5))

        print 'prfa Sv', p, r, f, a

        print 'mean(loss(Sv))', mean_loss
        state['valid_summary'].append(dict(
            valid_loss=mean_loss,
            precision=p,
            recall=r,
            fmeasure=f,
            accuracy=a,
            actual_learning_rate=self.learning_rate.get_value()
        ))
        # IMPORTANT LESSON (TODO: think about this, long and hard!)
        # we'd like to *maximize fmeasure*, not minimize valid-loss ?! let's do both...

        ##############################################################
        weights_link = os.path.join(state['run_path'], 'weights.pkl')
        weights_min_valid_loss = os.path.join(state['run_path'], 'weights_min_valid_loss.pkl')
        weights_max_valid_fmeasure = os.path.join(state['run_path'], 'weights_max_valid_fmeasure.pkl')
        weights_current = os.path.join(state['run_path'], 'current_weights.pkl')

        valid_losses = map(lambda vi: vi['valid_loss'], state['valid_summary'])
        min_valid_loss_index = np.argmin(valid_losses)

        # if the youngest entry is the lowest -- save model-state
        if min_valid_loss_index == (len(valid_losses) - 1):
            self.save(weights_min_valid_loss)

        ##############################################################
        valid_fmeasures = map(lambda vi: vi['fmeasure'], state['valid_summary'])
        max_fmeasure_index = np.argmax(valid_fmeasures)

        # if the youngest entry is the highest -- save model-state
        if max_fmeasure_index == (len(valid_losses) - 1):
            self.save(weights_max_valid_fmeasure)

        ##############################################################
        # save *current* weights
        self.save(weights_current)

        # this is (l)exists for links!
        if os.path.lexists(weights_link):
            os.remove(weights_link)

        if os.path.exists(weights_max_valid_fmeasure):
            print 'linking to', weights_max_valid_fmeasure
            os.symlink('weights_max_valid_fmeasure.pkl', weights_link)
        elif os.path.exists(weights_min_valid_loss):
            print 'linking to', weights_min_valid_loss
            os.symlink('weights_min_valid_loss.pkl', weights_link)
        elif os.path.exists(weights_current):
            print 'linking to', weights_current
            os.symlink('current_weights.pkl', weights_current)
        else:
            print 'no linking ?!'

        ##############################################################
        # save synth_sample_counter
        utils.dump(self.synth_sample_counter, os.path.join(state['run_path'], 'synth_sample_counter.pkl'))
        ##############################################################
        # save state after each episode
        utils.dump(state, os.path.join(state['run_path'], 'state.pkl'))

    def before_training(self, hyper, state):
        pass

    def after_training(self, hyper, state):
        utils.dump(state, os.path.join(state['run_path'], 'state.pkl'))

    def before_testing(self, hyper, state):
        pass

    def after_testing(self, hyper, state, test_summary):
        # this is a nasty workaround for stuff that is actually on a remote machine ...
        if not os.path.exists(state['run_path']):
            os.mkdir(state['run_path'])

        mean_loss = np.mean(map(lambda ti: ti['loss'], test_summary))
        print 'mean(loss(T))', mean_loss

        y_pred = np.vstack(map(lambda ti: ti['y_pred'], test_summary))
        y_true = np.vstack(map(lambda ti: ti['y_true'], test_summary))

        # do individual_eval
        native_fps_filename_y_pred = defaultdict(list)
        native_fps_filename_y_true = defaultdict(list)
        for ti in test_summary:
            native_fps_filename_y_pred[ti['f']].append(ti['y_pred'])
            native_fps_filename_y_true[ti['f']].append(ti['y_true'])

        # save for potential re-run at the last second ...
        utils.dump(
            native_fps_filename_y_pred,
            os.path.join(state['run_path'], 'native_fps_filename_y_pred.pkl')
        )
        utils.dump(
            native_fps_filename_y_true,
            os.path.join(state['run_path'], 'native_fps_filename_y_true.pkl')
        )

        # obtain high resolution groundtruth
        # do first 30s eval @ 100 fps
        from_fps = hyper['audio_options']['fps']
        to_fps = 100
        if from_fps == 31.25:
            N = 937
        else:
            N = 3000

        high_fps_filename_y_pred = dict()
        high_fps_filename_y_true = dict()
        for filename in native_fps_filename_y_true.keys():
            native_fps_y_pred = np.vstack(native_fps_filename_y_pred[filename])[0:N]
            high_fps_y_pred_mapped = map_from_to(native_fps_y_pred, from_fps, to_fps)

            high_fps_filename_y_pred[filename] = high_fps_y_pred_mapped

            midifilename = os.path.join(hyper['audio_path'], filename + '.mid')
            # obtain groundtruth with the higher framerate
            high_fps_y_true, _ = maps.midi_to_groundtruth_a(
                midifilename,
                1. / to_fps,
                len(high_fps_y_pred_mapped),
                maps.get_no_weighting(),
                maps.get_postprocess_none()
            )

            high_fps_filename_y_true[filename] = high_fps_y_true

        # save for potential re-run at the last second ...
        utils.dump(
            high_fps_filename_y_pred,
            os.path.join(state['run_path'], 'high_fps_filename_y_pred.pkl')
        )
        utils.dump(
            high_fps_filename_y_true,
            os.path.join(state['run_path'], 'high_fps_filename_y_true.pkl')
        )

        native_fps_all_y_pred = []
        native_fps_all_y_true = []
        for filename in native_fps_filename_y_true.keys():
            native_fps_all_y_pred.append(np.vstack(native_fps_filename_y_pred[filename])[0:N])
            native_fps_all_y_true.append(np.vstack(native_fps_filename_y_true[filename])[0:N])
        native_fps_all_y_pred = np.vstack(native_fps_all_y_pred)
        native_fps_all_y_true = np.vstack(native_fps_all_y_true)
        native_fps_result = fix_used(native_fps_all_y_true, native_fps_all_y_pred)

        high_fps_all_y_pred = []
        high_fps_all_y_true = []
        for filename in high_fps_filename_y_true.keys():
            high_fps_all_y_pred.append(high_fps_filename_y_pred[filename])
            high_fps_all_y_true.append(high_fps_filename_y_true[filename])
        high_fps_all_y_pred = np.vstack(high_fps_all_y_pred)
        high_fps_all_y_true = np.vstack(high_fps_all_y_true)
        high_fps_result = fix_used(high_fps_all_y_true, high_fps_all_y_pred)

        utils.dump(native_fps_result, os.path.join(state['run_path'], 'native_fps_result.pkl'))
        utils.dump(high_fps_result, os.path.join(state['run_path'], 'high_fps_result.pkl'))

        print 'prfa T first 30s @{} fps {} {} {} {}'.format(from_fps, *native_fps_result)
        print 'prfa T first 30s @{} fps {} {} {} {}'.format(to_fps, *high_fps_result)

    def before_postprocess(self, *a, **b):
        pass

    def after_postprocess(self, *a, **b):
        pass

    def postprocess(self, hyper, state, _a, _b, _c):
        self.postprocess_fold(hyper, state, 'train')
        self.postprocess_fold(hyper, state, 'valid')
        self.postprocess_fold(hyper, state, 'test')

    def postprocess_fold(self, hyper, state, fold_name):
        _, _, contextsize, _ = hyper['input_shape']
        fold_it = maps.get_fold_iterator(
            basedir=hyper['audio_path'],
            foldname=hyper['{}_fold'.format(fold_name)],
            batchsize=hyper['batchsize'],
            x_contextsize=contextsize,
            y_contextsize=0,
            # we'll use the weighting for *detailed* error analysis
            weighting=maps.get_phase_indexing_function()
        )

        def gen():
            batch_it = next(fold_it)
            for x, y, w, i, f in batch_it:
                yield dict(
                    x=x,
                    y=y,
                    w=w,
                    i=i,
                    f=f
                )

        accumulator = []
        for batch in tqdm.tqdm(gen()):
            accumulator.append(self.predict(batch))

        # do individual_eval
        filename_y_pred = defaultdict(list)
        filename_y_true = defaultdict(list)
        filename_y_phase = defaultdict(list)
        for ti in accumulator:
            filename_y_pred[ti['f']].append(ti['y_pred'])
            filename_y_true[ti['f']].append(ti['y_true'])
            filename_y_phase[ti['f']].append(ti['w'])

        for filename in filename_y_pred.keys():
            y_pred = np.vstack(filename_y_pred[filename])
            y_true = np.vstack(filename_y_true[filename])
            y_phase = np.vstack(filename_y_phase[filename])
            parent = os.path.join(state['run_path'], 'activations_{}'.format(fold_name), filename)
            if not os.path.exists(parent):
                os.makedirs(parent)

            utils.dump(y_true, os.path.join(parent, 'y_true.pkl'))
            utils.dump(y_pred, os.path.join(parent, 'y_pred.pkl'))
            utils.dump(y_phase, os.path.join(parent, 'y_phase.pkl'))

    def visualize(self, hyper, state, train_it, valid_it, test_it):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as grd
        batchsize, _, contextsize, n_bins = hyper['input_shape']
        n_notes = 88

        # compute different saliency maps ...
        def vis(grid, y_hat, saliencies, r):
            for i, ax, y_hat_i, saliency in zip(range(len(y_hat)), grid, y_hat, saliencies):
                im = ax.imshow(saliency.T, origin='lower', cmap='seismic', vmin=-r, vmax=r)
                ax.set_xticks([])
                ax.set_yticks([])
                if y_hat_i > 0.5:
                    ax.set_xticks([(contextsize - 1) / 2])  # in the middle ...
                    ax.set_xticklabels([str(i)])  # put the (shifted) note number

            return im

        for batch in train_it:
            y_hat_saliencies = self.f_saliency(batch['x'])
            y_hat = y_hat_saliencies[0]
            saliencies = y_hat_saliencies[1:]

            for bi in range(batchsize):
                sample_saliencies = [s[bi].reshape(contextsize, n_bins) for s in saliencies]
                sample_y_hat = y_hat[bi].flatten()

                vmin = np.inf
                vmax = -np.inf
                for s in sample_saliencies:
                    vmin = min(vmin, np.min(s))
                    vmax = max(vmax, np.max(s))

                r = max(abs(vmin), abs(vmax))

                fig = plt.figure()
                n_cols = 1 + n_notes + 2
                grid_spec = grd.GridSpec(1, n_cols, width_ratios=[1] * n_cols, wspace=1.1)

                grid = [plt.subplot(gs) for gs in grid_spec]

                ################################################################
                # show spectrogram in the first grid cell
                spec_im = grid[0].imshow(batch['x'][bi].reshape(contextsize, n_bins).T, origin='lower')
                grid[0].set_xticks([])
                grid[1].set_yticks([])

                ################################################################
                # show all saliencies
                sali_im = vis(grid[1:], sample_y_hat, sample_saliencies, r)

                ################################################################
                # plot colorbars separately
                spec_cax = grid[-2]
                sali_cax = grid[-1]
                fig.colorbar(spec_im, cax=spec_cax, use_gridspec=False)
                fig.colorbar(sali_im, cax=sali_cax, use_gridspec=False)

                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)

                plt.show()

    def save(self, filename):
        params = L.get_all_params(self.net)
        utils.dump([p.get_value() for p in params], filename)

    def load(self, filename):
        params = L.get_all_params(self.net)
        values = utils.load(filename)
        for p, v in zip(params, values):
            p.set_value(v)


def build_model(hyper):
    return Model(hyper)


def showstate(hyper, state):
    import matplotlib.pyplot as plt
    train_summary = state['train_summary']
    valid_summary = state['valid_summary']
    keys = train_summary[0].keys()
    print 'train_summary keys', keys
    fig, ax = plt.subplots()
    ax.set_title('train_summary: runkey={}'.format(hyper['run_key']))
    for key in keys:
        ax.plot(map(lambda ts: ts[key], train_summary), label=key)

    ax.legend().draggable()
    plt.show()

    keys = valid_summary[0].keys()
    print 'valid_summary keys', keys
    fig, ax = plt.subplots()
    ax.set_title('valid_summary: runkey={}'.format(hyper['run_key']))
    for key in keys:
        ax.plot(map(lambda ts: ts[key], valid_summary), label=key)

    ax.legend().draggable()
    plt.show()


def summarize(hypers, states):
    utils.dump(hypers, 'hypers.pkl')
    utils.dump(states, 'states.pkl')


def get_train_iterator(hyper):
    get_weight_function = getattr(maps, hyper['weight_function']['name'])
    get_postprocess_function = getattr(maps, hyper['postprocess_function']['name'])
    midi_to_groundtruth = getattr(maps, 'midi_to_groundtruth_{}'.format(hyper['labelfunction']))

    _, _, contextsize, _ = hyper['input_shape']

    if hyper['labelfunction'] in ['e', 'f']:
        midi_to_groundtruth = partial(midi_to_groundtruth, np.random.RandomState())

    it_function = None
    if hyper['stratify'] is True:
        it_function = maps.stratified_random_sample
    elif hyper['stratify'] is False:
        it_function = maps.uniformly_random_sample
    elif hyper['stratify'] == 'notes':
        it_function = maps.fully_stratified_random_sample
    elif hyper['stratify'] == 'single_notes':
        it_function = maps.single_note_stratified_random_sample

    srs_it = it_function(
        basedir=hyper['audio_path'],
        foldname=hyper['train_fold'],
        batchsize=hyper['batchsize'],
        audio_options=hyper['audio_options'],
        midi_to_groundtruth=midi_to_groundtruth,
        x_contextsize=contextsize,
        y_contextsize=0,
        weighting=get_weight_function(**hyper['weight_function']['params']),
        postprocess_weighting=get_postprocess_function(**hyper['postprocess_function']['params']),
        rng=np.random
    )

    def it():
        while True:
            x, y, w, i, f = next(srs_it)
            yield dict(
                x=x,
                y=y,
                w=w,
                i=i,
                f=f
            )

    return it


def get_valid_iterator(hyper):
    _, _, contextsize, _ = hyper['input_shape']
    fold_it = maps.get_fold_iterator(
        basedir=hyper['audio_path'],
        foldname=hyper['valid_fold'],
        audio_options=hyper['audio_options'],
        midi_to_groundtruth=maps.midi_to_groundtruth_a,
        batchsize=hyper['batchsize'],
        x_contextsize=contextsize,
        y_contextsize=0
    )

    def it():
        batch_it = next(fold_it)
        for x, y, i, f in batch_it:
            yield dict(
                x=x,
                y=y,
                i=i,
                f=f
            )

    return it


def get_test_iterator(hyper):
    _, _, contextsize, _ = hyper['input_shape']
    fold_it = maps.get_fold_iterator(
        basedir=hyper['audio_path'],
        foldname=hyper['test_fold'],
        audio_options=hyper['audio_options'],
        midi_to_groundtruth=maps.midi_to_groundtruth_a,
        batchsize=hyper['batchsize'],
        x_contextsize=contextsize,
        y_contextsize=0
    )

    def it():
        batch_it = next(fold_it)
        for x, y, i, f in batch_it:
            yield dict(
                x=x,
                y=y,
                i=i,
                f=f
            )

    return it
