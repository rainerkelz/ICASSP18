import cPickle
import hashlib
import traceback
import numpy as np
from collections import OrderedDict, Counter, defaultdict
import pyaudio
from matplotlib.widgets import SpanSelector
import colors


def canonicalize_audio_options(_audio_options, mmspec):
    audio_options = dict(_audio_options)

    if 'spectrogram_type' in audio_options:
        spectype = getattr(mmspec, audio_options['spectrogram_type'])
        del audio_options['spectrogram_type']
    else:
        spectype = getattr(mmspec, 'LogarithmicFilteredSpectrogram')

    if 'filterbank' in audio_options:
        audio_options['filterbank'] = getattr(mmspec, audio_options['filterbank'])
    else:
        audio_options['filterbank'] = getattr(mmspec, 'LogarithmicFilterbank')

    return spectype, audio_options


def eval_framewise(targets, predictions, thresh=0.5):
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


def prfa_framewise((tp, fp, tn, fn)):
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


def dict_hash(d):
    m = hashlib.sha1()

    if isinstance(d, dict):
        for _, value in sorted(d.items(), key=lambda (k, v): k):
            m.update(dict_hash(value))
    else:
        m.update(str(d))

    return m.hexdigest()


def dump(obj, filename):
    cPickle.dump(obj, open(filename, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)


def load(filename):
    return cPickle.load(open(filename, 'r'))


def get_trace(ex):
    return traceback.format_exc(ex)


def start_end_iterator(n, batchsize):
    x, r = divmod(n, batchsize)
    if r != 0:
        x += 1

    for i in xrange(0, x):
        start = i * batchsize
        end = min(n, (i + 1) * batchsize)
        yield start, end


def start_end_iterator_batches(n_batches, batchsize):
    for i in xrange(0, n_batches):
        start = i * batchsize
        end = (i + 1) * batchsize
        yield start, end


def indices_consecutive(indices, batchsize):
    length = len(indices)
    start = 0
    end = start + batchsize
    run = True
    while run:
        while end < length:
            yield indices[start:end]
            start += batchsize
            end += batchsize

        # start   length  end
        # |.......|........|
        #    a       b

        # end > high here
        rest = end - length

        # save the indices from the last batch ### MUST do a COPY here! ###
        a = indices[start:length]

        # draw the missing examples for the last batch from the newly shuffled array
        b = indices[0:rest]

        # yield the concatenation of the rest + new indices
        yield np.concatenate([a, b])
        run = False


def indices_without_replacement(_indices, batchsize, rng, finite=True):
    indices = _indices.copy()
    length = len(indices)
    start = 0
    end = start + batchsize
    rng.shuffle(indices)
    while True:
        while end < length:
            yield indices[start:end].copy()

            start += batchsize
            end += batchsize

        # start   _length  end
        # |.......|........|
        #    a       b

        # end > high here
        rest = end - length

        # save the indices from the last batch ### MUST do a COPY here! ###
        a = indices[start:length].copy()

        # re-shuffle the new indices ### BECAUSE THIS IS IN-PLACE ###
        rng.shuffle(indices)

        # draw the missing examples for the last batch from the newly shuffled array
        b = indices[0:rest]

        # yield the concatenation of the rest + new indices
        yield np.concatenate([a, b])

        if finite:
            break

        # start at pos '0 + rest' into the newly shuffled array ...
        start = rest
        end = start + batchsize


def wrap_function(theano_function,
                  inputs,
                  outputs,
                  mode=None,
                  updates=None,
                  givens=None,
                  no_default_updates=False,
                  accept_inplace=False,
                  name=None,
                  rebuild_strict=True,
                  allow_input_downcast=None,
                  profile=None,
                  on_unused_input='raise'):

    if not isinstance(outputs, OrderedDict):
        raise RuntimeError('outputs is not an ordered dictionary!')

    compiled_f = theano_function(inputs, outputs.values(), mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)

    def wrapped_f(*args, **kwargs):
        results = compiled_f(*args, **kwargs)

        mapping = dict()
        for output_key, result in zip(outputs.keys(), results):
            mapping[output_key] = result

        return mapping

    return wrapped_f


class Context2D4D(object):
    def __init__(self, array, context_size, padding=0.):
        if len(array.shape) != 2:
            raise ValueError('this class is for 2D -> 4D transformations only')

        if context_size % 2 == 0:
            raise ValueError('this class can only cope with odd values for context_size')

        self.array = array
        self.context_size = context_size
        self.csh = (self.context_size - 1) / 2
        self.padding = padding
        self.n = self.array.shape[0]
        self.m = self.array.shape[1]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_context(np.array([index]))

        elif isinstance(index, slice):
            if index.step is not None and index.step != 1:
                raise ValueError('no step sizes != 1 allowed "{}"'.format(index))

            return self._get_context(np.arange(index.start, index.stop))

        elif isinstance(index, np.ndarray):
            return self._get_context(index)

        raise RuntimeError('unrecognised type for index ({})'.format(type(index)))

    def _get_context(self, indices):
        if (indices < 0).any():
            raise ValueError('no negative indices allowed')

        shape = (len(indices), 1, self.context_size, self.m)
        context = np.zeros(shape).astype(self.array.dtype)
        context.fill(self.padding)

        for i, index in enumerate(indices):
            start = index - self.csh
            end = index + self.csh + 1

            c_start = 0
            c_end = self.context_size

            if start < 0:
                c_start = abs(start)
                start = 0

            if end > self.n:
                c_end = self.context_size - (end - self.n)
                end = self.n

            if start > self.n:
                return np.zeros(shape).astype(self.array.dtype)

            context[i, 0, c_start:c_end, :] = self.array[start:end, :]
        return context

    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        return (len(self.array), 1, self.context_size, self.m)


class Context(object):
    def __init__(self, array, context_lo, context_hi, hop_size, padding_value=0.0):
        self.array = array
        self.context_lo = context_lo
        self.context_hi = context_hi
        self.context_size = self.context_lo + 1 + self.context_hi
        self.hop_size = hop_size
        self.padding_value = padding_value

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_context(np.array([index]))

        elif isinstance(index, slice):
            if index.step is not None and index.step != 1:
                raise ValueError('no step sizes != 1 allowed "{}"'.format(index))

            return self._get_context(np.arange(index.start, index.stop))

        elif isinstance(index, np.ndarray):
            return self._get_context(index)

        raise RuntimeError('unrecognised type for index ({})\n{}'.format(type(index), index))

    def _get_context(self, indices):
        if (indices < 0).any():
            raise ValueError('no negative indices allowed')

        indices = indices * self.hop_size

        shape = (indices.shape[0], self.context_size,) + self.array.shape[1:]
        context = np.zeros(shape).astype(self.array.dtype)
        context.fill(self.padding_value)
        for i, index in enumerate(indices):
            start = index - self.context_lo
            end = index + self.context_hi + 1

            c_start = 0
            c_end = self.context_size

            if start < 0:
                c_start = abs(start)
                start = 0

            if end > self.array.shape[0]:
                c_end = self.context_size - (end - self.array.shape[0])
                end = self.array.shape[0]

            if start > self.array.shape[0]:
                return np.zeros(shape).astype(self.array.dtype)
            # print 'c_start', c_start
            # print 'c_end', c_end
            # print 'start', start
            # print 'end', end
            context[i, c_start:c_end] = self.array[start:end]
        return context

    def __len__(self):
        l, r = divmod(self.array.shape[0], self.hop_size)
        p1 = 1 if r > 0 else 0
        return l + p1

    @property
    def shape(self):
        return (self.__len__(), self.context_size,) + self.array.shape[1:]

    @property
    def dtype(self):
        return self.array.dtype


class StopTraining(Exception):
    pass


class StopTesting(Exception):
    pass


class NumericalException(Exception):
    pass


class ExternalInterrupt(Exception):
    pass


class UpdateableHistogram(object):
    """
    this is the simplest possible updateable histogram.
    it uses a fixed range, and does
    *** IGNORE ANY VALUES ***
    which fall outside the range [xmin, xmax] !!!
    """
    def __init__(self, xmin, xmax, n_bins):
        self.xmin = xmin
        self.xmax = xmax
        self.n_bins = n_bins
        self._binedges = np.linspace(self.xmin, self.xmax, self.n_bins + 1)
        self._bincounts = np.zeros(self.n_bins)
        self._bincenters = (self._binedges[1:] + self._binedges[0:-1]) / 2.

    def update(self, x):
        _x = x[x > self.xmin]
        _y = _x[_x < self.xmax]
        self._bincounts = self._bincounts + np.bincount(np.digitize(_y, self._binedges), minlength=self.n_bins + 1)[1:]

    def density(self):
        densitybins = np.array(np.diff(self._binedges), np.float)
        return self._bincounts / densitybins / self._bincounts.sum()

    def bincenters(self):
        return self._bincenters

    def merge(self, other):
        self._bincounts += other._bincounts


def ahash(a):
    if a.dtype == np.int8:
        a.flags.writeable = False
        return hash(a.data)
    else:
        raise ValueError


class NoteCounter(object):
    def __init__(self, y=None, tonal_range=88):
        self.tonal_range = tonal_range
        self.notes = dict()
        self.note_counts = Counter()
        if y is not None:
            self.update(y)

    def update(self, y):
        if isinstance(y, np.ndarray):
            for t in xrange(len(y)):
                y_frame = y[t, :].copy()
                y_frame_key = ahash(y_frame)
                self.notes[y_frame_key] = y_frame
                self.note_counts[y_frame_key] += 1

        if isinstance(y, NoteCounter):
            self.notes.update(y.notes)
            self.note_counts.update(y.note_counts)

    def keys(self):
        return self.notes.keys()

    def items(self):
        return self.note_counts.items()

    def items_dense_keys(self):
        keys = self.notes.keys()
        for key in keys:
            yield self.notes[key], self.note_counts[key]

    def items_sparse_keys(self):
        keys = self.notes.keys()
        for key in keys:
            yield sorted(np.nonzero(self.notes[key])[0].tolist()), self.note_counts[key]

    def intersect(self, other):
        result = NoteCounter()
        result.note_counts = self.note_counts & other.note_counts
        for key in result.note_counts:
            result.notes[key] = self.notes[key]
        return result

    def union(self, other):
        result = NoteCounter(tonal_range=self.tonal_range)
        result.note_counts = self.note_counts | other.note_counts
        result.notes.update(self.notes)
        result.notes.update(other.notes)
        return result

    def remove(self, other):
        result = NoteCounter(tonal_range=self.tonal_range)
        skeys = self.notes.keys()
        okeys = set(other.notes.keys())

        for key in skeys:
            # copy only if its not in the other
            if key not in okeys:
                result.notes[key] = self.notes[key]
                result.note_counts[key] = self.note_counts[key]
        return result

    def canonicalize(self, _key):
        if isinstance(_key, list):
            k = np.zeros(self.tonal_range).astype(np.int8)
            for n in _key:
                k[n] = 1

            return ahash(k)
        elif isinstance(_key, np.ndarray):
            if _key.dtype == np.int8:
                return ahash(_key)
        elif isinstance(_key, int):
            return _key

        raise ValueError('key must be either from type <list of integers from [0:87]> or <ndarray> or <int>')

    def get(self, _key, default=None):
        key = self.canonicalize(_key)
        return self.note_counts.get(key, default)

    def get_sparse(self, _key, default=None):
        key = self.canonicalize(_key)
        return np.nonzero(self.notes[key])[0].tolist()

    def __getitem__(self, _key):
        key = self.canonicalize(_key)
        return self.note_counts[key]

    # def __setitem__(self, _key, item):
    #     key = self.canonicalize(_key)
    #     self.notes[key] = _key
    #     self.note_counts[key] = item

    def most_common(self, n):
        return self.note_counts.most_common(n)

    def most_common_dense(self, n):
        return [self.notes[k] for k, c in self.note_counts.most_common(n)]

    def most_common_sparse(self, n):
        return [np.nonzero(self.notes[k])[0].tolist() for k, c in self.note_counts.most_common(n)]


class ReverseNoteIndex(object):
    def __init__(self):
        self.index = defaultdict(list)

    def update(self, piece_index, y):
        if not isinstance(y, np.ndarray):
            raise ValueError('not an ndarray')
        if not y.dtype == np.int8:
            raise ValueError('ndarray not int8')

        # t is the within-piece-index
        for t in xrange(len(y)):
            y_frame = y[t, :].copy()
            y_frame_key = ahash(y_frame)
            self.index[y_frame_key].append((piece_index, t))

    def keys(self):
        return self.index.keys()

    def sample(self):
        # choose combinations uniformly at random
        random_combination_index = np.random.randint(0, len(self.index))

        # TODO: is the order of keys random ? is this 'doubly stochastic' ?
        keys = self.index.keys()
        pieces_times = self.index[keys[random_combination_index]]

        # choose a piece and a time uniformly at random
        # TODO: could stratify here by synth as well ! (too much effort?)
        random_piece_time_index = np.random.randint(0, len(pieces_times))
        return pieces_times[random_piece_time_index]


class SimpleNoteCounter(object):
    def __init__(self, y=None):
        self.note_counts = Counter()
        if y is not None:
            self.update(y)

    def update(self, y, inc=1):
        if isinstance(y, np.ndarray):
            if len(y.shape) == 2:
                for t in xrange(len(y)):
                    key = tuple(np.nonzero(y[t, :])[0])
                    self.note_counts[key] += inc
            elif len(y.shape) == 1:
                key = tuple(np.nonzero(y)[0])
                self.note_counts[key] += inc
            else:
                raise ValueError('unprocessable shape?')

        if isinstance(y, SimpleNoteCounter):
            if inc != 1:
                raise ValueError('the "increment" is going to be ignored!?')
            self.note_counts.update(y.note_counts)

        if isinstance(y, list):
            self.note_counts[tuple(y)] += inc

        if isinstance(y, tuple):
            self.note_counts[y] += inc

        if isinstance(y, int):
            self.note_counts[tuple(y, )] += inc

    def keys(self):
        return self.note_counts.keys()

    def items(self):
        return self.note_counts.items()

    def intersect(self, other):
        result = SimpleNoteCounter()
        result.note_counts = self.note_counts & other.note_counts
        return result

    def union(self, other):
        result = SimpleNoteCounter()
        result.note_counts = self.note_counts | other.note_counts
        return result

    def remove(self, other):
        result = SimpleNoteCounter()
        result.note_counts = self.note_counts - other.note_counts
        return result

    def remove_keys(self, other):
        result = SimpleNoteCounter()
        keys_to_remove = set(other.keys())
        for key, count in self.note_counts.items():
            if key in keys_to_remove:
                pass
            else:
                result.note_counts[key] = count
        return result

    def canonicalize(self, _key):
        if isinstance(_key, np.ndarray):
            return tuple(np.nonzero(_key)[0])
        elif isinstance(_key, int):
            return (_key, )
        elif isinstance(_key, list):
            return tuple(_key)
        elif isinstance(_key, tuple):
            return _key

        raise ValueError('key must be either from type <list/tuple> of integers or <ndarray> or <int>')

    def get(self, _key, default=None):
        key = self.canonicalize(_key)
        return self.note_counts.get(key, default)

    def __getitem__(self, _key):
        key = self.canonicalize(_key)
        return self.note_counts[key]

    def most_common(self, n=None):
        return self.note_counts.most_common(n)

    def __len__(self):
        return len(self.note_counts)

    def __str__(self):
        return str(self.note_counts)


# this needs to keep the association between note and phase ?
class SimplePhaseAggregator(object):
    def __init__(self, y=None, p=None):
        self.note_phases = defaultdict(list)
        self.note_combination_phases = defaultdict(list)
        if y is not None and p is not None:
            self.update(y, p)

    def update(self, y, p):
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            if len(y.shape) == 2 and y.shape == p.shape:
                for t in xrange(len(y)):
                    active = np.nonzero(y[t, :])[0]
                    key = tuple(active)
                    for note in key:
                        self.note_phases[note].append(p[t, note])

                    self.note_combination_phases[key].extend(p[t, active].tolist())
            elif len(y.shape) == 1 and y.shape == p.shape:
                active = np.nonzero(y)[0]
                key = tuple(active)
                for note in key:
                    self.note_phases[note].append(p[note])
                self.note_combination_phases[key].extend(p[active].tolist())
            else:
                raise ValueError('unprocessable shape?')
        else:
            raise ValueError('input needs to be a phases array (y={}), (p={})'.format(type(y), type(p)))

    def keys(self):
        return self.note_phases.keys()

    def values(self):
        return self.note_phases.values()

    def items(self):
        return self.note_phases.items()

    def canonicalize(self, _key):
        if isinstance(_key, np.ndarray):
            return tuple(np.nonzero(_key)[0])
        elif isinstance(_key, int):
            return (_key, )
        elif isinstance(_key, list):
            return tuple(_key)
        elif isinstance(_key, tuple):
            return _key

        raise ValueError('key must be either from type <list/tuple> of integers or <ndarray> or <int>')

    def get(self, _key, default=None):
        key = self.canonicalize(_key)
        return self.note_phases.get(key, default)

    def __getitem__(self, _key):
        key = self.canonicalize(_key)
        return self.note_phases[key]

    def most_common(self, n=None):
        return self.note_phases.most_common(n)

    def __len__(self):
        return len(self.note_phases)


# TODO: come up with a nicer name, b/c this stratifies all examples for a note
class ReverseSingleNoteIndex(object):
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))

    def update(self, piece_index, y):
        if not isinstance(y, np.ndarray):
            raise ValueError('not an ndarray')
        for t in xrange(len(y)):
            notes = tuple(np.nonzero(y[t, :])[0])
            if len(notes) == 0:  # silence!
                self.index[-1][0].append((piece_index, t))
            for note in notes:
                self.index[note][notes].append((piece_index, t))

    def keys(self):
        return self.index.keys()

    def sample(self):
        note_keys = self.index.keys()

        # choose a note uniformly at random
        random_note_key = note_keys[np.random.randint(0, len(note_keys))]

        note_combination_keys = self.index[random_note_key].keys()

        # choose a note combination (containing the previously chosen random note) uniformly at random
        random_note_combination_key = note_combination_keys[np.random.randint(0, len(note_combination_keys))]

        pieces_times = self.index[random_note_key][random_note_combination_key]

        # choose a piece and a time uniformly at random
        return pieces_times[np.random.randint(0, len(pieces_times))]

    def sample_single_note(self, note_key, ratio=0.5):
        decision = np.random.uniform(0, 1)
        if decision < ratio:
            note_combination_keys = self.index[note_key].keys()

            # choose a note combination (containing the previously chosen random note) uniformly at random
            random_note_combination_key = note_combination_keys[np.random.randint(0, len(note_combination_keys))]

            pieces_times = self.index[note_key][random_note_combination_key]

            # choose a piece and a time uniformly at random
            return pieces_times[np.random.randint(0, len(pieces_times))]
        else:
            return self.sample()


class AudioPlayer(object):
    def __init__(self, ax, _waveform, sample_rate=44100, scaling=2 ** 15):
        self.span = SpanSelector(
            ax,
            self.onselect,
            'horizontal',
            useblit=True,
            rectprops=dict(alpha=0.5, facecolor='red')
        )
        ax.plot(_waveform)
        self.waveform = (_waveform * 0.98 * scaling).astype(np.int16)

        self.silence = np.zeros(4096, dtype=np.int16)
        self.t_end = len(self.waveform)
        self.t_cursor = self.t_end
        self.sample_rate = sample_rate

    def callback(self, in_data, frame_count, time_info, status):
        data = self.silence[0:frame_count]

        if self.t_cursor < self.t_end:
            start = self.t_cursor
            end = self.t_cursor + frame_count
            data = self.waveform[start:end]
            self.t_cursor += frame_count

        # pyaudio auto-stops, if len(data) < frame_count !
        if len(data) < frame_count:
            data = np.hstack([data, self.silence[0: frame_count - len(data)]])

        return (data, pyaudio.paContinue)

    def onselect(self, _xmin, _xmax):
        self.t_cursor = max(0, int(_xmin))
        self.t_end = min(len(self.waveform), int(_xmax))

    def start(self):
        self.pya = pyaudio.PyAudio()
        self.stream = self.pya.open(
            format=self.pya.get_format_from_width(2),
            channels=1,
            rate=self.sample_rate,
            start=True,
            output=True,
            stream_callback=self.callback
        )

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()


def minmax(x, symm):
    if symm:
        r = np.max(np.abs(x))
        vmin = -r
        vmax = r
    else:
        vmin = np.min(x)
        vmax = np.max(x)
    return vmin, vmax


def flip(x, preference):
    rows, cols = x.shape
    if preference == 'horizontal':
        if rows > cols:
            return x.T
    else:
        if rows < cols:
            return x.T
    return x


def tensor_plot_helper(ax, x, cmap, symm, origin='lower', interpolation='nearest', preference='horizontal'):
    if len(x.shape) == 1:
        vmin, vmax = minmax(x, symm)
        x = x.reshape(-1, 1)
        im = ax.imshow(flip(x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    if len(x.shape) == 2:
        w, h = x.shape
        vmin, vmax = minmax(x, symm)
        im = ax.imshow(flip(x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    elif len(x.shape) == 3:
        n, w, h = x.shape
        a = b = int(np.ceil(np.sqrt(n)))

        ai = 0
        matrices = []
        for i in range(a):
            row = []
            for j in range(b):
                if ai < n:
                    cell = np.pad(x[ai], (1,), mode='constant', constant_values=0.)
                    ai += 1
                else:
                    cell = np.zeros((w + 2, h + 2))
                row.append(cell)
            matrices.append(row)

        _x = np.bmat(matrices)
        vmin, vmax = minmax(_x, symm)
        im = ax.imshow(flip(_x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    elif len(x.shape) == 4:
        a, b, w, h = x.shape
        matrices = []
        for i in range(a):
            row = []
            for j in range(b):
                row.append(np.pad(x[i, j], (1,), mode='constant', constant_values=0.))
            matrices.append(row)

        _x = np.bmat(matrices)
        vmin, vmax = minmax(x, symm)
        im = ax.imshow(flip(_x, preference), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation)
    return im
