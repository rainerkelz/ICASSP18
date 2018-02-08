from .. import utils
import madmom.audio.spectrogram as mmspec
from madmom.utils import midi
from itertools import cycle
from scipy.signal import convolve
import numpy as np
import os
from functools import partial


synthnames = [
    'ENSTDkCl',
    'ENSTDkAm',
    'StbgTGd2',
    'SptkBGCl',
    'SptkBGAm',
    'AkPnStgb',
    'AkPnCGdD',
    'AkPnBsdf',
    'AkPnBcht'
]


def no_weighting(w, *args):
    return w


def get_no_weighting():
    return no_weighting


def nopp(raw):
    return raw


def get_postprocess_none():
    return nopp


def two_hump(start_weight, between_weight, end_weight, w, start, end):
    w[start] = start_weight
    w[start + 1:end] = between_weight
    w[end] = end_weight
    return w


def get_two_hump_weighting_function(start_weight, between_weight, end_weight):
    return partial(two_hump, start_weight, between_weight, end_weight)


def exponential_weighting(start_weight, end_weight, factor, w, start, end):
    length = end - start
    w[start:end] = 1 + np.exp(-np.linspace(start_weight, end_weight, length)) * factor
    return w


def get_exponential_weighting(start_weight=0, end_weight=3, factor=1):
    return partial(exponential_weighting, start_weight, end_weight, factor)


def phase_index(w, start, end):
    w[start:end] = np.arange(1, (end - start) + 1)
    return w


# this is only used for error analysis so far, to answer questions
# such as: in which phase of the note did we make the most errors?
# (does not make a lot of sense for anything else
def get_phase_indexing_function():
    return phase_index


def get_postprocess_smooth_weighting(kernel=np.array([1, 1, 1, 0, 0])):
    def smooth_weights(raw):
        w = np.zeros_like(raw)
        for note in xrange(w.shape[1]):
            w[:, note] = convolve(raw[:, note], kernel, mode='same')

        # moved weight normalizing to here, so each postprocessing function can opt out ...
        wmax = w.max()
        if wmax > 0:
            w /= wmax
        w += 1
        return w
    return smooth_weights


# this is the one that works very well
def midi_to_groundtruth_a(midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    y = np.zeros((n_frames, 88)).astype(np.int8)
    w = np.ones((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))
        label = pitch - 21
        y[frame_start:frame_end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], frame_start, frame_end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)
    return y, w


# this one uses ceil instead of round
def midi_to_groundtruth_b(midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    y = np.zeros((n_frames, 88)).astype(np.int8)
    w = np.ones((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.ceil(onset / dt))
        frame_end = int(np.ceil((onset + duration) / dt))
        label = pitch - 21
        y[frame_start:frame_end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], frame_start, frame_end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)
    return y, w


# this one uses floor instead of round
def midi_to_groundtruth_c(midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    y = np.zeros((n_frames, 88)).astype(np.int8)
    w = np.ones((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.floor(onset / dt))
        frame_end = int(np.floor((onset + duration) / dt))
        label = pitch - 21
        y[frame_start:frame_end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], frame_start, frame_end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)
    return y, w


# this is the one that produced the ISMIR2016 results (systematic error)
def midi_to_groundtruth_d(midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    notes = pattern.notes()

    piano_roll = np.zeros((n_frames, 109 - 21), dtype=np.int8)

    w = np.ones((n_frames, 88)).astype(np.float32)

    for n in notes:
        onset = int(np.ceil(n[0] / dt))
        end = onset + int(np.ceil(n[2] / dt))
        label = int(n[1] - 21)
        piano_roll[onset:end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], onset, end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)

    return piano_roll, w


# this is one that randomly shifts the whole note 1 frame
def midi_to_groundtruth_e(rng, midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    y = np.zeros((n_frames, 88)).astype(np.int8)
    w = np.ones((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))
        label = pitch - 21

        # randomly shift whole note
        shift = rng.randint(-1, 2)
        frame_start = max(0, frame_start + shift)
        frame_end = min(n_frames, frame_end + shift)

        y[frame_start:frame_end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], frame_start, frame_end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)
    return y, w


# this is one that randomly shifts the start and end
def midi_to_groundtruth_f(rng, midifile, dt, n_frames, weighting, postprocess_weighting):
    pattern = midi.MIDIFile.from_file(midifile)
    y = np.zeros((n_frames, 88)).astype(np.int8)
    w = np.ones((n_frames, 88)).astype(np.float32)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))
        label = pitch - 21

        # randomly shift start and end separately
        shift_start = rng.randint(-1, 2)
        frame_start = max(0, frame_start + shift_start)

        shift_end = rng.randint(-1, 2)
        frame_end = min(n_frames, frame_end + shift_end)

        y[frame_start:frame_end, label] = 1
        if weighting is not None:
            w[:, label] = weighting(w[:, label], frame_start, frame_end)

    if postprocess_weighting is not None:
        w = postprocess_weighting(w)
    return y, w


def spec_notes_weight_from_file(basedir, filename, _audio_options, midi_to_groundtruth, weighting, postprocess_weighting):
    audiofilename = os.path.join(basedir, filename + '.flac')
    midifilename = os.path.join(basedir, filename + '.mid')

    spec_type, audio_options = utils.canonicalize_audio_options(_audio_options, mmspec)

    # it's necessary to cast this to np.array, b/c the madmom-class holds references to wayyy too much memory ...
    x = np.array(spec_type(audiofilename, **audio_options))
    y, w = midi_to_groundtruth(midifilename, 1. / audio_options['fps'], x.shape[0], weighting, postprocess_weighting)
    return x, y, w


def random_log_filtered_mono_from_file(basedir, filename, audio_options, midi_to_groundtruth, samplesize, x_contextsize, y_contextsize, weighting, postprocess_weighting, rng, finite):
    x, y, w = spec_notes_weight_from_file(basedir, filename, audio_options, midi_to_groundtruth, weighting, postprocess_weighting)

    if x_contextsize > 0:
        x = utils.Context2D4D(x, x_contextsize)

    if y_contextsize > 0:
        y = utils.Context2D4D(y, y_contextsize)
        w = utils.Context2D4D(w, y_contextsize)

    indices = utils.indices_without_replacement(np.arange(0, len(x)), samplesize, rng, finite=finite)
    for idx in indices:
        yield x[idx], y[idx], w[idx], idx


def uniformly_random_sample(basedir, foldname, audio_options, midi_to_groundtruth, batchsize, x_contextsize, y_contextsize, weighting, postprocess_weighting, rng):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    specs = []
    for filename in filenames:
        specs.append(
            (
                filename,
                random_log_filtered_mono_from_file(
                    basedir,
                    filename,
                    audio_options,
                    midi_to_groundtruth,
                    1,
                    x_contextsize,
                    y_contextsize,
                    weighting,
                    postprocess_weighting,
                    rng=rng,
                    finite=False
                )
            )
        )

    while True:
        x_batch = []
        y_batch = []
        i_batch = []
        w_batch = []
        f_batch = []

        for bi in xrange(batchsize):
            # choose any file from specs
            fi = rng.randint(0, len(specs))
            filename, xywi_it = specs[fi]
            x, y, w, i = next(xywi_it)
            x_batch.append(x)
            y_batch.append(y)
            w_batch.append(w)
            i_batch.append(i)
            f_batch.append(filename)

        yield np.vstack(x_batch), np.vstack(y_batch), np.vstack(w_batch), np.hstack(i_batch), f_batch


def stratified_random_sample(basedir, foldname, audio_options, midi_to_groundtruth, batchsize, x_contextsize, y_contextsize, weighting, postprocess_weighting, rng):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    present_synthnames = set()
    for filename in filenames:
        synthname = filename[0:len(synthnames[0])]
        present_synthnames.add(synthname)

    # IMPORTANT:
    # we do away with this kind-of artificial restriction,
    # set n_samples_per_label to 1
    # and sample round-robin style from the synthnames

    # n_samples_per_label, rest = divmod(batchsize, len(present_synthnames))
    # if rest != 0:
    #     raise RuntimeError('batchsize not divisible by len(present_synthnames)={}'.format(len(present_synthnames)))

    synthname_to_specs = {synthname: list() for synthname in present_synthnames}
    for filename in filenames:
        synthname = filename[0:len(synthnames[0])]
        synthname_to_specs[synthname].append(
            (
                filename,
                random_log_filtered_mono_from_file(
                    basedir,
                    filename,
                    audio_options,
                    midi_to_groundtruth,
                    1,
                    x_contextsize,
                    y_contextsize,
                    weighting,
                    postprocess_weighting,
                    rng=rng,
                    finite=False
                )
            )
        )

    cycling_synthname_to_specs = {synthname: cycle(v) for synthname, v in synthname_to_specs.iteritems()}

    while True:
        x_batch = []
        y_batch = []
        i_batch = []
        w_batch = []
        f_batch = []

        # IMPORTANT:
        # this way, if the batchsize is *not* an integer-multiple of the number of len(present_synthnames),
        # we randomize, which synthesizer will be under/over represented in each batch
        random_synthnames = list(present_synthnames)
        rng.shuffle(random_synthnames)
        cycling_synthnames = cycle(random_synthnames)

        for bi in xrange(batchsize):
            synthname = next(cycling_synthnames)
            filename, xywi_it = next(cycling_synthname_to_specs[synthname])
            x, y, w, i = next(xywi_it)
            x_batch.append(x)
            y_batch.append(y)
            w_batch.append(w)
            i_batch.append(i)
            f_batch.append(filename)

        yield np.vstack(x_batch), np.vstack(y_batch), np.vstack(w_batch), np.hstack(i_batch), f_batch


def get_fold_iterator(
        basedir,
        foldname,
        audio_options,
        midi_to_groundtruth,
        batchsize,
        x_contextsize,
        y_contextsize,
        weighting=None,
        postprocess_weighting=None):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    # extract all the filenames from the foldfile
    labelled_specs = []
    for filename in filenames:
        x, y, w = spec_notes_weight_from_file(
            basedir,
            filename,
            audio_options,
            midi_to_groundtruth,
            weighting=weighting,
            postprocess_weighting=postprocess_weighting
        )

        if x_contextsize > 0:
            x = utils.Context2D4D(x, x_contextsize)

        if y_contextsize > 0:
            y = utils.Context2D4D(y, y_contextsize)
            w = utils.Context2D4D(w, y_contextsize)

        labelled_specs.append((filename, (x, y, w)))

    n_samples = 0
    for f, (x, y, w) in labelled_specs:
        n_samples += len(x)

    print 'fold_iterator (n_samples={})'.format(n_samples)

    def it():
        if weighting is None:
            for f, (x, y, w) in labelled_specs:
                n_batches = (len(x) // batchsize) + 1
                indices = cycle(xrange(0, len(x)))
                for i_batch in (np.array([indices.next() for _ in xrange(batchsize)]) for b in xrange(n_batches)):
                    yield x[i_batch], y[i_batch], i_batch, f
        else:
            # TODO: very hacky ...
            # the return values change, if weighting is used ...
            # this should *all* be dictionaries ...
            for f, (x, y, w) in labelled_specs:
                n_batches = (len(x) // batchsize) + 1
                indices = cycle(xrange(0, len(x)))
                for i_batch in (np.array([indices.next() for _ in xrange(batchsize)]) for b in xrange(n_batches)):
                    yield x[i_batch], y[i_batch], w[i_batch], i_batch, f

    # each time 'next' is called, yield a new generator that iterates over the (cached!) mem-mapped spectrograms
    while True:
        yield it()


def fully_stratified_random_sample(basedir, foldname, audio_options, midi_to_groundtruth, batchsize, x_contextsize, y_contextsize, weighting, postprocess_weighting, rng):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    pieces = []
    note_index = utils.ReverseNoteIndex()
    for piece_index, filename in enumerate(filenames):
        x, y, w = spec_notes_weight_from_file(basedir, filename, audio_options, midi_to_groundtruth, weighting, postprocess_weighting)

        note_index.update(piece_index, y.astype(np.int8))

        if x_contextsize > 0:
            x = utils.Context2D4D(x, x_contextsize)

        if y_contextsize > 0:
            y = utils.Context2D4D(y, y_contextsize)
            w = utils.Context2D4D(w, y_contextsize)

        pieces.append((x, y, w, filename))

    while True:
        x_batch = []
        y_batch = []
        i_batch = []
        w_batch = []
        f_batch = []

        for bi in xrange(batchsize):
            piece_index, time_index = note_index.sample()
            x, y, w, f = pieces[piece_index]
            x_batch.append(x[time_index])
            y_batch.append(y[time_index])
            w_batch.append(w[time_index])
            i_batch.append(time_index)
            f_batch.append(f)

        yield np.vstack(x_batch), np.vstack(y_batch), np.vstack(w_batch), np.hstack(i_batch), f_batch


def single_note_stratified_random_sample(basedir, foldname, audio_options, midi_to_groundtruth, batchsize, x_contextsize, y_contextsize, weighting, postprocess_weighting, rng):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    pieces = []
    note_index = utils.ReverseSingleNoteIndex()
    for piece_index, filename in enumerate(filenames):
        x, y, w = spec_notes_weight_from_file(basedir, filename, audio_options, midi_to_groundtruth, weighting, postprocess_weighting)

        note_index.update(piece_index, y.astype(np.int8))

        if x_contextsize > 0:
            x = utils.Context2D4D(x, x_contextsize)

        if y_contextsize > 0:
            y = utils.Context2D4D(y, y_contextsize)
            w = utils.Context2D4D(w, y_contextsize)

        pieces.append((x, y, w, filename))

    while True:
        x_batch = []
        y_batch = []
        i_batch = []
        w_batch = []
        f_batch = []

        for bi in xrange(batchsize):
            piece_index, time_index = note_index.sample()
            x, y, w, f = pieces[piece_index]
            x_batch.append(x[time_index])
            y_batch.append(y[time_index])
            w_batch.append(w[time_index])
            i_batch.append(time_index)
            f_batch.append(f)

        yield np.vstack(x_batch), np.vstack(y_batch), np.vstack(w_batch), np.hstack(i_batch), f_batch


def diverse_random_sample_for_single_note(basedir, foldname, audio_options, midi_to_groundtruth, batchsize, x_contextsize, y_contextsize, note, weighting, postprocess_weighting, rng):
    filenames = open(foldname, 'r').readlines()
    filenames = [f.strip() for f in filenames]

    pieces = []
    note_index = utils.ReverseSingleNoteIndex()
    for piece_index, filename in enumerate(filenames):
        x, y, w = spec_notes_weight_from_file(basedir, filename, audio_options, midi_to_groundtruth, weighting, postprocess_weighting)

        note_index.update(piece_index, y.astype(np.int8))

        if x_contextsize > 0:
            x = utils.Context2D4D(x, x_contextsize)

        if y_contextsize > 0:
            y = utils.Context2D4D(y, y_contextsize)
            w = utils.Context2D4D(w, y_contextsize)

        pieces.append((x, y, w, filename))

    while True:
        x_batch = []
        y_batch = []
        i_batch = []
        w_batch = []
        f_batch = []

        for bi in xrange(batchsize):
            # half of the time, we'll draw this note?
            piece_index, time_index = note_index.sample_single_note(note, ratio=0.5)
            x, y, w, f = pieces[piece_index]
            x_batch.append(x[time_index])
            y_batch.append(y[time_index])
            w_batch.append(w[time_index])
            i_batch.append(time_index)
            f_batch.append(f)

        yield np.vstack(x_batch), np.vstack(y_batch), np.vstack(w_batch), np.hstack(i_batch), f_batch
