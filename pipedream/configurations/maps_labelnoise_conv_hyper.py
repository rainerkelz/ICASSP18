def get_hyperparameters():
    bs = 8
    stratify = False
    splits = [
        'sigtia-4-splits/fold_1',
        'sigtia-4-splits/fold_2',
        'sigtia-4-splits/fold_3',
        'sigtia-4-splits/fold_4',
        'sigtia-conf2-splits/fold_1',
        'sigtia-conf2-splits/fold_2',
        'sigtia-conf2-splits/fold_3',
        'sigtia-conf2-splits/fold_4',
    ]

    sample_rate = 44100

    for split in splits:
        for fps, contextsize in [(100, 17), (31.25, 5)]:
            for labelfunction in ['a', 'b', 'c', 'd', 'e', 'f']:
                config = {'batchsize': bs,
                          'input_shape': (bs, 1, contextsize, 229),
                          'learning_rate': 0.1,
                          'log_frequency': 100,
                          'n_epoch': 50,
                          'n_updates': 40000,
                          'run_key': 'fold_{}_fps_{}_cs_{}_lf_{}_st_{}'.format(
                              split.replace('/', '_'),
                              fps,
                              contextsize,
                              labelfunction,
                              stratify
                          ),
                          'stepwise_lr': {'factor': 0.5, 'n_epoch': 10},

                          'audio_path': 'data/maps_piano/data',
                          'train_fold': 'data/maps_piano/splits/{}/train'.format(split),
                          'valid_fold': 'data/maps_piano/splits/{}/valid'.format(split),
                          'test_fold': 'data/maps_piano/splits/{}/test'.format(split),

                          'stratify': stratify,
                          'labelfunction': labelfunction,
                          'audio_options': {
                              'num_channels': 1,
                              'sample_rate': sample_rate,
                              'filterbank': 'LogarithmicFilterbank',
                              'frame_size': 4096,
                              'fft_size': 4096,
                              'fps': fps,
                              'num_bands': 48,
                              'fmin': 30,
                              'fmax': 8000.0,
                              'fref': 440.0,
                              'norm_filters': True,
                              'unique_filters': True,
                              'circular_shift': False,
                              'norm': True
                          },

                          'weight_function': {'name': 'get_no_weighting', 'params': {}},
                          'postprocess_function': {'name': 'get_postprocess_none', 'params': {}}}
                yield config
