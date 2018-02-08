from multiprocessing import Process
from collections import defaultdict
import pipedream.utils as utils
import numpy as np
import importlib
import argparse
import inspect
import shutil
import signal
import pprint
import tqdm
import os
import re


DEFAULT_RUN_DIR = 'runs'


def signal_handler(signum, frame):
    raise utils.ExternalInterrupt('received signal ({})'.format(signum))


def regex_in(key, regexes):
    for regex in regexes:
        match = re.match(regex, key)
        if match is not None:
            return True
    return False


def __one_train_run(args, configname, run_keys, force, hyper):
    if run_keys is not None and len(run_keys) > 0 and not regex_in(hyper['run_key'], run_keys):
        print 'train: skipping (run_key={}), not matched by any in ({})'.format(hyper['run_key'], run_keys)
        return
    else:
        if args.dry_run:
            print 'train: dry run (run_key={})'.format(hyper['run_key'])
            return
        else:
            print 'train: trying to run (run_key={})'.format(hyper['run_key'])

    run_path = os.path.join(args.run_dir, configname, hyper['run_key'])

    if os.path.exists(run_path):
        if force:
            print 'train: forced re-run of (run_key={})'.format(hyper['run_key'])
        else:
            print 'train: skipping (run_key={}), because run-directory exists'.format(hyper['run_key'])
            return
    else:
        os.makedirs(run_path)

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'train: unknown configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    # copy the source file used there
    config_src_path = os.path.join('..', 'pipedream', 'configurations', '{}.py'.format(configname))
    config_dst_path = os.path.join(run_path, 'config.py')
    shutil.copyfile(config_src_path, config_dst_path)

    # save away the hyper-parameters
    utils.dump(dict(hyper), os.path.join(run_path, 'hyper.pkl'))

    model = config.build_model(hyper)
    train_gen = config.get_train_iterator(hyper)
    train_it = train_gen()
    valid_gen = config.get_valid_iterator(hyper)

    print 'train: starting (run_key={})'.format(hyper['run_key'])
    state = dict(
        run_path=run_path,
        train_summary = [],
        valid_summary = [],
        status = dict(
            name='running'
        )
    )
    model.before_training(hyper, state)
    stopped = False
    try:
        for i_epoch in range(hyper['n_epoch']):
            print 'train: epoch {}/{}'.format(i_epoch, hyper['n_epoch'])
            for i_update in tqdm.tqdm(range(0, hyper['n_updates']), leave=True, mininterval=1.0, maxinterval=10.0, miniters=1):
                state.update(dict(
                    i_epoch=i_epoch,
                    i_update=i_update
                ))

                update_result = model.update(hyper, state, next(train_it))
                model.after_update(hyper, state, update_result)

            print 'train: validating'
            valid_accumulator = []
            valid_it = valid_gen()
            for valid_batch in tqdm.tqdm(valid_it, leave=True, mininterval=1.0, maxinterval=10.0, miniters=1):
                valid_accumulator.append(model.predict(valid_batch))
            del valid_it

            model.after_validation(hyper, state, valid_accumulator)

    except Exception as ex:
        stopped = True
        cause = utils.get_trace(ex)
        print 'train: stopping'
        print 'cause:', cause
        state['status'] = dict(
            name=str(type(ex).__name__),
            exception_backtrace = cause
        )

    if not stopped:
        state['status'] = dict(
            name='completed',
            exception_backtrace = None
        )

    del train_gen
    del train_it
    del valid_gen

    model.after_training(hyper, state)


def train(args):
    configname = args.configname
    run_keys = args.run_keys
    force = args.force

    signal.signal(signal.SIGINT, signal_handler)

    hyper_config = None
    try:
        hyper_config = importlib.import_module('pipedream.configurations.{}_hyper'.format(configname))
    except ImportError as ex:
        print 'train: unknown hyper parameter configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    # ensure configuration-directory exists
    runs_config_path = os.path.join(args.run_dir, configname)
    if not os.path.exists(runs_config_path):
        os.makedirs(runs_config_path)

    # run model with all hyper-parameters
    for hyper in hyper_config.get_hyperparameters():
        p = Process(target=__one_train_run, args=(args, configname, run_keys, force, hyper))
        p.daemon = True
        p.start()
        p.join()


def __advance_n_units(args, configname, hyper, n_units_of_work):
    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'advance-n: unknown configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    # build the model now
    model = config.build_model(hyper)

    run_path = os.path.join(args.run_dir, configname, hyper['run_key'])
    if not os.path.exists(run_path):
        raise RuntimeError('run_path ({}) does not exist ...'.format(run_path))

    print 'advance-n: continuing (run_key={})'.format(hyper['run_key'])
    # load the state of the optimization (model weights, updates and generator states(?))

    # (1) model weights (if they already exist)
    weights_path = os.path.join(run_path, 'current_weights.pkl')
    if os.path.exists(weights_path):
        model.load(weights_path)

    # (2) update state
    state = utils.load(os.path.join(run_path, 'state.pkl'))
    model.load_updates_from_state(state)

    # (3) generator state (TODO: this might prove to be extraordinarily difficult to get right ...)
    # ...

    # TODO: this needs to go, if we really restore generator states (3) !
    train_gen = config.get_train_iterator(hyper)
    train_it = train_gen()
    valid_gen = config.get_valid_iterator(hyper)

    stopped = False
    try:
        for i_unit in range(n_units_of_work):
            print 'advance-n: doing {}/{} unit(s) of work'.format(i_unit, n_units_of_work)

            # 1 unit of work == epoch ...
            state['i_epoch'] = state['i_epoch'] + 1

            for i_update in tqdm.tqdm(range(0, hyper['n_updates']), leave=True, mininterval=1.0, maxinterval=10.0, miniters=1):
                state['i_update'] = i_update
                update_result = model.update(hyper, state, next(train_it))
                if np.isinf(update_result['loss']):
                    print 'encountered Inf ({})'.format(update_result['loss'])
                    break
                elif np.isnan(update_result['loss']):
                    print 'encountered NaN ({})'.format(update_result['loss'])
                    break

                model.after_update(hyper, state, update_result)

            print 'advance-n: validating'
            valid_accumulator = []
            valid_it = valid_gen()
            for valid_batch in tqdm.tqdm(valid_it, leave=True, mininterval=1.0, maxinterval=10.0, miniters=1):
                valid_accumulator.append(model.predict(valid_batch))
            del valid_it

            model.after_validation(hyper, state, valid_accumulator)

    except Exception as ex:
        stopped = True
        cause = utils.get_trace(ex)
        print 'advance-n: stopping'
        print 'cause:', cause
        state['status'] = dict(
            name=str(type(ex).__name__),
            exception_backtrace=cause
        )

    if not stopped:
        state['status'] = dict(
            name='completed',
            exception_backtrace=None
        )

    del train_gen
    del train_it
    del valid_gen


# this does not actually *build* a model
def __initialize_run(configname, hyper):
    print 'advance: initializing (run_key={})'.format(hyper['run_key'])
    run_path = os.path.join(args.run_dir, configname, hyper['run_key'])

    os.makedirs(run_path)

    # copy the source file used there
    config_src_path = os.path.join('..', 'pipedream', 'configurations', '{}.py'.format(configname))
    config_dst_path = os.path.join(run_path, 'config.py')
    shutil.copyfile(config_src_path, config_dst_path)

    # save away the hyper-parameters
    utils.dump(dict(hyper), os.path.join(run_path, 'hyper.pkl'))

    # initialize state
    print 'advance: starting (run_key={})'.format(hyper['run_key'])
    state = dict(
        run_path=run_path,
        train_summary=[],
        valid_summary=[],
        i_epoch=0,
        status = dict(
            name='running'
        )
    )
    state = utils.dump(state, os.path.join(run_path, 'state.pkl'))


def advance(args):
    configname = args.configname
    run_keys = args.run_keys
    n_units_of_work = args.n_units_of_work

    if run_keys is None or len(run_keys) == 0:
        print 'advance: you need to specify run_keys!'
        exit(-1)

    signal.signal(signal.SIGINT, signal_handler)

    # ensure configuration-directory exists
    runs_config_path = os.path.join(args.run_dir, configname)
    if not os.path.exists(runs_config_path):
        print 'advance: no runs exist for ({})'.format(configname)
        exit(-1)

    # run model with specified run_keys
    for run_key in run_keys:
        run_path = os.path.join(args.run_dir, configname, run_key)
        if not os.path.exists(run_path):
            print 'advance: skipping, no run_path exists for ({})'.format(run_path)
            continue

        hyper_path = os.path.join(run_path, 'hyper.pkl')
        if not os.path.exists(hyper_path):
            print 'advance: skipping, no hyper_path exists for ({})'.format(hyper_path)
            continue

        hyper = utils.load(hyper_path)
        p = Process(target=__advance_n_units, args=(args, configname, hyper, n_units_of_work))
        p.daemon = True
        p.start()
        p.join()


def hyperband(args):
    configname = args.configname
    R = args.R
    eta = args.eta
    run_key_prefix = args.run_key_prefix

    signal.signal(signal.SIGINT, signal_handler)

    # load hyper parameter generator
    hyper_config = None
    try:
        hyper_config = importlib.import_module('pipedream.configurations.{}_hyper'.format(configname))
    except ImportError as ex:
        print 'train: unknown hyper parameter configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    # ensure configuration-directory exists
    runs_config_path = os.path.join(args.run_dir, configname)
    if not os.path.exists(runs_config_path):
        os.makedirs(runs_config_path)

    # log to the base of eta
    def logeta(x):
        return np.log(x) / np.log(eta)

    # number of unique executions of successive halving (minus one)
    s_max = int(logeta(R))

    # one execution of successive halving == one 'bracket'
    # total number of iterations (without reuse) per execution of successive halving (n,r)
    # this is the bracket budget
    B = (s_max + 1) * R

    print '### hyperband ######################################################'
    print 's_max {:>5} (number of successive halving calls)'.format(s_max)
    print 'B     {:>5} (bracket budget)'.format(B)
    print 'R     {:>5} (max amount of resource for one configuration)'.format(R)
    print 'eta   {:>5} (n_i/eta -> fraction of configurations kept when halving)'.format(eta)

    hypergen = hyper_config.get_hyperparameters()

    ##################################################################
    # begin finite horizon hyperband outerloop. (may be repeated indefinitely)
    for s in reversed(range(s_max + 1)):

        # initial number of configurations
        n = int(np.ceil(B / R / (s + 1) * eta**s))

        # initial number of iterations to run configurations for
        r = R * eta**(-s)

        print '#' * 60
        print 'suha(s={}, n={}, r={})'.format(s, n, r)

        ###########################################################
        # begin finite horizon successive halving with (n,r)
        T = [next(hypergen) for _ in range(n)]

        # prepend the run_key_prefix to the random run_keys in T
        # this is so we do not have collisions, when running on multiple
        # gpu's (there is a very small chance of that happening (1./ 2**63),
        # so this is just for exactness
        for t in T:
            t['run_key'] = run_key_prefix + '_' + t['run_key']

        # initialize hyper parameters
        for t in T:
            __initialize_run(configname, t)

        for i in range(s + 1):
            # run each of the configs for r_i iterations and keep best n_i/eta
            r_i = int(r * eta**(i))
            print 'exec(r_i={}, t) (|T|={}, T={})'.format(r_i, len(T), [t['run_key'] for t in T])

            # L = [run_then_return_validation_loss(r_i, t) for t in T]
            L = []
            for t in T:
                p = Process(target=__advance_n_units, args=(args, configname, t, r_i))
                p.daemon = True
                p.start()
                p.join()

                # now we need to read out the last validation loss ...
                run_path = os.path.join(args.run_dir, configname, t['run_key'])
                state_path = os.path.join(run_path, 'state.pkl')
                state = utils.load(state_path)

                # get the last entry ...
                valid_loss = float('Inf')
                if 'valid_summary' in state:
                    if len(state['valid_summary']) > 0:
                        valid_loss = state['valid_summary'][-1]['valid_loss']

                L.append(valid_loss)

            n_i = int(n * eta**(-i))
            n_keep = int(n_i / eta)
            T = [T[j] for j in np.argsort(L)[0:n_keep]]

        # end finite horizon successive halving with (n,r)
        ###########################################################


def test(args):
    configname = args.configname
    selected_run_keys = args.run_keys

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'test: unknown configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'test: no runs for configuration'
        exit(-1)

    _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
    potential_run_keys = sorted(dirs)

    if len(selected_run_keys) > 0:
        run_keys = []
        for potential_run_key in potential_run_keys:
            if regex_in(potential_run_key, selected_run_keys):
                run_keys.append(potential_run_key)
    else:
        run_keys = potential_run_keys

    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            model = config.build_model(hyper)
            model.load(os.path.join(run_path, 'weights.pkl'))
            state = utils.load(os.path.join(run_path, 'state.pkl'))
            test_gen = config.get_test_iterator(hyper)

            print 'test: starting (run_key={})'.format(run_key)
            model.before_testing(hyper, state)

            test_summary = []
            for test_batch in tqdm.tqdm(test_gen(), mininterval=1.0, maxinterval=10.0, miniters=1):
                test_result = model.predict(test_batch)
                test_summary.append(test_result)

            model.after_testing(hyper, state, test_summary)

        except Exception as skip_reason:
            print 'test: skipping due to exception'
            print utils.get_trace(skip_reason)


def postprocess(args):
    configname = args.configname
    run_keys = args.run_keys

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'postprocess: unknown configuration ({})'.format(configname)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'postprocess: no runs for configuration'
        exit(-1)

    if len(run_keys) == 0:
        _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
        run_keys = sorted(dirs)

    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            model = config.build_model(hyper)
            model.load(os.path.join(run_path, 'weights.pkl'))
            state = utils.load(os.path.join(run_path, 'state.pkl'))

            pp = getattr(model, 'postprocess')
            signature = inspect.getargspec(pp)
            if signature.args == ['self', 'hyper', 'state']:
                print 'postprocess: starting (run_key={})'.format(run_key)
                model.before_postprocess(hyper, state)
                model.postprocess(hyper, state)
                model.after_postprocess(hyper, state)
            else:
                train_gen = config.get_train_iterator(hyper)
                valid_gen = config.get_valid_iterator(hyper)
                test_gen = config.get_test_iterator(hyper)

                print 'postprocess: starting (run_key={})'.format(run_key)
                model.before_postprocess(hyper, state)
                model.postprocess(hyper, state, train_gen, valid_gen, test_gen)
                model.after_postprocess(hyper, state)

        except Exception as skip_reason:
            print 'postprocess: skipping due to exception'
            print utils.get_trace(skip_reason)


def visualize(args):
    configname = args.configname
    run_keys = args.run_keys

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'visualize: unknown configuration ({})'.format(configname)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'visualize: no runs for configuration'
        exit(-1)

    if len(run_keys) == 0:
        _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
        run_keys = sorted(dirs)

    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            model = config.build_model(hyper)
            model.load(os.path.join(run_path, 'weights.pkl'))
            state = utils.load(os.path.join(run_path, 'state.pkl'))

            train_it = config.get_train_iterator(hyper)()
            valid_it = config.get_valid_iterator(hyper)()
            test_it = config.get_test_iterator(hyper)()

            print 'visualize: starting (run_key={})'.format(run_key)
            model.visualize(hyper, state, train_it, valid_it, test_it)

        except Exception as skip_reason:
            print 'visualize: skipping due to exception'
            print utils.get_trace(skip_reason)


def showstate(args):
    configname = args.configname
    run_keys = args.run_keys

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'showstate: unknown configuration ({})'.format(configname)
        print utils.get_trace(ex)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'showstate: no runs for configuration'
        exit(-1)

    if len(run_keys) == 0:
        _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
        run_keys = sorted(dirs)

    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            state = utils.load(os.path.join(run_path, 'state.pkl'))
            print 'showstate: starting (run_key={})'.format(run_key)
            config.showstate(hyper, state)

        except Exception as skip_reason:
            print 'showstate: skipping due to exception'
            print utils.get_trace(skip_reason)


def summarize(args):
    configname = args.configname
    selected_run_keys = args.run_keys

    config = None
    try:
        config = importlib.import_module('pipedream.configurations.{}'.format(configname))
    except ImportError as ex:
        print 'summarize: unknown configuration ({})'.format(configname)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'summarize: no runs for configuration'
        exit(-1)

    _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
    potential_run_keys = sorted(dirs)

    if len(selected_run_keys) > 0:
        run_keys = []
        for potential_run_key in potential_run_keys:
            if regex_in(potential_run_key, selected_run_keys):
                run_keys.append(potential_run_key)
    else:
        run_keys = potential_run_keys

    print 'summarize: collecting data'
    hypers = defaultdict(list)
    states = defaultdict(list)
    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            state = utils.load(os.path.join(run_path, 'state.pkl'))
            print 'summarize: collecting (run_key={})'.format(run_key)
            hypers[run_key] = hyper
            states[run_key] = state

        except Exception as skip_reason:
            print 'summarize: skipping due to exception'
            print utils.get_trace(skip_reason)

    config.summarize(hypers, states)


def showhyper(args):
    configname = args.configname
    run_keys = args.run_keys

    config = None
    if not os.path.exists('../pipedream/configurations/{}.py'.format(configname)):
        print 'showhyper: unknown configuration ({})'.format(configname)
        exit(-1)

    if not os.path.exists(os.path.join(args.run_dir, configname)):
        print 'showhyper: no runs for configuration'
        exit(-1)

    if len(run_keys) == 0:
        _, dirs, _ = next(os.walk(os.path.join(args.run_dir, configname)))
        run_keys = sorted(dirs)

    print 'showhyper: collecting data'
    hypers = defaultdict(list)
    states = defaultdict(list)
    for run_key in run_keys:
        try:
            run_path = os.path.join(args.run_dir, configname, str(run_key))
            hyper = utils.load(os.path.join(run_path, 'hyper.pkl'))
            print 'showhyper: collecting (run_key={})'.format(run_key)
            hypers[run_key] = hyper

        except Exception as skip_reason:
            print 'showhyper: skipping due to exception'
            print utils.get_trace(skip_reason)

    for run_key, hyper in hypers.iteritems():
        pprint.pprint(hyper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()

    train_parser = sp.add_parser('train')
    train_parser.add_argument('configname', type=str)
    train_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    train_parser.add_argument('--run_keys', type=str, nargs='+', default=None)
    train_parser.add_argument('--force', action='store_true', default=False)
    train_parser.add_argument('--dry_run', action='store_true', default=False)
    train_parser.set_defaults(func=train)

    advance_parser = sp.add_parser('advance')
    advance_parser.add_argument('configname', type=str)
    advance_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    advance_parser.add_argument('--run_keys', type=str, nargs='+', default=None)
    advance_parser.add_argument('--n_units_of_work', type=int, default=1)
    advance_parser.set_defaults(func=advance)

    hyperband_parser = sp.add_parser('hyperband')
    hyperband_parser.add_argument('configname', type=str)
    hyperband_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    hyperband_parser.add_argument('--R', type=int, default=81)
    hyperband_parser.add_argument('--eta', type=int, default=3)
    hyperband_parser.add_argument('--run_key_prefix', type=str, default="")
    hyperband_parser.set_defaults(func=hyperband)

    test_parser = sp.add_parser('test')
    test_parser.add_argument('configname', type=str)
    test_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    test_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    test_parser.set_defaults(func=test)

    postprocess_parser = sp.add_parser('postprocess')
    postprocess_parser.add_argument('configname', type=str)
    postprocess_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    postprocess_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    postprocess_parser.set_defaults(func=postprocess)

    summarize_parser = sp.add_parser('summarize')
    summarize_parser.add_argument('configname', type=str)
    summarize_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    summarize_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    summarize_parser.set_defaults(func=summarize)

    visualize_parser = sp.add_parser('visualize')
    visualize_parser.add_argument('configname', type=str)
    visualize_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    visualize_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    visualize_parser.set_defaults(func=visualize)

    showstate_parser = sp.add_parser('showstate')
    showstate_parser.add_argument('configname', type=str)
    showstate_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    showstate_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    showstate_parser.set_defaults(func=showstate)

    showhyper_parser = sp.add_parser('showhyper')
    showhyper_parser.add_argument('configname', type=str)
    showhyper_parser.add_argument('--run_dir', type=str, default=DEFAULT_RUN_DIR)
    showhyper_parser.add_argument('--run_keys', type=str, nargs='+', default=[])
    showhyper_parser.set_defaults(func=showhyper)

    args = parser.parse_args()
    args.func(args)
