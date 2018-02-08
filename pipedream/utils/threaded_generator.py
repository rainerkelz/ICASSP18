# TG from jan
# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
from threading import Thread, Event, Lock, RLock

try:
    from queue import Queue, Full

except ImportError:
    from Queue import Queue, Full

import time


class LockProxy(object):
    def __init__(self, obj):
        self.__obj = obj
        self.__lock = RLock()

    # RLock because object methods may call own methods
    def __getattr__(self, name):
        def wrapped(*a, **k):
            result = None
            with self.__lock:
                result = getattr(self.__obj, name)(*a, **k)
            return result

        return wrapped

    def __getitem__(self, key):
        result = None
        with self.__lock:
            result = self.__obj[key]
        return result


class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self,
                 iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 daemon=True,
                 Thread=Thread,
                 Queue=Queue):
        self.iterator = iterator
        self.sentinel = sentinel
        self.queue = Queue(maxsize=queue_maxsize)
        self.thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self.thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self.iterator)

    def _run(self):
        try:
            for value in self.iterator:
                self.queue.put(value)

        finally:
            self.queue.put(self.sentinel)

    def __iter__(self):
        self.thread.start()
        for value in iter(self.queue.get, self.sentinel):
            yield value

        self.thread.join()


class StoppableThread(Thread):
    """Threaded iterator feeding a queue, with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, iterator, queue, sentinel, debug):
        super(StoppableThread, self).__init__()
        self._stop = Event()
        self.iterator = iterator
        self.queue = queue
        self.sentinel = sentinel
        self.daemon = True
        self.debug = debug

    def __dbg(self, msg):
        if self.debug:
            print '{}({}): {}'.format(self.__class__.__name__, self.ident, msg)

    def run(self):
        try:
            it = iter(self.iterator)
            i = 0
            while not self.stopped():
                self.__dbg('getting next item {} from iterator'.format(i))
                item = next(it)
                trying = True
                while not self.stopped() and trying:
                    try:
                        self.__dbg('trying to put next item {} into queue'.format(i))
                        # self.queue.put(item, block=True, timeout=0.01)
                        self.queue.put(item, block=False)
                        trying = False
                    except Full as qf:
                        self.__dbg('full queue {}'.format(i))
                        time.sleep(0.1)

                i += 1
        except Exception as e:
            import traceback
            import sys

            self.__dbg('exception in run() exception was "{}"'.format(e))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type,
                exc_value,
                exc_traceback,
                limit=20,
                file=sys.stdout
            )

        finally:
            if self.stopped():
                self.__dbg('thread stopped, no sentinel')
            else:
                self.__dbg('trying to put sentinel')
                self.queue.put(self.sentinel)

    def stop(self):
        self.__dbg('stop()')
        self._stop.set()

    def stopped(self):
        self.__dbg('stopped({})'.format(self._stop.isSet()))
        return self._stop.isSet()


class MultiThreadedGenerators(object):
    """
    Multiple generators that run on separate threads, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self,
                 iterators,
                 sentinel=object(),
                 queue_maxsize=128,
                 debug=False):
        self.iterators = iterators
        self.sentinel = sentinel
        self.queue = Queue(maxsize=queue_maxsize)
        self.threads = []
        self.debug = debug

        # for checking if queue is actually full most of the time
        # self.qsizes = []

        self.__dbg('passed {} iterators'.format(len(iterators)))

        for iterator in self.iterators:
            self.threads.append(StoppableThread(iterator, self.queue, self.sentinel, self.debug))

        self.__dbg('created {} threads'.format(len(self.threads)))
        self.__dbg('starting {} threads'.format(len(self.threads)))

        for thread in self.threads:
            thread.start()
            self.__dbg('starting thread {}'.format(thread.ident))

    def __dbg(self, msg):
        if self.debug:
            print '{}: {}'.format(self.__class__.__name__, msg)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.iterators)

    def __iter__(self):

        for value in iter(self.queue.get, self.sentinel):
            yield value
            self.queue.task_done()
            # for checking if queue is actually full most of the time
            # self.qsizes.append((time.clock(), self.queue.qsize()))

        for thread in self.threads:
            self.__dbg('joining thread {}'.format(thread.ident))
            thread.join()
            self.__dbg('joined thread {}'.format(thread.ident))

    def __del__(self):
        self.__dbg('del self')
        self.stop()

    def stop(self):
        self.__dbg('stopping threads')
        for thread in self.threads:
            self.__dbg('stopping thread')
            thread.stop()
            self.__dbg('joining thread')
            thread.join()


class Demultiplexer(object):
    def __init__(self, generators):
        self.generators = generators

    def next(self):
        cursor = 0
        total = len(self.generators)
        while True:
            yield next(self.generators[cursor])
            cursor = (cursor + 1) % total


class RandomSelector(object):
    def __init__(self, rng, generators):
        self.rng = rng
        self.generators = generators
        self.n_generators = len(self.generators)

    def next(self):
        while True:
            yield next(self.generators[self.rng.randint(0, self.n_generators)])


# from http://anandology.com/blog/using-iterators-and-generators/
class ThreadsafeIterator(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


# from http://anandology.com/blog/using-iterators-and-generators/
def ThreadsafeGenerator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadsafeIterator(f(*a, **kw))
    return g
