import lasagne.layers as L
import lasagne.init as I
import lasagne.nonlinearities as A
import theano
import theano.tensor as T
import numpy as np
import operator
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict


# snatched from jan, https://github.com/f0k/Lasagne/blob/807763f64bbdcb8dfbe3ef415c030495ff34d4e8/lasagne/nonlinearities.py
class Softmax(object):
    """
    Softmax activation function across given axes
    Parameters
    ----------
    axes : int, tuple of int or 'locations'
        The axes to apply the softmax over, such that the outputs over these
        axes can be interpreted as a probability distribution across classes.
        If `'locations'`, uses all axes except the first two.
    Methods
    -------
    __call__(x)
        Apply the softmax function to the activation `x`.
    Examples
    --------
    In contrast to most other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:
    >>> from lasagne.layers import InputLayer, Conv2DLayer
    >>> l_in = InputLayer((None, 1, 30, 50))
    >>> from lasagne.nonlinearities import Softmax
    >>> spatial_softmax = Softmax(axes=(2, 3))
    >>> l1 = DenseLayer(l_in, num_units=5, nonlinearity=spatial_softmax)
    See also
    --------
    softmax: softmax across the second axis of a 2D tensor
    softmax_per_location: Instance with ``axes=1`` (across channels)
    softmax_over_locations: Instance with ``axes='locations'`` (per channel)
    spatial_softmax: Alias for :func:`softmax_per_location`
    """

    def __init__(self, axes):
        if isinstance(axes, int):
            axes = (axes,)
        elif not isinstance(axes, tuple) and axes != 'locations':
            raise ValueError("axes must be int, tuple of int, or 'locations',"
                             " got %r instead" % axes)
        self.axes = axes

    def __call__(self, x):
        # determine axes to softmax over
        if self.axes == 'locations':
            axes = tuple(range(2, x.ndim))
        else:
            axes = self.axes
        if len(axes) == 0:
            raise ValueError("there are no axes for the softmax with "
                             "axes=%r and input dimensionality %d" %
                             (axes, x.ndim))

        # dimshuffle softmax axes to the end, if needed
        other_axes = tuple(ax for ax in range(x.ndim) if ax not in axes)
        pattern = other_axes + axes
        if pattern != tuple(range(x.ndim)):
            x = x.dimshuffle(pattern)

        # flatten or expand to two dimensions, if needed
        if not len(other_axes) == len(axes) == 1:
            shape = x.shape  # for later restoration
            if len(axes) == 1:
                # JAN: small bug fix
                x = x.reshape((-1, x.shape[x.ndim - 1]))
            else:
                x = x.reshape((reduce(operator.mul, x.shape[:len(other_axes)]), -1))

        # apply softmax
        x = theano.tensor.nnet.softmax(x)

        # unflatten, if needed
        if not len(other_axes) == len(axes) == 1:
            x = x.reshape(shape)

        # restore axis order, if needed
        if pattern != tuple(range(x.ndim)):
            anti_pattern = tuple(np.argsort(pattern))
            x = x.dimshuffle(anti_pattern)

        return x


softmax_per_location = Softmax(axes=1)  # shortcut for softmax across channels


# snatched from https://github.com/Lasagne/Lasagne/pull/843/files?diff=unified, modified names
class SELU(object):
    """Scaled Exponential Linear Unit :math:`\\varphi(x) = \\lambda (x > 0 ? x : \\alpha(e^x-1)`

    The Scaled Exponential Linear Unit (SELU) was introduced  in [1]
    as an activation function that allows the construction of
    self-normalizing neural networks.

    Parameters
    ----------
    scale : float32
        The scale parameter :math:`\\lambda` for scaling all output.

    scale_neg  : float32
        The scale parameter :math:`\\alpha` for scaling output for negative argument values.

    Methods
    -------
    __call__(x)
        Apply the SELU function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import SELU
    >>> selu = SELU(2, 3)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=selu)

    References
    ----------
    .. [1] Klambauer, Guenter, et al. (2017):
       Self-Normalizing Neural Networks,
       https://arxiv.org/abs/1706.02515
    """
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946

    def __init__(self, alpha=ALPHA, scale=SCALE):
        self.alpha = alpha
        self.scale = scale

    def __call__(self, x):
        return self.scale * theano.tensor.switch(
            x >= 0.0,
            x,
            self.alpha * (theano.tensor.exp(x) - 1)
        )
        # return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def additional_info(layer):
    if isinstance(layer, L.GaussianNoiseLayer):
        return 'GaussianNoiseLayer({})'.format(layer.sigma)
    if isinstance(layer, L.DropoutLayer):
        return 'DropoutLayer({})'.format(layer.p)
    if isinstance(layer, L.BatchNormLayer):
        return 'BatchNormLayer(alpha={})'.format(layer.alpha)
    else:
        return layer.__class__.__name__


def to_string(network):
    """
    author: filip/rainer made it less specific ...
    """
    repr_str = ''
    for layer in L.get_all_layers(network):
        n_outputs = int(np.prod(layer.output_shape[1:]))
        n_params = int(np.sum([np.prod(p.get_value().shape) for p in layer.get_params()]))
        repr_str += '\t{:<20} - #a {:>8} - #p {:>10} - {:<15} - {:<25}\n'.format(layer.output_shape, n_outputs, n_params, layer.name, additional_info(layer))

    return repr_str


def get_n_params(network):
    s = 0
    for p in L.get_all_params(network):
        s += np.prod(p.get_value().shape)
    return s


class MaskedInitializer(object):
    def __init__(self, wrapped, mask):
        self.wrapped = wrapped
        self.mask = mask

    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        return self.wrapped(shape) * self.mask


class MaskedLayer(L.Layer):
    def __init__(self, incoming, num_units, W_mask, W=I.GlorotUniform(), b=I.Constant(0.), nonlinearity=A.rectify, **kwargs):
        super(MaskedLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (A.identity if nonlinearity is None else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W_mask = W_mask
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W * self.W_mask)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class MaskedRecurrentLayer(L.CustomRecurrentLayer):
    def __init__(self, incoming, num_units,
                 W_in_to_hid_mask,
                 W_hid_to_hid_mask,
                 W_in_to_hid=I.Uniform(),
                 W_hid_to_hid=I.Uniform(),
                 b=I.Constant(0.),
                 nonlinearity=A.rectify,
                 hid_init=I.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        # EDIT: we pass a masked dense layer here ...
        in_to_hid = MaskedLayer(L.InputLayer((None,) + input_shape[2:]),
                                num_units, W_mask=W_in_to_hid_mask, W=W_in_to_hid, b=b,
                                nonlinearity=None,
                                name=basename + 'input_to_hidden',
                                **layer_kwargs)

        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        # EDIT: we pass a masked dense layer here ...
        hid_to_hid = MaskedLayer(L.InputLayer((None, num_units)),
                                 num_units, W_mask=W_hid_to_hid_mask, W=W_hid_to_hid, b=None,
                                 nonlinearity=None,
                                 name=basename + 'hidden_to_hidden',
                                 **layer_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(MaskedRecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, **kwargs)


def geometric(b, rho):
    return rho ** (b - 1) * (1 - rho)


def geometric_cdf(b, rho):
    return 1 - rho ** b


def geometric_inverse_cdf(b, rho):
    return T.log(1 - b) / T.log(rho)


def random_geometric(size, rho, srng, dtype):
    samples = srng.uniform(size=size, low=0.0, high=1.0)
    return T.cast(T.ceil(geometric_inverse_cdf(samples, rho)), 'int64')


class NestedDropoutLayer(L.Layer):
    """Nested Dropout layer

    Sets values above a certain index 'b' to zero. The index 'b' is distributed according to
    p_B(b, rho) = rho^{(b - 1)} (1 - rho)

    See notes for disabling dropout during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    rho : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.
    """
    def __init__(self, incoming, rho=0.9, rescale=True, **kwargs):
        super(NestedDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.rho = rho
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.rho == 0:
            return input
        else:
            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            # draw as many indices as 'batchsize' from a geometric distribution [input_shape = (batchsize, dim1, dim2, ...)]
            batchsize, dimension = input_shape

            # TODO: think about this some more. for 'normal' dropout, this is rescaled with [input /= (1 - p)]
            # so for the geometric distribution, this should be the cumulative probability for an index actually
            if self.rescale:
                # compute rescaling_vector based on the geometric pdf
                ones = T.ones(dimension, dtype='float32')
                rescaling_vector = geometric(T.arange(dimension, dtype='float32'), self.rho)
                input /= ones - rescaling_vector

            # indices = T.clip(random_geometric((batchsize, ), rho=self.rho, srng=self._srng, dtype=input.dtype), 0, dimension - 1)
            indices = T.mod(random_geometric((batchsize, ), rho=self.rho, srng=self._srng, dtype=input.dtype), dimension)
            mask = T.tri(dimension, k=0, dtype=input.dtype)[indices]
            return input * mask


# "All you need is a good init", Mishkin, Matas, ICLR16
def get_outputs(network, output_transform, output_transform_kwargs, types=[L.Conv2DLayer, L.Conv1DLayer, L.DilatedConv2DLayer, L.DenseLayer]):
    outputs = OrderedDict()

    def is_included_type(layer):
        for t in types:
            if isinstance(layer, t):
                return True
        return False

    input_layer = None
    layers = OrderedDict()
    for layer in L.get_all_layers(network):
        if isinstance(layer, L.InputLayer):
            input_layer = layer

        if is_included_type(layer):
            original_nonlinearity = getattr(layer, 'nonlinearity', None)
            if original_nonlinearity is not None:
                # temporarily replace the nonlinearity
                layer.nonlinearity = A.identity

            # get output with nonlin == 'identity', switch off dropout / batchnorm inbetween!
            outputs[layer.name] = output_transform(L.get_output(layer, **output_transform_kwargs))

            # restore original nonlinearity
            layer.nonlinearity = original_nonlinearity

            layers[layer.name] = layer
        else:
            print 'skipping layer ({}, {})'.format(layer.__class__.__name__, layer.name)
    return outputs, layers, input_layer


def get_ayniagi(network):
    return get_outputs(network, T.var, dict(deterministic=True))


def ayniagi_burn_in(network, data):
    ayniagi, layers, net_input = get_ayniagi(network)
    f_ayniagi = theano.function(
        inputs=[net_input.input_var],
        outputs=ayniagi.values()  # these are the expressions for variance
    )

    steps = 20
    tolerance = 0.01
    index = dict([(name, i) for name, i in zip(layers.keys(), range(len(layers.keys())))])

    for name in layers.keys():
        print 'aynagi for layer ({})'.format(name)

        i = 0
        variance = 0
        for param in layers[name].get_params():
            print 'before, param {} mean {}, var {}'.format(param.name, np.mean(param.get_value()), np.std(param.get_value()))

        while i < steps and abs(variance - 1) > tolerance:
            batch = next(data)
            variances = f_ayniagi(batch)
            variance = variances[index[name]]
            for param in layers[name].get_params():
                param.set_value(param.get_value() / np.sqrt(variance))
            i += 1

        for param in layers[name].get_params():
            print 'after ', param.name, np.mean(param.get_value()), np.std(param.get_value())

            for t in range(10):
                batch = next(data)
                variances = f_ayniagi(batch)
                print 'after', name, variances[index[name]]


# "All you need is a good init", Mishkin, Matas, ICLR16
# where each feature-map is variance normalized separately
def get_ayniagi_conv(network):
    outputs = OrderedDict()

    def is_excluded_type(layer):
        types = [L.InputLayer]
        for t in types:
            if isinstance(layer, t):
                return True
        return False

    input_layer = None
    layers = OrderedDict()
    for layer in L.get_all_layers(network):
        if isinstance(layer, L.InputLayer):
            input_layer = layer

        if not is_excluded_type(layer):
            original_nonlinearity = getattr(layer, 'nonlinearity', None)
            if original_nonlinearity is not None:
                # temporarily replace the nonlinearity
                layer.nonlinearity = A.identity

            # get output with nonlin == 'identity'
            if len(layer.output_shape) == 2:
                outputs[layer.name] = T.var(L.get_output(layer))
            elif len(layer.output_shape) == 4:
                # obtain variance per 'feature' per feature-map, over all batches
                outputs[layer.name] = T.var(L.get_output(layer), axis=(0, 2, 3))
            else:
                raise RuntimeError('weird shape! {} "{}"'.format(layer.name, layer.output_shape))

            # restore original nonlinearity
            layer.nonlinearity = original_nonlinearity

            layers[layer.name] = layer
    return outputs, layers, input_layer


def ayniagi_burn_in_conv(network, data):
    ayniagi, layers, net_input = get_ayniagi_conv(network)
    f_ayniagi = theano.function(
        inputs=[net_input.input_var],
        outputs=ayniagi.values()  # these are the expressions for variance
    )

    steps = 20
    tolerance = 0.01
    index = dict([(name, i) for name, i in zip(layers.keys(), range(len(layers.keys())))])

    for name in layers.keys():
        print 'aynagi for layer ({})'.format(name)

        i = 0
        variance = 0
        for param in layers[name].get_params():
            print 'before', param.name, np.mean(param.get_value()), np.std(param.get_value())

        while i < steps and abs(variance - 1) > tolerance:
            batch = next(data)
            variances = f_ayniagi(batch)
            for param in layers[name].get_params():
                value = param.get_value()

                if len(value.shape) == 4:
                    param.set_value(value / np.sqrt(variances[index[name]])[:, None, None, None])
                else:
                    param.set_value(value / np.sqrt(variances[index[name]]))

            i += 1

        for param in layers[name].get_params():
            print 'after ', param.name, np.mean(param.get_value()), np.std(param.get_value())

            for t in range(3):
                batch = next(data)
                variances = f_ayniagi(batch)
                print 'after', name, variances[index[name]]


# class InstanceScalingLayer(L.Layer):
#     def __init__(self, incoming, alpha=1e-9, **kwargs):
#         super(InstanceScalingLayer, self).__init__(incoming, **kwargs)
#         self.alpha = alpha
#         self.k = k
#         self.beta = beta
#         self.n = n
#         if n % 2 == 0:
#             raise NotImplementedError("Only works with odd n")

#     def get_output_shape_for(self, input_shape):
#         return input_shape

#     def get_output_for(self, input, **kwargs):
#         input_shape = self.input_shape
#         if any(s is None for s in input_shape):
#             input_shape = input.shape
#         half_n = self.n // 2
#         input_sqr = T.sqr(input)
#         b, ch, r, c = input_shape
#         extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
#         input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
#                                     input_sqr)
#         scale = self.k
#         for i in range(self.n):
#             scale += self.alpha * input_sqr[:, i:i+ch, :, :]
#         scale = scale ** self.beta
#         return input / scale
