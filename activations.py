import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

        
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'
        

get_custom_objects().update({'swish': swish})
get_custom_objects().update({'mish': mish})
# get_custom_objects().update({'mish': Mish(mish)})
# get_custom_objects().update({'swish': Swish(swish)})