import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.models import Model

from config import image_size, num_classes, num_box, num_grid


def build_model():
    base_model = VGG16(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet', pooling=None)
    input_image = base_model.input
    x = base_model.layers[-1].output
    x = Conv2D(num_box * (4 + 1 + num_classes), (1, 1), strides=(1, 1), padding='same')(x)
    output = Reshape((num_grid, num_grid, 4 + 1 + num_classes))(x)
    model = Model(input_image, output)
    return model


if __name__ == '__main__':
    m = build_model()
    print(m.summary())
    from keras.utils import plot_model

    plot_model(m, to_file='model.svg', show_layer_names=True, show_shapes=True)
    K.clear_session()
