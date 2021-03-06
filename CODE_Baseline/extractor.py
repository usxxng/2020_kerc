from keras.preprocessing import image

# from keras.applications.vgg16 import VGG16, preprocess_input
from keras_vggface.vggface import VGGFace
# import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras_vggface import utils
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
           
            # base_model = VGG16(weights='imagenet', include_top=True)
            base_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
            import pdb; pdb.set_trace()
            # We'll extract features at the final pool layer.
            
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            #self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        
     
        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # import pdb; pdb.set_trace()
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        x = utils.preprocess_input(x, version=1)
        # Get the prediction.
        features = self.model.predict(x)
        
        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
