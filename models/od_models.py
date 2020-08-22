import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class Network:
    def __init__(self, core_net="vgg19", head_net="yolov4"):
        """
            Network design here
        """
        self.head_net = head_net.lower()
        self.corenet_list = ["vgg19", "mobilenetv2", "resnet50"]
        if core_net not in self.corenet_list:
            raise AssertionError("{} not in support list of core networks: {}".format(
                core_net, ", ".join(self.corenet_list)))
        else:
            self.core_net = core_net.lower()

        self.core_network = None
        self.network = None

    def build_network(self, input_shape, trainable=False):
        """
            Add core network and head network to the model
        """
        if self.core_net == "vgg19":
            self.vgg16_base(input_shape, trainable)

        elif self.core_net == "mobilenetv2":
            self.mobilenetv2_base(input_shape, trainable)

        elif self.core_net == "resnet50":
            self.resnet50_base(input_shape, trainable)

        else:
            print("Wrong Choice")
            exit(-1)

        if self.head_net == "yolov4":
            self.yolov4_head()


# ========================= OBJECT DETECTION HEADS =========================================== 

    def mobilenetv2_base(self, input_shape, trainable=False):
        """
        """
        core_network = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights='imagenet')
        core_network.trainable = trainable

        print("============================== CORE NETWORK =================================")
        print(core_network.summary())
        print("============================== CORE NETWORK ENDS =================================")

        self.core_network = core_network
        return

    def vgg16_base(self, input_shape, trainable=False):
        """
        """
        core_network = tf.keras.applications.vgg19.VGG19(
            input_shape=input_shape, include_top=False, weights='imagenet')
        core_network.trainable = trainable

        print("============================== CORE NETWORK =================================")
        print(core_network.summary())
        print("============================== CORE NETWORK ENDS =================================")

        self.core_network = core_network
        return

    def resnet50_base(self, input_shape, trainable=False):
        """
        """
        
        core_network = tf.keras.applications.ResNet50V2(
            input_shape=input_shape, include_top=False, weights='imagenet')
        core_network.trainable = trainable

        print("============================== CORE NETWORK =================================")
        print(core_network.summary())
        print("============================== CORE NETWORK ENDS =================================")

        self.core_network = core_network
        return

# ========================= OBJECT DETECTION HEADS ===========================================    

    def yolov4_head(self):
        """
        
        """
        x = self.core_network.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        self.network = models.Model(inputs=self.core_network.input, outputs=x)
        print("============================== CORE + HEAD NETWORK =================================")
        print(self.network.summary())
        print("============================== CORE + HEAD NETWORK ENDS =================================")
        return 


