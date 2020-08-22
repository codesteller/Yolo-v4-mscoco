import tensorflow as tf 


class Network:
    def __init__(self, core_net, head_net):
      self.core_net = core_net
      self.head_net = head_net

    def build_model(self):
        """
        Add core network and head network to the model
        """
