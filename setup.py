from distutils.core import setup, Extension

sources = ["bindings/bindings.cpp",
           "bindings/GameBinding.cpp", "Game/Game.cpp", "Game/Player.cpp",
           "NeuralNetwork/NN.cpp", "NeuralNetwork/Topology.cpp", "Private/Connection.cpp",
           "Private/Generation.cpp", "Private/Layer.cpp", "Private/Mutation.cpp",
           "Private/MutationField.cpp", "Private/Neuron.cpp", "Private/Phenotype.cpp",
           "Private/Random.cpp", "Private/routines.cpp", "Private/Species.cpp",
           "Serializer/Serializable.cpp", "Train/Train.cpp", "Timer.cpp",
           "bindings/python.cpp"]

module = Extension('neat', sources=sources)

setup(name='neat',
      version='1.0',
      description='neat',
      ext_modules=[module])
