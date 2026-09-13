Google just mapped the complete fruit fly brain (139,255 neurons, 54.5M synapses).

So I taught it how to play Wordle.

Using a biological spiking neural network (LIF) connected to a 3D biomechanical embodiment, the connectome plays Wordle against human players in real time:

- Smell (Antennae): Letters A-Z injected as chemical odor currents into 26 olfactory receptor neuron (ORN) classes.
- Eyes (Optic Lobes): Tile positions and color feedback (Green / Yellow / Gray) encoded retinotopically across visual neurons.
- Reward (Dopamine): PAM / PPL1 dopaminergic neurons emit reward surges upon solving, reinforcing the readout weights.

The RL pipeline:
1. Sensory encoding into real FlyWire connectome circuit (3,127 active neurons, 179,516 synapses)
2. Mushroom Body working memory candidate filtering
3. Linear readout trained with policy-gradient reinforcement learning

Benchmark results:
- Average guesses: 4.12 per word (optimal mathematical solver is ~3.42)
- Win rate: 97.0% across the dictionary (99.0% on 100-word benchmark)
- Decision latency: ~28 ms per guess
- Baseline random guesser: < 2% win rate

Full open-source code, 3D embodiment, and interactive web demo:
https://github.com/Quantum-Coded/fruitfly

#Neuroscience #MachineLearning #Connectomics #AI #BioInspiredAI #Drosophila #GoogleResearch #ReinforcementLearning
