import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from numpy import mean
from torch.autograd import Variable
import experience_replay, image_preprocessing
import gym


class CNN(nn.Module):
    def __init__(self, num_actions):
        super(CNN, self).__init__()
        '''
        Definiamo 3 strati convoluti. Ogni strato passerà allo strato successivo tante immagini quante feature riesce a rilevare
        - in_channels = input dello strato (colori, 1 è bianco e nero)
        - out_channel = output dello strato (quante immagini vuoi in output, ovvero quante feature.All'inizio mettere 32)
        - kernel_size = dimensione del feature detector
        '''
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        '''
        Ora Definiamo lo strato Flatten che ritornerà il vettore da mandare agli strati full connection
        '''
        self.fc1 = nn.Linear(self.count_neurons((1, 80, 80)), 40)
        self.fc2 = nn.Linear(40, num_actions)

    def count_neurons(self, image_dim):
        """
        image_dim = dimensione dell'immagine (80x80) + channel
        Per definire le reti, creiamo una immagine farlocca, in quanto non abbiamo ancora immagini da doom
        - Applichiamo lo strato convoluto all'immagine
        - Applichiamo maxpooling al risultato dello strato convoluto
        - applichiamo la rectified function al risultato del pooling
        - Applichiamo lo strato flatten allo strato finale (x.data.view(1, -1))
        """
        x = Variable(torch.rand(1, *image_dim))
        x = self.convolution1(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)

        x = self.convolution2(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)

        x = self.convolution3(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = self.convolution1(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)

        x = self.convolution2(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)

        x = self.convolution3(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=1)
        actions = probs.multinomial()
        return actions


'''
In questa classe, definiamo la logica della AI. Riceverà in input le immagini di Doom, le propagherà alla network e dopodichè 
applicherà la softmax function per ricavare le azioni da eseguire
'''


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        """
        In questa funzione:
        - convertiamo prima le immagini di input in numpy array e poi convertiamo in tensor.
        - dopodichè propaghiamo il tensor all'interno degli strati della convolutional network
        - infine applichiamo la funzione softmax per avere le azioni da eseguire
        - ritorniamo le azioni in un numpy array
        """
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        q_values = self.brain.forward(input)
        actions = self.body.forward(q_values)
        return actions.data.numpy()


'''
In questa parte, definiamo l'enviroment usando OpenAI Gym, definiamo come il modello viene allenato e usiamo la tecnica dell'Eligibility Trace
'''
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width=80, height=80)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)
number_actions = doom_env.action_space.n

cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T=1.0)
ai = AI(cnn, softmax_body)


'''
In questa parte, definiamo l'implementazione dell'Experience Replay con l'eligibilty trace (Asynchronous n.step Q-Learning). Quest'ultima, ci consente di imparare da un "batch" di reward
in quanto la rete ci ritornerà una serie di azioni, non una sola.
batch = contiene input e targets
series = è una istanza di una named tuple definita nel file experience_replay
'''
n_steps = experience_replay.NStepProgress(doom_env, ai, 10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)
def eligibility_trace(batch):
    gamma = 0.99
    inputs, targets = [], []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn.forward(input)
        cumul_reward = 0.0 if series[-0].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = cumul_reward * gamma + step.reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


'''
Definiamo la classe per avere la media di 100 steps
'''
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return mean(self.list_of_rewards)


ma = MA(100)


'''
Alleniamo infine l'agent
- definiamo optimizer e funzione di loss (mean square function)
- in ogni epoca, l'ai performerà 200 azioni, e la funzione sample_batch ritornerà un batch di size = parametro del metodo
'''
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(1, num_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn.forward(inputs)
        loss_error = loss.forward(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

