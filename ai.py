import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# state = {3 signali, orientazione, -orientazione}
# Transition = (last state, last action, last reward, next state)

# Creo La Rete Neurale

class Network(nn.Module):  # Questa classe eredita dalla classe Module di torch

    def __init__(self, input_size, nb_action):  # nb_action è il numero delle azioni che può fare
        super(Network,
              self).__init__()  # Chiamo nel costruttore della classe children, il costruttore della classe Module
        self.input_size = input_size
        self.nb_action = nb_action
        # Questa variabile inizializza la connessione tra gli input e l'hidden layer dei neuroni (num input, num neuroni del hidden layer)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)
        # Questa variabile inizializza la connessione tra l'hidden layer e output (num neuroni nella 1 full connection, num output ovvero le azioni)
        self.fc3 = nn.Linear(30, nb_action)

    # Questo metodo è quello responsabile della forward propagation della rete (come impara). Ritorna il nostro valore Q
    def forward(self, state):  # state è l'input della rete in base al Deep Q-Learning
        # Attiviamo l'hidden layer attraverso la rectifile function
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        # Ritorniamo il valore di Q, passandolo alla full connection dove risiede il layer output
        q_values = self.fc3(l2)
        return q_values


# Implementazione dell'Experience Replay
class ExperienceReplay(object):
    def __init__(self,
                 capacity):  # capacity è la grandezza della memoria (numero di transazioni che l'agente conserva prima di inviarla alla rete
        self.capacity = capacity
        self.memory = []  # dove vengono salvate le transazioni (last state, action, reward, next state)

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:  # se la memoria supera la capacità, rimuovo transazione più vecchia
            del self.memory[0]

    def sample(self, batch_size):
        # Prendo i miei sample in maniera randomica con la size di batch size. La funzione zip permette di raggruppare in tuple tutti gli stati, tutte le azioni e tutte le reward
        samples = zip(*random.sample(self.memory, batch_size))  # a = [(1,2,3),(4,5,6)]  zip(*a) = ((1, 4), (2, 5), (3, 6))
        return map(lambda x: Variable(torch.cat(x, 0)),samples)  # ritorniamo i sample in variabili torch che contengono un tensor e un gradiente


# Implementazione del Deep Q-Learning
class Dqn:
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma  # delay coefficiente
        self.reward_window = []  # dove risiedono i reward
        self.model = Network(input_size, nb_action)
        self.memory = ExperienceReplay(100000)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=0.001)  # optimizer per stocastic gradient descent (lr e quanto veloce apprende il modello)
        self.last_state = torch.Tensor(input_size).unsqueeze(
            0)  # Tensor è una matrice multidimensionale che contiene elementi di un sigolo tipo. unsqueeze lo converte in unidirezionale
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):  # state è un Tensor (orientation, signal 1, signal 2, signal 3)
        # implementazione di Softmax per la scelta dell'azione da intraprendere
        probs = F.softmax(self.model(Variable(state)) * 100, dim=1)  # self.model() == self.model.forward()
        action = probs.multinomial(num_samples=1)  # prende l'azione da intraprendere in base alle probabilità
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)  # calcoliamo il loss che poi propagheremo alla rete
        # usiamo l'optimizer per aggiornare le weight della rete. Ad ogni iterazione bisogna reinizializzare l'optimizer per via del gradient descent
        self.optimizer.zero_grad()
        # Back propagation
        td_loss.backward(retain_graph=True)
        # Update weight
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # La rete impara quando la memoria ha almeno 100 transazioni salvate in memoria
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    # Ritorna la media dei reward
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    # Salva il modello della rete neurale all'ultimo weight
    def save(self):
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
                   "last_brain.pth")  # state_dict è una funzione che ereditiamo da nn.Module

    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> Loading checkpoint...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> Done!")
        else:
            print("=> No checkpoint found!")
