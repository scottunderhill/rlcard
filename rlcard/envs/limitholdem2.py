import json
import os
import numpy as np

import rlcard
from rlcard.envs.env import Env
from rlcard.games.limitholdem.game import LimitholdemGame as Game

class LimitholdemEnv2(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, allow_step_back=False):
        ''' Initialize the Limitholdem environment
        '''
        super().__init__(Game(allow_step_back), allow_step_back)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape=[72]

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        processed_state = {}

        legal_actions = [self.actions.index(a) for a in state['legal_actions']]
        processed_state['legal_actions'] = legal_actions

        public_cards = state['public_cards']
        hand = state['hand']
        raise_nums = state['raise_nums']

        # Change 1, public cards and hand should not be combined. 
        # Card encoding format SA
        # We will use a 17 vector to represent this. This is in not specific order
        lookupsuite = {'S' : 0 , 'C' : 1, 'D' : 2, 'H' : 3} 
        lookupcard = {'J' : 10, 'Q' : 11, 'K' : 12, 'A' : 13}
        indexcard = lookupcard.get(hand[0][1],hand[0][1])
        indexsuite  = lookupcard.get(hand[0][0])
        
        indexcard2 = lookupcard.get(hand[1[1],hand[1][1])
        indexsuite2  = lookupcard.get(hand[1][0])
        
        handindex = np.zeros(34)
        #encoding with suite 
        handindex[indexcard - 1] = 1;
        handindex[13 + indexsuite] = 1
        handindex[17 + indexcard2 - 1] = 1;
        handindex[17 + 13 + indexsuite2] = 1
        
        
        cards = public_cards 
        idx = [self.card2index[card] for card in cards]
        obs = np.zeros(52)
        obs[idx] = 1
        obs.append(handindex)
        obs.append(np.zeros(20))
        for i, num in enumerate(raise_nums):
            obs[52 + 34 + i * 5 + num] = 1
        processed_state['obs'] = obs

        return processed_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]
