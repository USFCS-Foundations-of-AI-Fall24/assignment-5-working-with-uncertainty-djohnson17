

import random
import argparse

#from setuptools.dist import sequence


# from torchgen.api.types import streamT


# import codecs
# import os
# import numpy
# from pandas.core.common import random_state
# from sympy import sequence


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions if transitions else {}
        self.emissions = emissions if emissions else {}

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        if '#' not in self.transitions:
            self.transitions['#'] = {}

        trans_file = f"{basename}.trans"
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    state_from_or_state, state_to_or_observation, prob = parts
                    if state_from_or_state not in self.transitions:
                        self.transitions[state_from_or_state] = {}
                    self.transitions[state_from_or_state][state_to_or_observation] = float(prob)


        emit_file = f"{basename}.emit"
        with open(emit_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, observation, prob = parts
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][observation] = float(prob)



## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        current_state = '#'

        stateseq = [current_state]
        outputseq = []

        for _ in range(n):
            next_state = random.choices(
                population=list(self.transitions[current_state].keys()),
                weights=list(self.transitions[current_state].values())
            )[0]

            if next_state in self.emissions:
                output = random.choices(
                    population=list(self.emissions[next_state].keys()),
                    weights=list(self.emissions[next_state].values())
                )[0]
            else:
                output = "default_observation"

            stateseq.append(next_state)
            outputseq.append(output)

            current_state = next_state

        return Sequence(stateseq, outputseq)

    def forward(self, sequence):
        observation = sequence.outputseq
        num_observations = len(observation)
        states = list(self.transitions.keys())

        alpha = [{} for _ in range(num_observations)]

        for state in self.transitions['#']:
            emission = self.emissions.get(state, {}).get(observation[0], 0) if state != '#' else 1
            alpha[0][state] = self.transitions['#'][state] * emission

        for t in range(1, num_observations):
            for state in states:
                emission_prob = self.emissions.get(state, {}).get(observation[t], 0)
                alpha[t][state] = sum(
                    alpha[t-1][prev_state] * self.transitions[prev_state].get(state, 0) *
                    emission_prob
                    for prev_state in alpha[t-1]
                )
        final_probs = alpha[-1]
        most_probable_state = max(final_probs, key=final_probs.get)

        if 'X' in most_probable_state:
            print(f"Most probable state: {most_probable_state} (Safe to land)")
        else:
            print(f"Most probable state: {most_probable_state} (Not safe to land)")

        return most_probable_state

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        observation = sequence.outputseq
        if not observation:
            raise ValueError("Observation sequence is empty.")

        num_observations = len(observation)
        states = list(self.transitions.keys())

        viterbi = [{} for _ in range(num_observations)]
        backpointer = [{} for _ in range(num_observations)]

        for state in self.transitions['#']:
            emission_prob = self.emissions.get(state, {}).get(observation[0], 0) if state != '#' else 1
            viterbi[0][state] = self.transitions['#'].get(state, 0) * emission_prob
            backpointer[0][state] = '#'

        for t in range(1, num_observations):
            for state in states:
                if state  == '#':
                    continue
                max_prob, prev_state = max(
                    (viterbi[t-1][prev_state] * self.transitions[prev_state].get(state,0) *
                     self.emissions[state].get(observation[t], 0), prev_state)
                    for prev_state in self.transitions if prev_state != '#'
                )
                viterbi[t][state] = max_prob
                backpointer[t][state] = prev_state

        final_state = max(viterbi[-1], key=viterbi[-1].get)

        most_likely_states = [final_state]
        for t in range(num_observations - 1, 0, -1):
            most_likely_states.insert(0, backpointer[t][most_likely_states[0]])

        print(f"Most likely sequence of states: {' '.join(most_likely_states)}")
        return most_likely_states
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.




def main():
    parser = argparse.ArgumentParser(description="HMM Sequence Generator")
    parser.add_argument('basename', help='The base name of the files (without extension)')
    parser.add_argument('--generate', type=int, help='Generate a sequence of the given length')
    parser.add_argument('--forward', help='Run forward algorithm on the given observation sequence file')
    parser.add_argument('--viterbi', help='Run Viterbi algorithm on the given observation sequence file')
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print("Generated sequence:")
        print(sequence)

        most_probable_state = hmm.forward(sequence)
        print(f"Most probable final state: {most_probable_state}")

    elif args.forward:
        with open(args.forward, 'r') as f:
            lines = f.readlines()
            observations = lines[0].strip().split()

        sequence = Sequence([], observations)

        most_probable_state = hmm.forward(sequence)
        print(f"Most probable final state: {most_probable_state}")

    elif args.viterbi:
        with open(args.viterbi, 'r') as f:
            lines = f.readlines()
        observations = [word for line in lines for word in line.strip().split() if word.strip()]


        sequence = Sequence([], observations)
        most_likely_states = hmm.viterbi(sequence)
        print(f"Most probable final state sequence: {' '.join(most_likely_states)}")

if __name__ == "__main__":
    main()



