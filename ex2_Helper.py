####
#matan halfon 205680648
#Tal Gliksman 203834486
####
import argparse
import pandas as pd
import numpy as np
from scipy.special import logsumexp

BACKGROUND_E_FRAME = [[0.25, 0.25, 0.25, 0.25, 0, 0], [0.25, 0.25, 0.25, 0.25, 0, 0], [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0]]
LETTERS = 'ACGT$^'
B_END_LETTER = '$'
B_START_LETTER = '^'
BACKGROUND = 'B'
MOTIF = 'M'
ROW_INDEX = 0
COL_INDEX = 1
INITIATION_COL_INDEX_FORWARD = 0
INITIATION_COL_INDEX_BACKWARD = -1
BACKGROUND_STATES_NUM = 4
NO_EMISSION_PROBABILITY = 0
SURE_TRANSIT = 1
SURE_START = 1
LAST_STATE_IN_MOTIF = 1
B1_INDEX = 0
B2_INDEX = 1
B_START_INDEX = 2
B_END_INDEX = 3
M1_INDEX = 4
MK_INDEX = -1
LETTERS_DICT = {"A": 0, "C": 1, "G": 2, "T": 3, "$": 4, "^": 5}
MAX_LINE_LEN = 50
NEW_LINE = '\n'
NO_DELIMITER = ''
ORIGINAL_SEQ_START_INDEX = 1
ORIGINAL_SEQ_END_INDEX = -1
EMIT_SEP = '\t'
FINISH_COL_INDEX_FORWARD = -1
FINISH_COL_INDEX_BACKWARD = 0


#HELPER

def parse_initial_emission(file):
    '''
    parses the initial emission file
    :param file: tsv file contains the emission probabilities of the states in the motif
    :return:emition_table
    '''
    background = pd.DataFrame(BACKGROUND_E_FRAME, columns=list(LETTERS))
    df = pd.read_csv(file, sep=EMIT_SEP)
    df[B_END_LETTER] = NO_EMISSION_PROBABILITY
    df[B_START_LETTER] = NO_EMISSION_PROBABILITY
    df = background.append(df)
    return df.values


def get_transition(p, q, states_number):
    '''
    calculates transition probability between states
    :param p:probability to move from B1 to M1
    :param q:probability to move from Bstart to B1
    :param states_number: the number of states
    :return: transition table thu
    '''
    thu = np.zeros((states_number, states_number))
    thu[B1_INDEX][B1_INDEX] = 1 - p
    thu[B1_INDEX][M1_INDEX] = p
    thu[B2_INDEX][B2_INDEX] = 1 - p
    thu[B2_INDEX][B_END_INDEX] = p
    thu[B_START_INDEX][B1_INDEX] = q
    thu[B_START_INDEX][B2_INDEX] = 1 - q
    thu[MK_INDEX][B2_INDEX] = SURE_TRANSIT
    for i in range(BACKGROUND_STATES_NUM, states_number - LAST_STATE_IN_MOTIF):
        thu[i][i + 1] = SURE_TRANSIT
    return thu

def get_viterbi_tables(seq, e_table, t_table):
    '''
    computes the viterbi table for the seq and return the state table
    :param seq: DNA sequence
    :param e_table: the emission table
    :param t_table: the transition table
    :return: states table- the state from which we arrived to each state
    '''
    v_table = init_table(t_table, len(seq), B_START_INDEX, INITIATION_COL_INDEX_FORWARD)
    states_number = t_table.shape[ROW_INDEX]
    states_table = np.zeros((states_number, len(seq))).astype(np.int)
    for i in range(1, len(seq)):
        v_mat = np.tile(v_table[:, i - 1], (t_table.shape[ROW_INDEX], 1))
        addition = v_mat + t_table.T
        v_table[:, i] = np.max(addition, axis=COL_INDEX) + e_table[:, LETTERS_DICT[seq[i]]]
        states_table[:, i] = np.argmax(addition, axis=COL_INDEX)
    return states_table


def pad_seq(seq):
    '''
    pads sequence with ^ at the beggining and $ at the end
    :param seq: sequence to pad
    :return: pad sequence
    '''
    return B_START_LETTER + seq + B_END_LETTER


def cover_viterbi_path(s_table):
    '''
    return the state shift along the seq
    :param s_table: the score table that set the liklehood  for the state according to the seq
    :return:the state shift
    '''
    seq_len = len(s_table[ROW_INDEX])
    path = np.zeros(seq_len)
    current_state = B_END_INDEX
    for i in range(seq_len - 1, 0, -1):
        current_state = s_table[current_state][i]
        path[i - 1] = current_state
    return path[ORIGINAL_SEQ_START_INDEX:ORIGINAL_SEQ_END_INDEX]

def print_path(path, seq):
    """
    print the vitrebi path next to the seq
    :param path: the state shift along the seq
    :param seq: the seq
    """
    path = NO_DELIMITER.join([BACKGROUND if l in (B1_INDEX, B2_INDEX) else MOTIF for l in path])
    index = 0
    for i in range(len(seq) // MAX_LINE_LEN):
        print(path[i * MAX_LINE_LEN:(i + 1) * MAX_LINE_LEN])
        print(seq[i * MAX_LINE_LEN:(i + 1) * MAX_LINE_LEN] + NEW_LINE)
        index += 1
    print(path[index * MAX_LINE_LEN:])
    print(seq[index * MAX_LINE_LEN:]+NEW_LINE)


def init_table(t_table, seq_len, init_state_index, init_col_index):
    """
    init a table for the forward and backward algorithm
    :param t_table: the transition table
    :param seq_len: the length of the seq
    :param init_state_index:  the index where the state start for the algorithm
    :param init_col_index: the col to start the algorithm
    :return: the table of zeros with an an 1 probability on the starting state (preform log on all of
    the table to prevant underflow)
    """
    states_number = t_table.shape[ROW_INDEX]
    f_table = np.zeros((states_number, seq_len))
    f_table[init_state_index][init_col_index] = SURE_START
    with np.errstate(divide='ignore'):
        return np.log(f_table)


def forward(seq, e_table, t_table):
    """
    run the forward algorithm
    :param seq: the seq
    :param e_table: the emit table
    :param t_table: the transtion table
    :return: the table after the forward algorithm fill with the log liklehood for etch state and emition
    """
    f_table = init_table(t_table, len(seq), B_START_INDEX, INITIATION_COL_INDEX_FORWARD)
    for i in range(1, len(seq)):
        f_mat = np.tile(f_table[:, i - 1], (t_table.shape[ROW_INDEX], 1))
        e_mat = np.tile(e_table[:, LETTERS_DICT[seq[i]]], (t_table.shape[ROW_INDEX], 1)).T
        addition = f_mat + t_table.T + e_mat
        f_table[:, i] = logsumexp(addition, axis=COL_INDEX)
    return f_table


def backward(seq, e_table, t_table):
    """
       run the backward algorithm
       :param seq: the seq
       :param e_table: the emit table
       :param t_table: the transtion table
       :return: the table after the backward algorithm fill with the log liklehood for etch state and emition
       """
    b_table = init_table(t_table, len(seq), B_END_INDEX, INITIATION_COL_INDEX_BACKWARD)
    for i in range(len(seq) - 1, 0, -1):
        b_mat = np.tile(b_table[:, i], (t_table.shape[ROW_INDEX], 1))
        e_mat = np.tile(e_table[:, LETTERS_DICT[seq[i]]], (t_table.shape[ROW_INDEX], 1))
        addition = b_mat + t_table + e_mat
        b_table[:, i - 1] = logsumexp(addition, axis=COL_INDEX)
    return b_table


def posterior(seq, e_table, t_table):
    """
    compute the posterior in each letter in the seq
    :param seq: DNA seq
    :param e_table: emission table
    :param t_table: transition table
    :return: the posterior
    """
    f_table = forward(seq, e_table, t_table)
    b_table = backward(seq, e_table, t_table)
    posterior_table = f_table + b_table
    maxies = posterior_table.argmax(axis=ROW_INDEX)
    return maxies[ORIGINAL_SEQ_START_INDEX:ORIGINAL_SEQ_END_INDEX]


