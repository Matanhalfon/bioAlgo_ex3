####
# matan halfon 205680648
# Tal Gliksman 203834486
####
import argparse
import numpy as np
from itertools import groupby
import ex2_Helper as helper2
from scipy.special import logsumexp

Y_AXIS = 0

X_AXIS = 1

ADDED_LETTERS = 2
SEPERATOR_FOR_E_TABLE = "\t"
NO_MOTIF = -1
NEW_LINE = "\n"
GENERATOR_FINISHED = (None, None)
LETTERS_NUMBER = 6
BACKGROUND_STATES = 4
DONT_WARN = 'ignore'
EMPTY_STRING = ""
WRITING_MODE = 'w'
DIGITS_AFTER_DOT = 2
BACKGROUND_E_FRAME = [[0.25, 0.25, 0.25, 0.25, 0, 0], [0.25, 0.25, 0.25, 0.25, 0, 0], [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0]]
LETTERS_DICT = {"A": 0, "C": 1, "G": 2, "T": 3, "$": 4, "^": 5}
LETTERS = 'ACGT$^'
START_SEQ_SIGN = ">"
B_END_INDEX = 3
FINISH_COL_INDEX_FORWARD = -1


def fastaread(fasta_name):
    '''
    generator for fasta files
    :param fasta_name: name of fasta file
    '''
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(START_SEQ_SIGN)))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = EMPTY_STRING.join(s.strip() for s in next(faiter))
        yield header, seq


def compute_q(seqs, seed):
    """
    compute  the prob that there is a motif in the seq
    :param seqs:a list of seqs
    :param seed: the motif
    :return:the prob to get a motif in the seqs
    """
    counter = 0
    for seq in seqs:
        if seed in seq:
            counter += 1
    with np.errstate(divide=DONT_WARN):
        return counter / len(seqs)


def create_emit_table(seed, alpha):
    """
    create the emission table
    :param seed: the motif
    :param alpha: the prob to get a letter  not from the motif while in motif
    :return: the initial emission table in log space
    """
    e_table = np.full((len(seed) + BACKGROUND_STATES, LETTERS_NUMBER), alpha)
    e_table[:, BACKGROUND_STATES:] = 0
    e_table[:BACKGROUND_STATES] = np.array(BACKGROUND_E_FRAME)
    for i, let in enumerate(seed):
        e_table[i + BACKGROUND_STATES, LETTERS_DICT[let]] = 1 - (3 * alpha)
    with np.errstate(divide=DONT_WARN):
        return np.log(e_table)


def get_sequences(fasta_file):
    """
    parse all of the seq into a list
    :param fasta_file:
    :return: a list which contains the padded sequences
    """
    generator = fastaread(fasta_file)
    sequences = []
    title, seq = next(generator, GENERATOR_FINISHED)
    while seq != None:
        sequences.append(helper2.pad_seq(seq))
        title, seq = next(generator, GENERATOR_FINISHED)
    return sequences


def update_Nkx(fb, seq, pxj, Nkx):
    """
    update the Nkx table
    :param fb: the forward * backward table (posterior)
    :param seq: the seq to compute on
    :param pxj: the ll to get that seq
    :param Nkx: the table that  represent the ratio of the probability  to emit a letter in given state
    :return: the updated Nkx table
    """
    for char in LETTERS:
        char_indexes = [i for i, ltr in enumerate(seq) if ltr == char]
        if len(char_indexes):
            columns = fb[:, char_indexes]
            array = np.array([logsumexp(columns, axis=X_AXIS) - pxj, Nkx[:, LETTERS_DICT[char]]])
            Nkx[:, LETTERS_DICT[char]] = logsumexp(array, axis=Y_AXIS)


def logdotexp(A, B):
    '''
    applys log on the dot product on the exp of the given matrixes
    :param A: firs matrix
    :param B: second matrix
    :return: the solution
    '''
    Astack = np.stack([A] * A.shape[0]).transpose(2, 1, 0)
    Bstack = np.stack([B] * B.shape[1]).transpose(1, 0, 2)
    return logsumexp(Astack + Bstack, axis=Y_AXIS)


def parse_args():
    """
    parse the program arguments
    :return: parse args which contains the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. yeastGenes.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


def computeP(cur, t_table):
    divisorP = logsumexp((cur[helper2.B1_INDEX][helper2.M1_INDEX], cur[helper2.B2_INDEX][helper2.B_END_INDEX]))
    line1 = cur[helper2.B1_INDEX]
    line2 = cur[helper2.B2_INDEX]
    line1[helper2.M1_INDEX] = -np.inf
    line2[helper2.B_END_INDEX] = -np.inf
    divisor1_minus_p = logsumexp(np.concatenate((line1, line2), axis=Y_AXIS).flatten())
    divider = logsumexp(np.concatenate((cur[helper2.B1_INDEX], cur[helper2.B2_INDEX]), axis=Y_AXIS).flatten())
    p = divisorP - divider
    t_table[helper2.B1_INDEX][helper2.M1_INDEX] = p
    t_table[helper2.B1_INDEX][helper2.B1_INDEX] = divisor1_minus_p - divider
    t_table[helper2.B2_INDEX][helper2.B_END_INDEX] = p
    t_table[helper2.B2_INDEX][helper2.B2_INDEX] = divisor1_minus_p - divider
    t_table[helper2.B_START_INDEX][helper2.B1_INDEX] = cur[helper2.B_START_INDEX][helper2.B1_INDEX]
    t_table[helper2.B_START_INDEX][helper2.B2_INDEX] = cur[helper2.B_START_INDEX][helper2.B2_INDEX]


def update_Nkl(seq, f_table, b_table, e_table, pxj, nkl):
    """
    update the Nkl table that represents the transtion ratio of probabilities
    :param seq: a given seq
    :param f_table: the forward table
    :param b_table: the backward table
    :param e_table: the emission table
    :param pxj: the ll of the given seq
    :param nkl: the last nkl table
    :return: the updated nkl table
    """
    seq = seq[1:]
    f_table = f_table[:, :-1]
    b_table = b_table[:, 1:]
    for i, char in enumerate(seq):
        b_table[:, i] += e_table[:, LETTERS_DICT[seq[i]]]
    update = logdotexp(f_table, b_table.T)
    cur = logsumexp(np.stack([update - pxj, nkl], axis=Y_AXIS), axis=Y_AXIS)
    return cur


def normlaize_rows(table):
    '''
    normalize the rows of matrix to sum 1
    :param table: table to normalize
    :return: the normalized table
    '''
    with np.errstate(divide=DONT_WARN):
        sums = logsumexp(table, axis=X_AXIS)
    with np.errstate(invalid=DONT_WARN):
        table -= sums.reshape((sums.shape[0], 1))
        table[np.isnan(table)] = -np.inf
        return table


def e_table_to_save(e_table):
    '''
    fits e_table for the format of motif_profile file
    :param e_table: e_table
    :return: e_table for the file
    '''
    e_table = np.exp(e_table[BACKGROUND_STATES:, :-ADDED_LETTERS]).T
    e_table = np.round(e_table, DIGITS_AFTER_DOT)
    rows = []
    for i in range(len(e_table)):
        row = [str(l) for l in e_table[i]]
        rows.append(SEPERATOR_FOR_E_TABLE.join(row))
    return NEW_LINE.join(rows)


def main():
    '''
    applies Baum Welch algorithm on args and generates file of ll history, file of motif profile, file of motif positions
    '''
    with open("ll_history.txt", WRITING_MODE) as ll_history:
        args = parse_args()
        sequences = get_sequences(args.fasta)
        e_table = create_emit_table(args.seed, args.alpha)
        q = compute_q(sequences, args.seed)
        with np.errstate(divide=DONT_WARN):
            t_table = np.log(helper2.get_transition(args.p, q, len(e_table[:, 0])))
        prev_ll = 0
        cur_ll = np.inf
        while (np.abs(cur_ll - prev_ll)) > args.convergenceThr:
            prev_ll = cur_ll
            cur_ll = 0
            nkx = np.full(e_table.shape, -np.inf)
            nkl = np.full(t_table.shape, -np.inf)
            for seq in sequences:
                f_table = helper2.forward(seq, e_table, t_table)
                b_table = helper2.backward(seq, e_table, t_table)
                cur_ll += f_table[B_END_INDEX][FINISH_COL_INDEX_FORWARD]
                fb = (f_table + b_table)
                pxj = f_table[B_END_INDEX][FINISH_COL_INDEX_FORWARD]
                update_Nkx(fb, seq, pxj, nkx)
                nkl = update_Nkl(seq, f_table, b_table, e_table, pxj, nkl)
            nkl = nkl + t_table
            e_table[BACKGROUND_STATES:] = nkx[BACKGROUND_STATES:]
            e_table = normlaize_rows(e_table)
            computeP(nkl, t_table)
            t_table = normlaize_rows(t_table)
            ll_history.writelines([str(cur_ll) + NEW_LINE])
    with open("motif_profile.txt", WRITING_MODE) as motif_profile:
        motif_profile.writelines([e_table_to_save(e_table), NEW_LINE, str(
            round(np.exp(t_table[helper2.B_START_INDEX, helper2.B1_INDEX]), DIGITS_AFTER_DOT)), NEW_LINE,
                                  str(round(np.exp(t_table[helper2.B1_INDEX, helper2.M1_INDEX]),
                                            DIGITS_AFTER_DOT))])
    with open("motif_positions", WRITING_MODE) as motif_positions:
        for seq in sequences:
            s_table = helper2.get_viterbi_tables(seq, e_table, t_table)
            path = helper2.cover_viterbi_path(s_table)
            index = np.argwhere(path == helper2.M1_INDEX)
            if len(index):
                index = index[0][0]
            else:
                index = NO_MOTIF
            motif_positions.write(str(index) + NEW_LINE)


if __name__ == '__main__':
    main()
