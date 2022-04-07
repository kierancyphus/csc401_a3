import os
import re
import string
from typing import List

import numpy as np

dataDir = '/u/cs401/A3/data/'
dataDir = "./data"


def print_backtrack(a):
    for row in a:
        print(row)


def get_errors(backtrack, matrix):
    insertion, substitution, deletion = 0, 0, 0
    current = backtrack[-1][-1]
    prev = (len(backtrack) - 1, len(backtrack[0]) - 1)
    while current is not None:
        i_prev, j_prev = prev
        i, j = current

        # print(f"@({i_prev}, {j_prev}) going {prev} -> {current}")

        if matrix[i, j] == matrix[i_prev, j_prev]:
            # do nothing since this was valid
            pass
        elif i_prev - 1 == i and j_prev - 1 == j:
            substitution += 1
        elif i_prev - 1 == i:
            deletion += 1
        else:
            insertion += 1

        prev = current
        current = backtrack[i][j]

    return substitution, insertion, deletion


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    # edge cases
    if len(r) == 0:
        return np.inf, 0, len(h), 0

    if len(h) == 0:
        return 1.0, 0, 0, len(r)

    # initialize - need to add start and end tags
    r = ["<s>"] + r + ["</s>"]
    h = ["<s>"] + h + ["</s>"]

    m, n = len(r), len(h)
    matrix = np.zeros((m, n))
    backtrack = [[None for _ in range(n)] for _ in range(m)]
    for i in range(1, n):
        backtrack[0][i] = (0, i - 1)

    for j in range(1, m):
        backtrack[j][0] = (j - 1, 0)

    # initialization
    matrix[0, :] = np.arange(n)
    matrix[:, 0] = np.arange(m)

    for i in range(1, m):
        for j in range(1, n):
            diagonal, above, left = matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i, j - 1]
            min_value = np.min((diagonal, above, left))

            matrix[i, j] = min_value if r[i] == h[j] else min_value + 1
            if min_value == diagonal:
                backtrack[i][j] = (i - 1, j - 1)
            elif min_value == above:
                backtrack[i][j] = (i - 1, j)
            else:
                backtrack[i][j] = (i, j - 1)

    # print_backtrack(backtrack)
    substitution, insertion, deletion = get_errors(backtrack, matrix)

    # print(f"insertion, substitution, deletion = {insertion}, {substitution}, {deletion}")
    return matrix[-1, -1] / (m - 2), substitution, insertion, deletion


def test_distance():
    r = "who is there".split()
    h = "is there".split()
    assert Levenshtein(r, h) == (1 / 3, 0, 0, 1)

    r = "who is there".split()
    h = "".split()
    assert Levenshtein(r, h) == (1.0, 0, 0, 3)

    h = "who is there".split()
    r = "".split()
    assert Levenshtein(r, h) == (np.inf, 0, 3, 0)

    return


def process_file(filepath: str) -> List[List[str]]:
    with open(filepath, 'r') as f:
        lines = f.readlines()

        if len(lines) == 0:
            return []

        processed_lines = []
        for line in lines:
            transcript = " ".join(line.split()[2:])
            # remove anything in [] or <> brackets
            transcript = re.sub(r"\<[^\>]*\>|\[[^\]]*\]", "", transcript)
            transcript = transcript.translate(str.maketrans('', '', string.punctuation))
            transcript = transcript.lower()
            processed_lines.append(transcript.split())

    return processed_lines


if __name__ == "__main__":
    test_distance()

    google_wers, kaldi_wers = [], []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            google_path = os.path.join(dataDir, speaker, "transcripts.Google.txt")
            kaldi_path = os.path.join(dataDir, speaker, "transcripts.Kaldi.txt")
            transcript_path = os.path.join(dataDir, speaker, "transcripts.txt")

            transcript_lines = process_file(transcript_path)
            googles_lines = process_file(google_path)
            kaldi_lines = process_file(kaldi_path)

            if len(transcript_lines) != 0 and len(googles_lines) != 0:
                for i, (t_line, g_line) in enumerate(zip(transcript_lines, googles_lines)):
                    wer, substitution, insertion, deletion = Levenshtein(t_line, g_line)
                    print(f"{speaker} Google {i} {wer} S:{substitution}, I:{insertion}, D:{deletion}")
                    google_wers.append(wer)

            if len(transcript_lines) != 0 and len(kaldi_lines) != 0:
                for i, (t_line, k_line) in enumerate(zip(transcript_lines, kaldi_lines)):
                    wer, substitution, insertion, deletion = Levenshtein(t_line, k_line)
                    print(f"{speaker} Kaldi {i} {wer} S:{substitution}, I:{insertion}, D:{deletion}")
                    kaldi_wers.append(wer)

    print(f"[Google] average wer: {np.average(google_wers)}  std dev: {np.std(google_wers)} [Kaldi] average wer: {np.average(kaldi_wers)}  std dev: {np.std(kaldi_wers)}")
