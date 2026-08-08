def fibonacci_sequence(n):
    """Calculates the sequence of numbers where each number is the sum of the two preceding ones."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq
