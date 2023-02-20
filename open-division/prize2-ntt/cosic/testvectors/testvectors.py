import random

q = 2**64 - 2**32 + 1

omegas = {2: 18446744069414584320, 4: 281474976710656, 8: 16777216, 16: 4096, 32: 64, 64: 8, 128: 18446741870424883713, 256: 5575382163818481237, 512: 6968564197111712876, 1024: 16597218757394956566, 2048: 12224595309747104101, 4096: 7162411506628992569, 8192: 17370749796773648850, 16384: 6336109244748055921, 32768: 1899055609354042605, 65536: 13852080018225038782, 131072: 10923341407476031646, 262144: 11741558660741847279,
          524288: 10857530471142180514, 1048576: 9295618658634918088, 2097152: 127589073375127907, 4194304: 16468075455353463192, 8388608: 3875249612062100546, 16777216: 1178141731658273728, 33554432: 16741045073888939265, 67108864: 18116652949657751419, 134217728: 16548855811671180642, 268435456: 11004664669452058453, 536870912: 13456279912083982979, 1073741824: 4738080411982103950, 2147483648: 6752129933716898348, 4294967296: 11724716146725638212}

for (N, omega_N) in omegas.items():
    assert (pow(omega_N, N//2, q) == q-1)


try:
    from tqdm import tqdm
    timer = tqdm(total=2**25)
except ModuleNotFoundError:
    timer = None


def ntt(x):

    if timer:
        timer.update()

    N = len(x)

    if N <= 1:
        return x

    even = ntt(x[0::2])
    odd = ntt(x[1::2])

    odd_x_twiddle = [pow(omegas[N], k, q) * odd[k] % q for k in range(N//2)]

    return [(even[k] + odd_x_twiddle[k]) % q for k in range(N//2)] + [(even[k] - odd_x_twiddle[k]) % q for k in range(N//2)]


if __name__ == "__main__":

    random.seed(0)
    x = [random.randint(0, q) for _ in range(2**24)]

    inFile = f'in_fully_random_2_24.txt'
    outFile = f'out_fully_random_2_24.txt'

    print("Writing input testvector")
    with open(inFile, 'w') as f:
        f.write('\n'.join(str(coeff) for coeff in x))

    print("Computing NTT")
    y = ntt(x)

    print("Writing output testvector")
    with open(outFile, 'w') as f:
        f.write('\n'.join(str(coeff) for coeff in y))
