import numpy as np


def symbol2bin(symbol, width):
    symbol = np.atleast_2d(symbol)[0]
    biterror = np.empty((len(symbol), width), dtype=np.bool)
    new_symbol = symbol.copy()
    new_symbol.shape = -1, 1
    for i in np.arange(width):
        biterror[:, i] = ((new_symbol >> i) & 1)[:, 0]

    return biterror


def calc_ber(rx_symbols, tx_symbols, qam_order):
    '''
        rx_symbols: 1d-array
        tx_symbols: 1d-array
        qam_order: int

        return res
    '''
    nbit = int(np.log2(qam_order))
    if 2 ** nbit <= np.max(rx_symbols) or 2 ** nbit <= np.max(tx_symbols):
        raise Exception("Qam order is wrong")

    rx_symbols = np.atleast_2d(rx_symbols)[0]
    tx_symbols = np.atleast_2d(tx_symbols)[0]
    mask = rx_symbols != tx_symbols

    error1 = symbol2bin(rx_symbols[mask], nbit)
    error2 = symbol2bin(tx_symbols[mask], nbit)
    error_num = np.sum(np.logical_xor(error1, error2))
    return error_num / (len(rx_symbols) * nbit)


def decision(decision_symbols,const):
    decision_symbols = np.atleast_2d(decision_symbols)
    const = np.atleast_2d(const)[0]
    res = np.zeros_like(decision_symbols,dtype=np.complex128)
    for row_index,row in enumerate(decision_symbols):
        for index,symbol in enumerate(row):
            index_min = np.argmin(np.abs(symbol - const))
            res[row_index,index] = const[index_min]
    return res

def from_symbol_tomsg(symbols,order):
    import joblib
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))

    constl = joblib.load(BASE + '/constl')[order][0]

    msg = np.zeros_like(symbols,dtype=np.int)

    for row_index,row in enumerate(symbols):
        for temp_index,temp in enumerate(row):
            dis = np.abs(temp - constl)
            msg_one = np.argmin(dis)
            msg[row_index,temp_index] = msg_one

    return msg




