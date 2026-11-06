def reverse_bits(rank, power):
    key = 0
    for bit in range(power.bit_length() - 1):
        key = 2 * key + (rank >> bit & 1)
    return key

def collective(nodes, elements, algorithm=None, inactive=(), gpu=(), transport='mpi', max_ranks=10):
    n = len(nodes)
    assert transport in ('mpi', 'direct_tcp')
    assert len(set(nodes)) == n and n >= 2 and (max_ranks is None or n <= max_ranks) and (elements > 0)
    power = 1 << n.bit_length() - 1
    algorithm = algorithm or (6 if power >= 4 and elements * 4 >= 131072 else 3)
    assert algorithm in (3, 6) and (algorithm != 6 or elements >= power)
    rank_of = {host: rank for rank, host in enumerate(nodes)}
    assert set(inactive) <= set(nodes) and set(gpu) <= set(nodes)
    size = 4 * elements
    phases = []

    def work(rank, count=0, copy_bytes=0, zero_bytes=0, operation='reduce'):
        return {'host': nodes[rank], 'rank': rank, 'operation': operation, 'F': count, 'D': 12 * count + 2 * copy_bytes + zero_bytes, 'native_graph_calls': 0}

    def message(source, destination, count, offset=0):
        return {'source': nodes[source], 'destination': nodes[destination], 'source_rank': source, 'destination_rank': destination, 'elements': count, 'offset_elements': offset, 'M': 4 * count}
    initial = []
    for host in nodes:
        rank = rank_of[host]
        if host in gpu and host not in inactive:
            initial.append(work(rank, copy_bytes=size, operation='GPU_to_host_staging'))
        if host in inactive:
            initial.append(work(rank, zero_bytes=size, operation='inactive_zero_fill'))
    phases.append({'name': 'prepare', 'messages': [], 'work': initial})
    phases.append({'name': 'fold_excess', 'messages': [message(r, r - power, elements) for r in range(power, n)], 'work': [work(r - power, elements, operation='fold_add') for r in range(power, n)]})
    order = [reverse_bits(rank, power) for rank in range(power)]
    if algorithm == 3:
        if transport == 'mpi':
            phases.append({'name': 'recursive_doubling_initial_copy', 'messages': [], 'work': [work(r, copy_bytes=size, operation='MPI_initial_copy') for r in range(power)]})
        in_result = [False] * power
        mask = 1
        while mask < power:
            exchanges = [message(order[key], order[key ^ mask], elements) for key in range(power)]
            for key in range(power):
                if key < key ^ mask:
                    in_result[key] = not in_result[key]
            phases.append({'name': 'recursive_doubling_' + str(mask), 'messages': exchanges, 'work': [work(r, elements) for r in range(power)]})
            mask *= 2
        if transport == 'mpi':
            phases.append({'name': 'recursive_doubling_final_copy', 'messages': [], 'work': [work(order[key], copy_bytes=size, operation='MPI_final_copy') for key in range(power) if not in_result[key]]})
    else:
        windows = [(0, elements)] * power
        history = []
        mask = 1
        while mask < power:
            steps, exchanges, reductions = ([], [], [])
            for key, (offset, count) in enumerate(windows):
                left = count // 2
                if key < key ^ mask:
                    recv_offset, recv_count, send_offset, send_count = (offset, left, offset + left, count - left)
                else:
                    recv_offset, recv_count, send_offset, send_count = (offset + left, count - left, offset, left)
                steps.append((recv_offset, recv_count, send_offset, send_count))
                exchanges.append(message(order[key], order[key ^ mask], send_count, send_offset))
                reductions.append(work(order[key], recv_count))
            phases.append({'name': 'reduce_scatter_' + str(mask), 'messages': exchanges, 'work': reductions})
            windows = [(s[0], s[1]) for s in steps]
            history.append((mask, steps))
            mask *= 2
        for mask, steps in reversed(history):
            phases.append({'name': 'allgather_' + str(mask), 'messages': [message(order[key], order[key ^ mask], s[1], s[0]) for key, s in enumerate(steps)], 'work': []})
    phases.append({'name': 'return_excess', 'messages': [message(r - power, r, elements) for r in range(power, n)], 'work': []})
    phases.append({'name': 'finish', 'messages': [], 'work': [work(rank_of[host], copy_bytes=size, operation='host_to_GPU_staging') for host in gpu]})
    totals = {host: {'tx_bytes': 0, 'rx_bytes': 0, 'F': 0, 'D': 0} for host in nodes}
    for phase in phases:
        for msg in phase['messages']:
            totals[msg['source']]['tx_bytes'] += msg['M']
            totals[msg['destination']]['rx_bytes'] += msg['M']
        for row in phase['work']:
            totals[row['host']]['F'] += row['F']
            totals[row['host']]['D'] += row['D']
    return {'nodes': nodes, 'elements': elements, 'algorithm': algorithm, 'active_power': power, 'transport': transport, 'bit_reversed_active_ranks': order, 'phases': phases, 'totals': totals, 'scope': 'Logical user payload and explicit reduction/staging memory work only; not an elapsed-time prediction.'}
