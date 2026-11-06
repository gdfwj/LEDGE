GPU_CLASSES = {'Orin', 'Xavier'}

def nccl_rounds(network, nodes, size, operation):
    n = len(nodes)
    if n < 2:
        return []
    power = 1 << n.bit_length() - 1
    rounds = []

    def phase(name, pairs):
        if pairs:
            rounds.append(dict(name=name, messages=[network.cost(nodes[a], nodes[b], size) for a, b in pairs]))
    if operation in ('allreduce', 'reduce'):
        phase('nccl_fold', [(r, r - power) for r in range(power, n)])
        distance = power // 2
        while distance:
            pairs = [(r, r ^ distance) for r in range(power)] if operation == 'allreduce' else [(r, r - distance) for r in range(distance, 2 * distance)]
            phase('nccl_' + operation + '_' + str(distance), pairs)
            distance //= 2
    if operation == 'broadcast':
        distance = 1
        while distance < power:
            phase('nccl_broadcast_' + str(distance), [(r, r + distance) for r in range(distance)])
            distance *= 2
    if operation in ('allreduce', 'broadcast'):
        phase('nccl_copy_back', [(r - power, r) for r in range(power, n)])
    return rounds

def collective_rounds(network, nodes, size, classes, policy):
    gpu = [h for h in nodes if classes[h] in GPU_CLASSES]
    if policy == 'native_mpi' or len(gpu) < 2:
        return network.tp_collective(nodes, size)['rounds']
    if len(gpu) == len(nodes):
        return nccl_rounds(network, gpu, size, 'allreduce')
    bridge = [h for h in nodes if h == gpu[0] or h not in gpu]
    return nccl_rounds(network, gpu, size, 'reduce') + network.tp_collective(bridge, size)['rounds'] + nccl_rounds(network, gpu, size, 'broadcast')

def policy_for_row(row):
    explicit = row.get('collective_policy')
    if explicit is not None:
        if explicit not in ('native_mpi', 'nccl_hierarchical'):
            raise ValueError('Unknown collective_policy: ' + explicit)
        return explicit
    return 'nccl_hierarchical'
