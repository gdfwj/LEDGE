def compile_dag(job):
    tasks = []
    cursor = None
    pending = None
    compute_start = None
    ops = job['ops']
    i = 0

    def add(kind, deps=(), **kw):
        task = {'kind': kind, 'deps': list(dict.fromkeys((x for x in deps if x is not None))), **kw}
        tasks.append(task)
        return len(tasks) - 1

    def rounds(seq, start, meta, ready=None):
        at = start
        for k, rd in enumerate(seq):
            offsets = ready if k == 0 and ready else {}
            at = add('round', [at], messages=[dict(m, source_ready=offsets.get(m['source'], 0.0), target_ready=offsets.get(m['target'], 0.0)) for m in rd['messages']], duration=rd['floor'] + max(offsets.values(), default=0.0), token=meta['token'], layer=meta.get('layer'), microbatch=meta.get('microbatch'), work=rd['name'])
        return at
    while i < len(ops):
        op = ops[i]
        if op['kind'] == 'compute':
            compute_start = cursor
            collective = ops[i + 1] if i + 1 < len(ops) and ops[i + 1]['kind'] == 'collective' else None
            gpu = [r for r in op['ranks'] if r['device'] in job['gpu_devices']]
            cpu = [r for r in op['ranks'] if r not in gpu]
            starts = {}
            workids = []
            for rank in op['ranks']:
                prefix = pending if rank in gpu else None
                start = add('join', [cursor, prefix])
                starts[rank['device']] = start
                part = dict(op, ranks=[rank], duration=rank['duration'])
                workids.append(add('work', [start], duration=rank['duration'], op=part, nominal_start=cursor))
            if gpu:
                pending = None
            if collective:
                rds = collective['rounds']
                mixed = collective['hierarchical'] and gpu and cpu and (len(gpu) > 1)
                if mixed:
                    a = next((k for k, r in enumerate(rds) if not r['name'].startswith('nccl_')))
                    b = next((k for k in range(a, len(rds)) if rds[k]['name'].startswith('nccl_')), len(rds))
                    gpu_start = starts[gpu[0]['device']]
                    reduction = rounds(rds[:a], gpu_start, op, {r['device']: r['duration'] for r in gpu})
                    joined = add('join', workids + [reduction])
                    cursor = rounds(rds[a:b], joined, op)
                    pending = rounds(rds[b:], cursor, op) if b < len(rds) else None
                    add('join', [cursor], sync=workids, token=op['token'], layer=op['layer'])
                else:
                    start = add('join', list(starts.values()))
                    cursor = rounds(rds, start, op, {r['device']: r['duration'] for r in op['ranks']})
                    cursor = add('join', workids + [cursor], sync=workids, token=op['token'], layer=op['layer'])
                i += 1
            else:
                cursor = add('join', workids, sync=workids, token=op['token'], layer=op['layer'])
        elif op['kind'] == 'messages':
            seq = op['rounds']
            if seq and all((m.get('command') == 'GET_request' for m in seq[0]['messages'])):
                request = rounds(seq[:1], compute_start, op)
                cursor = add('join', [cursor, request])
                cursor = rounds(seq[1:], cursor, op)
            else:
                cursor = rounds(seq, cursor, op)
        elif op['kind'] == 'token':
            cursor = add('token', [cursor], token=op['token'])
        else:
            raise ValueError('Unconsumed collective dependency')
        i += 1
    if pending is not None:
        raise ValueError('Unconsumed GPU broadcast at request boundary')
    return tasks
