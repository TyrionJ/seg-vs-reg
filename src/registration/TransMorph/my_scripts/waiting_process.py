from tqdm import tqdm
from typing import Any
from time import sleep
import multiprocessing as mp


def waiting_proc(r, ps: Any, desc=None):
    remaining = list(range(len(r)))

    with tqdm(desc=desc, total=len(r), disable=not desc) as pbar:
        while len(remaining) > 0:
            all_alive = all([j.is_alive() for j in ps._pool])
            if not all_alive:
                raise RuntimeError('Some background worker is break.')
            done = [i for i in remaining if r[i].ready()]
            remaining = [i for i in remaining if i not in done]
            pbar.update(len(done))
            sleep(0.02)


def process_one(idx1, idx2):
    return idx1 + idx2


if __name__ == '__main__':
    rs = []
    with mp.get_context('spawn').Pool(12) as p:
        for idx in range(100):
            rs.append(p.starmap_async(process_one, ((idx, idx), )))
        waiting_proc(rs, p, 'State')
    results = [r.get()[0] for r in rs]

    print(results)
