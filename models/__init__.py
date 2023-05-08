from .GridGCN import GridGCN

def model_select(name):
    name = name.upper()

    if name in ("GRIDGCN", "GCN"):
        return GridGCN
    else:
        raise NotImplementedError
    