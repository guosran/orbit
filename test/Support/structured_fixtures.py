"""Small explicit test inputs; no production planner or numerical runtime."""
import hashlib
import json

SCHEMA = "orbit.taskflow.plan.v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_plan(workload, m=5, n=19, k=23, d=16, i=24):
    buffers, tasks = [], []

    def buffer(name, rows, cols, role):
        buffers.append(dict(id=name, shape=[rows, cols], dtype="f32", role=role))

    def matmul(name, a, b, c, rows, cols, reduction, deps=()):
        tasks.append(dict(id=name, depends_on=list(deps), operations=[
            dict(kind="fill", inputs=[], output=c, m=[0, rows], n=[0, cols]),
            dict(kind="matmul", inputs=[a, b], output=c, m=[0, rows], n=[0, cols],
                 k=[0, reduction], accumulate=True)]))

    if workload in ("gemm", "gemv", "gemm_relu"):
        if workload == "gemv":
            m = 1
        buffer("A", m, k, "input")
        buffer("B", k, n, "input")
        buffer("C", m, n, "temporary" if workload == "gemm_relu" else "output")
        matmul("gemm", "A", "B", "C", m, n, k)
        outputs = ["C"]
        if workload == "gemm_relu":
            buffer("R", m, n, "output")
            tasks.append(dict(id="relu", depends_on=["gemm"], operations=[
                dict(kind="relu", inputs=["C"], output="R", m=[0, m], n=[0, n])]))
            outputs = ["R"]
    elif workload == "ffn":
        for name, rows, cols in (("X", m, d), ("W_gate", d, i), ("W_up", d, i), ("W_down", i, d)):
            buffer(name, rows, cols, "input")
        for name in ("G", "U", "H"):
            buffer(name, m, i, "temporary")
        buffer("Y", m, d, "output")
        matmul("gate", "X", "W_gate", "G", m, i, d)
        matmul("up", "X", "W_up", "U", m, i, d)
        tasks.append(dict(id="activation", depends_on=["gate", "up"], operations=[
            dict(kind="swiglu", inputs=["G", "U"], output="H", m=[0, m], n=[0, i])]))
        matmul("down", "H", "W_down", "Y", m, d, i, ["activation"])
        outputs = ["Y"]
    else:
        raise ValueError(f"unsupported fixture: {workload}")
    plan = dict(schema=SCHEMA, buffers=buffers, tasks=tasks, outputs=outputs, transforms=[])
    plan["source_id"] = fingerprint(plan)
    plan["candidate_id"] = plan["source_id"]
    return plan


def chain_source(m,k,h,n):
    shapes={'X':[m,k],'W1':[k,h],'H':[m,h],'W2':[h,n],'Y':[m,n]}
    p=dict(schema='orbit.taskflow.plan.v1',buffers=[dict(id=b,shape=s,dtype='f32',role='input' if b in ('X','W1','W2') else 'output' if b=='Y' else 'temporary') for b,s in shapes.items()],tasks=[],outputs=['Y'],transforms=[])
    for name,a,b,c,kk,nn,deps in [('first','X','W1','H',k,h,[]),('second','H','W2','Y',h,n,['first'])]:
        p['tasks'].append(dict(id=name,depends_on=deps,operations=[dict(kind='fill',inputs=[],output=c,m=[0,m],n=[0,nn]),dict(kind='matmul',inputs=[a,b],output=c,m=[0,m],n=[0,nn],k=[0,kk],accumulate=True)]))
    p['source_id']=fingerprint(p);p['candidate_id']=p['source_id'];return p
