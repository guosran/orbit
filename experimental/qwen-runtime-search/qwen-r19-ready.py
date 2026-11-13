#!/usr/bin/env python3
"""Owned SSA dependency-ready CPU/main + single-owner NPU execution.

Callbacks are registered implementations, not a Python search. Immutable DAG
validation is done once by PreparedProgram. Actual values stay alive until all
consumers have completed, including asynchronous device readback. The single
NPU owner alone touches its context/BOs. Failure drains the active invocation.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import heapq
from types import MappingProxyType
import threading
import time


class Operation:
    def __init__(self,identity,inputs,outputs,device,call,dependencies=(),priority=0):
        if device not in ('cpu','npu'):raise ValueError('Only actual CPU and serialized NPU resources are supported')
        self.id=identity;self.inputs=tuple(inputs);self.outputs=tuple(outputs)
        self.device=device;self.call=call;self.dependencies=set(dependencies);self.priority=priority


class Result:
    def __init__(self,values,phases=()):self.values=values;self.phases=list(phases)


class PreparedProgram:
    """Certify source dependencies/SSA once; retain no invocation tensors."""
    def __init__(self,operations,input_names,outputs):
        self.inputs=frozenset(input_names);self.outputs=tuple(outputs)
        self.operations={op.id:Operation(op.id,op.inputs,op.outputs,op.device,op.call,op.dependencies,op.priority) for op in operations}
        if len(self.operations)!=len(operations):raise ValueError('Duplicate operation identity')
        producers={};consumers={name:set() for name in self.inputs}
        for op in self.operations.values():
            if len(set(op.outputs))!=len(op.outputs):raise ValueError('Repeated output')
            for name in op.outputs:
                if name in self.inputs or name in producers:raise ValueError('Non-SSA value: '+name)
                producers[name]=op.id;consumers[name]=set()
        self.successors={name:[] for name in self.operations}
        for op in self.operations.values():
            for name in op.inputs:
                if name not in consumers:raise ValueError('Missing source value: '+name)
                consumers[name].add(op.id)
                if name in producers:op.dependencies.add(producers[name])
            if op.id in op.dependencies or not op.dependencies.issubset(self.operations):raise ValueError('Invalid dependency')
            if type(op.priority)!=int:raise ValueError('Ready priority must be an integer')
            for name in op.dependencies:self.successors[name].append(op.id)
        if not set(outputs).issubset(consumers):raise ValueError('Unknown requested output')
        self.consumer_counts={name:len(users) for name,users in consumers.items()}
        self.indegrees={op.id:len(op.dependencies) for op in self.operations.values()}
        pending=[name for name,count in self.indegrees.items() if not count];degree=dict(self.indegrees);checked=0
        while pending:
            name=pending.pop();checked+=1
            for child in self.successors[name]:
                degree[child]-=1
                if degree[child]==0:pending.append(child)
        if checked!=len(self.operations):raise ValueError('Source dependency cycle')
        self.initial_ready={device:tuple(sorted((op.priority,op.id) for op in self.operations.values() if not self.indegrees[op.id] and op.device==device)) for device in ('cpu','npu')}


class ReadyExecutor:
    """CPU caller executes while a persistent (or scoped) device owner waits.

    Future.result uses Condition.wait and releases the GIL. Each device callback
    must additionally use a GIL-releasing device wait; the XRT adapter does so.
    """
    def __init__(self,worker=None):self.worker=worker

    def run(self,operations,inputs,outputs):
        return self.run_prepared(PreparedProgram(list(operations),inputs,outputs),inputs)

    def run_prepared(self,program,inputs):
        if set(inputs)!=program.inputs:raise ValueError('Prepared invocation input ABI mismatch')
        values=dict(inputs);remaining=dict(program.consumer_counts);degree=dict(program.indegrees)
        queues={device:list(program.initial_ready[device]) for device in ('cpu','npu')}
        events=[];lifetimes=[];future=None;active=None;completed=0
        caller=threading.get_ident();stamp=time.perf_counter_ns()
        def invoke(op,args):
            start=time.perf_counter_ns();owner=threading.get_ident()
            if (op.device=='cpu')!=(owner==caller):raise RuntimeError('Execution owner mismatch')
            result=op.call(MappingProxyType(args))
            if not isinstance(result,Result):result=Result(result)
            if set(result.values)!=set(op.outputs):raise ValueError('Implementation output ABI mismatch: '+op.id)
            end=time.perf_counter_ns()
            return result,dict(operator=op.id,device=op.device,owner_thread=owner,start_ns=start,end_ns=end,wall_ns=end-start,phases=result.phases)
        def complete(op,result):
            nonlocal completed
            value,event=result;events.append(event);values.update(value.values);completed+=1
            for name in op.outputs:lifetimes.append(dict(value=name,action='publish',at_ns=event['end_ns'],producer=op.id))
            for name in set(op.inputs):
                remaining[name]-=1
                if remaining[name]==0 and name not in program.outputs and name not in program.inputs:
                    del values[name];lifetimes.append(dict(value=name,action='release',at_ns=event['end_ns'],consumer=op.id))
            for child in program.successors[op.id]:
                degree[child]-=1
                if degree[child]==0:
                    node=program.operations[child];heapq.heappush(queues[node.device],(node.priority,node.id))
        scope=nullcontext(self.worker) if self.worker is not None else ThreadPoolExecutor(max_workers=1,thread_name_prefix='orbit-r19-npu-owner')
        with scope as worker:
            try:
                while completed<len(program.operations):
                    if future is not None and future.done():
                        complete(active,future.result());future=None;active=None
                    if future is None and queues['npu']:
                        _,name=heapq.heappop(queues['npu']);active=program.operations[name]
                        future=worker.submit(invoke,active,{k:values[k] for k in active.inputs})
                    if queues['cpu']:
                        _,name=heapq.heappop(queues['cpu']);op=program.operations[name]
                        complete(op,invoke(op,{k:values[k] for k in op.inputs}))
                    elif future is not None:
                        complete(active,future.result());future=None;active=None
                    elif completed<len(program.operations):raise RuntimeError('Validated DAG has no ready implementation')
            finally:
                # Persistent owners are not closed per graph. Drain the active
                # callback before captured inputs or its BO owners can die.
                if future is not None:future.result()
        end=time.perf_counter_ns()
        # CPU and NPU each serialize their callbacks; every intersection is
        # counted once. Device-active overlap needs actual launch/complete data.
        overlap=sum(max(0,min(a['end_ns'],b['end_ns'])-max(a['start_ns'],b['start_ns'])) for a in events if a['device']=='cpu' for b in events if b['device']=='npu')
        return dict(values={k:values[k] for k in program.outputs},trace=dict(schema='orbit.qwen-r19-ready-execution.v1',execution_policy='source-dependency-ready',
            start_ns=stamp,end_ns=end,wall_ns=end-stamp,events=events,lifetimes=lifetimes,measured_cpu_npu_callback_overlap_ns=overlap,
            overlap_boundary='callback wall intersection; device-active overlap requires actual launch/complete phase timestamps'))
