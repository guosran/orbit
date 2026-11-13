#!/usr/bin/env python3
"""Original decode plus next-position host-only RoPE preparation overlap.

The current token embedding/launch/KV append/readback/argmax remain ordered.
Only _rope_slab(p+1), which reads immutable biases and the position, may run
while position p is on the device. It never writes the current RMS/BO/KV.
The original arithmetic is called exactly; no speculative token is consumed.
"""
import threading
import time


def make_decoder(decoder,profiled_class,ready,last_position):
    if decoder.elf_mode:raise ValueError('r19 ready decode requires certified xclbin dispatch')
    from triton.backends.amd_triton_npu.multilaunch import _wait
    native_kernel=decoder.kern;native_rope=decoder._rope_slab
    state={};prefetched={};launched=threading.Event()
    class Run:
        def __init__(self,native):self.native=native
        def wait(self,*args):
            # pyxrt.wait may hold the GIL. Reuse the original backend's proven
            # npu_wait/polling implementation instead, then report actual state.
            state['wait_start_ns']=time.perf_counter_ns()
            _wait(self.native)
            state['complete_ns']=time.perf_counter_ns()
            return self.native.state()
    class Kernel:
        def __call__(self,*args):
            before=time.perf_counter_ns();run=native_kernel(*args);after=time.perf_counter_ns()
            state['launch']=(before,after);launched.set();return Run(run)
    def rope(position):
        cached=prefetched.pop(position,None)
        return cached if cached is not None else native_rope(position)
    decoder.kern=Kernel();decoder._rope_slab=rope
    original=profiled_class(decoder)
    class ReadyDecoder:
        def __init__(self):self.dispatch_events=[]
        def dispatch(self,token,position):
            launched.clear();state.clear();begin=time.perf_counter_ns()
            if position>=last_position:
                # No unused next-position task is charged to the final token.
                result=ready.ReadyExecutor().run([
                    ready.Operation(f'decode.{position}.dispatch',['token','position'],['logits'],'npu',
                        lambda values:{'logits':original.dispatch(values['token'],values['position'])})],
                    {'token':token,'position':position},['logits'])
                event=dict(original.dispatch_events[-1]);end=time.perf_counter_ns()
                extra=end-begin-event['wall_ns']
                if extra<0:raise RuntimeError('Final ready decode wall smaller than original dispatch')
                event['wall_ns']=end-begin;event['host_control_ns']+=extra
                event['ready_execution']=result['trace']
                self.dispatch_events.append(event);return result['values']['logits']
            def device(values):
                try:return {'logits':original.dispatch(values['token'],values['position'])}
                except BaseException as error:
                    state['error']=error;launched.set();raise
            def host(values):
                if not launched.wait(60):raise RuntimeError('Decode did not reach an actual launch')
                if 'error' in state:raise state['error']
                # A distinct array, not a write to the active r_bo. Original
                # dispatch performs next token's sync only after its dependency.
                begin=time.perf_counter_ns();value=native_rope(values['position']+1);end=time.perf_counter_ns()
                return ready.Result({'next_rope':value},[dict(kind='next-position-pure-host-rope',start_ns=begin,end_ns=end,wall_ns=end-begin,position=values['position']+1,bo_writes=0)])
            result=ready.ReadyExecutor().run([
                ready.Operation(f'decode.{position}.dispatch',['token','position'],['logits'],'npu',device),
                ready.Operation(f'decode.{position+1}.host-rope',['position'],['next_rope'],'cpu',host)],
                {'token':token,'position':position},['logits','next_rope'])
            prefetched[position+1]=result['values']['next_rope']
            end=time.perf_counter_ns();event=dict(original.dispatch_events[-1]);extra=end-begin-event['wall_ns']
            if extra<0:raise RuntimeError('Nested decode wall exceeds enclosing ready dispatch')
            event['wall_ns']=end-begin;event['host_control_ns']+=extra
            event['ready_execution']=result['trace']
            event['source_transformation']='next position pure host RoPE moved into current device wait; token/KV dependencies retained'
            event['prefetch_bo_writes']=0
            event['launch_boundary_ns']=list(state['launch'])
            event['device_complete_ns']=state['complete_ns']
            cpu=next(e for e in result['trace']['events'] if e['device']=='cpu')['phases'][0]
            event['pure_host_rope_device_overlap_ns']=max(0,min(cpu['end_ns'],state['complete_ns'])-max(cpu['start_ns'],state['launch'][1]))
            event['exclusive_sum_equals_wall']=True
            self.dispatch_events.append(event)
            return result['values']['logits']
    return ReadyDecoder()
