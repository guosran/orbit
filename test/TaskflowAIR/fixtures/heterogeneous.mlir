module {
 func.func @pipeline(%a:memref<16xf32> {orbit.role="input",orbit.name="A"},
                     %b:memref<16xf32> {orbit.role="input",orbit.name="B"},
                     %c:memref<16xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias} {
  %t=memref.alloc() {orbit.name="T"}:memref<16xf32>
  %u=memref.alloc() {orbit.name="U"}:memref<16xf32>
  %td=taskflow.task @add will_reads(%a,%b:memref<16xf32>,memref<16xf32>) will_writes(%t:memref<16xf32>) [original_read_memrefs(%a,%b:memref<16xf32>,memref<16xf32>),original_write_memrefs(%t:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%y:memref<16xf32>,%z:memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %w=affine.load %y[%i]:memref<16xf32>
    %r=arith.addf %v,%w:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  %ud=taskflow.task @scale will_reads(%td:memref<16xf32>) will_writes(%u:memref<16xf32>) [original_read_memrefs(%td:memref<16xf32>),original_write_memrefs(%u:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%z:memref<16xf32>):
   %two=arith.constant 2.0:f32
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %r=arith.mulf %v,%two:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  %cd=taskflow.task @finish will_reads(%ud,%b:memref<16xf32>,memref<16xf32>) will_writes(%c:memref<16xf32>) [original_read_memrefs(%ud,%b:memref<16xf32>,memref<16xf32>),original_write_memrefs(%c:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%y:memref<16xf32>,%z:memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %w=affine.load %y[%i]:memref<16xf32>
    %r=arith.addf %v,%w:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  memref.dealloc %u:memref<16xf32>
  memref.dealloc %t:memref<16xf32>
  return
 }
}
