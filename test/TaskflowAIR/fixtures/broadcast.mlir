module {
 func.func @broadcast(%a: memref<1x16xf32> {orbit.role="input",orbit.name="A"},
                      %c: memref<4x16xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias} {
  %done = taskflow.task @broadcast will_reads(%a : memref<1x16xf32>) will_writes(%c : memref<4x16xf32>) [original_read_memrefs(%a : memref<1x16xf32>), original_write_memrefs(%c : memref<4x16xf32>)] : (memref<1x16xf32>,memref<4x16xf32>) -> memref<4x16xf32> {
  ^bb0(%in: memref<1x16xf32>,%out: memref<4x16xf32>):
   affine.for %i=0 to 4 {
    affine.for %j=0 to 16 {
     %v=affine.load %in[0,%j] : memref<1x16xf32>
     affine.store %v,%out[%i,%j] : memref<4x16xf32>
    }
   }
   taskflow.yield done_writes(%out : memref<4x16xf32>)
  }
  return
 }
}
