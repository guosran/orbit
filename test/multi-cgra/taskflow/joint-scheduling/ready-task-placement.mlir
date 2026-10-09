// RUN: python3 %S/run-ready-task-placement-checks.py --optimizer mlir-amoeba-opt --fixture %s --architecture %S/ready-task-placement-arch.yaml --output %t
// Scheduling-only regression: supplied durations isolate spatial/temporal policy.
module {
 func.func @two(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>, %d: memref<16xi32>, %x: memref<16xi32>, %y: memref<16xi32>) attributes {joint_scheduling_candidate_id="placement-regression", joint_scheduling_fixed_point_max_iterations=64:i64} {
  %t0 = taskflow.task @First will_reads(%a, %b, %c, %d : memref<16xi32>, memref<16xi32>, memref<16xi32>, memref<16xi32>) will_writes(%x:memref<16xi32>) [original_read_memrefs(%a,%b,%c,%d:memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>), original_write_memrefs(%x:memref<16xi32>)] {cgra_count=4:i32,cgra_shape="2x2",amoeba.joint_shape_orientation_fixed,profile_info={duration=10:i64}} : (memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>) -> memref<16xi32> {
  ^bb0(%i0:memref<16xi32>,%i1:memref<16xi32>,%i2:memref<16xi32>,%i3:memref<16xi32>,%o:memref<16xi32>):
    taskflow.yield done_writes(%o:memref<16xi32>)
  }
  %t1 = taskflow.task @Second will_reads(%a, %b, %c, %d : memref<16xi32>, memref<16xi32>, memref<16xi32>, memref<16xi32>) will_writes(%y:memref<16xi32>) [original_read_memrefs(%a,%b,%c,%d:memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>), original_write_memrefs(%y:memref<16xi32>)] {cgra_count=4:i32,cgra_shape="2x2",amoeba.joint_shape_orientation_fixed,profile_info={duration=10:i64}} : (memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>,memref<16xi32>) -> memref<16xi32> {
  ^bb0(%i0:memref<16xi32>,%i1:memref<16xi32>,%i2:memref<16xi32>,%i3:memref<16xi32>,%o:memref<16xi32>):
    taskflow.yield done_writes(%o:memref<16xi32>)
  }
 return
 }
}
