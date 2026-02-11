import * as THREE from "three";
import { MeshBVH } from "three-mesh-bvh";
/**
 * Finds the minimal distance between two crowns along the center-to-center vector
 * such that the gap or overlap is just below the given threshold (in mm).
 * Uses binary search and .intersectsGeometry for speed.
 * @param {THREE.BufferGeometry} geom1 - Fixed geometry
 * @param {THREE.BufferGeometry} geom2 - Movable geometry
 * @param {THREE.Vector3} center1 - Center of geom1
 * @param {THREE.Vector3} center2 - Center of geom2
 * @param {number} threshold - Minimal gap/overlap (in mm)
 * @param {number} [maxSteps=20] - Binary search steps
 * @returns {number} Target distance between centers
 */
export function findTargetDistanceBinarySearch(mesh1, mesh2, center1, center2, threshold = 0.01, maxSteps = 20) {
  // Direction from center1 to center2
  const direction = new THREE.Vector3().subVectors(center2, center1).normalize();
  const initialDistance = center1.distanceTo(center2);
  // Use mesh.geometry and mesh.matrixWorld
  const geom1 = mesh1.geometry;
  const geom2 = mesh2.geometry;
  // Ensure BVH exists
  if (!geom1.boundsTree) geom1.boundsTree = new MeshBVH(geom1);
  if (!geom2.boundsTree) geom2.boundsTree = new MeshBVH(geom2);
  // Build transform from mesh1 to mesh2
  const initialMatrix = new THREE.Matrix4()
    .copy(mesh1.matrixWorld).invert()
    .multiply(mesh2.matrixWorld);
  // debug session 
  const box1 = geom1.boundingBox || (geom1.computeBoundingBox(), geom1.boundingBox);
  const box2 = geom2.boundingBox || (geom2.computeBoundingBox(), geom2.boundingBox);
  const localCenter1 = new THREE.Vector3();
  const localCenter2 = new THREE.Vector3();
  box1.getCenter(localCenter1);
  box2.getCenter(localCenter2);
  console.log(`Geom1 center: (${localCenter1.x.toFixed(2)}, ${localCenter1.y.toFixed(2)}, ${localCenter1.z.toFixed(2)})`);
  console.log(`Geom2 center: (${localCenter2.x.toFixed(2)}, ${localCenter2.y.toFixed(2)}, ${localCenter2.z.toFixed(2)})`);
  console.log(`findTargetDistanceBinarySearch: initial distance = ${center1.distanceTo(center2).toFixed(4)} mm`);
  console.log(`findTargetDistanceBinarySearch: direction = (${direction.x.toFixed(4)}, ${direction.y.toFixed(4)}, ${direction.z.toFixed(4)})`);
  console.log(`findTargetDistanceBinarySearch: initialCollision = ${geom1.boundsTree.intersectsGeometry(geom2, initialMatrix)}`);

  const initialCollision = geom1.boundsTree.intersectsGeometry(geom2, initialMatrix);
  let low, high;
  if (initialCollision) {
    // If initially intersecting, move geom2 outward until no collision
    low = initialDistance;
    high = initialDistance;
    let step = 0.01;
    let found = false;
    for (let i = 0; i < maxSteps; i++) {
      high += step;
      const moveVec = direction.clone().multiplyScalar(high - initialDistance);
      // Смещаем matrixWorld mesh2
      const movedMatrixWorld = mesh2.matrixWorld.clone().premultiply(new THREE.Matrix4().makeTranslation(moveVec.x, moveVec.y, moveVec.z));
      const testMatrix = mesh1.matrixWorld.clone().invert().multiply(movedMatrixWorld);
      if (!geom1.boundsTree.intersectsGeometry(geom2, testMatrix)) {
        found = true;
        break;
      }
      step *= 1.; // Exponential search for speed
    }
    if (!found) return high; // Could not find non-colliding position
    // Now binary search between low (colliding) and high (non-colliding)
  } else {
    // If not intersecting, search inward
    low = 0;
    high = initialDistance;
  }
  let result = high;
  for (let i = 0; i < maxSteps; i++) {
    const mid = (low + high) / 2;
    const moveVec = direction.clone().multiplyScalar(mid - initialDistance);
    const movedMatrixWorld = mesh2.matrixWorld.clone().premultiply(new THREE.Matrix4().makeTranslation(moveVec.x, moveVec.y, moveVec.z));
    const testMatrix = mesh1.matrixWorld.clone().invert().multiply(movedMatrixWorld);
    const collision = geom1.boundsTree.intersectsGeometry(geom2, testMatrix);
    if (collision) {
      low = mid;
    } else {
      result = mid;
      high = mid;
    }
    if (Math.abs(high - low) < threshold) break;
  }
  console.log(`findTargetDistanceBinarySearch: target distance = ${result.toFixed(4)} mm`);
  return result;
}

function cost(centers, targetDists) {
  // centers: Array of [x, y, z] or THREE.Vector3
  // targetDists: Array of numbers
  const dists = centers.slice(1).map((c, i) =>
    new THREE.Vector3().subVectors(c, centers[i]).length()
  );
  return dists.reduce((sum, d, i) => sum + Math.pow(d - targetDists[i], 2), 0);
}

function distResolve(centers, targetDists, lr = 0.1, accuracy = 0.001, maxSteps = 1000) {
  // centers: Array of THREE.Vector3
  // targetDists: Array of numbers
  const n = centers.length;
  for (let step = 0; step < maxSteps; step++) {
    for (let i = 1; i < n - 1; i++) {
      const left = centers[i - 1];
      const right = centers[i + 1];
      const curr = centers[i];

      const leftDist = curr.distanceTo(left);
      const rightDist = curr.distanceTo(right);

      const gradLeft = curr.clone().sub(left).multiplyScalar((leftDist - targetDists[i - 1]) / (leftDist + 1e-6));
      const gradRight = curr.clone().sub(right).multiplyScalar((rightDist - targetDists[i]) / (rightDist + 1e-6));
      const grad = gradLeft.add(gradRight);

      curr.sub(grad.multiplyScalar(lr));
    }
    if (cost(centers, targetDists) < accuracy) {
      console.log(`stops iterating on step ${step}`);
      break;
    }
  }
  return centers;
}

/**
 * Computes target distances for all neighboring teeth using binary search,
 * then optimizes centers using gradient descent so all teeth are positioned
 * with no gaps or collisions.
 * @param {Array<THREE.BufferGeometry>} geometries - Array of tooth geometries (ordered)
 * @param {Array<THREE.Vector3>} centers - Array of initial crown centers (ordered)
 * @param {number} threshold - Minimal gap/overlap (in mm)
 * @param {number} lr - Learning rate for distResolve
 * @param {number} accuracy - Cost threshold for distResolve
 * @param {number} maxSteps - Max steps for distResolve
 * @returns {{centers: Array<THREE.Vector3>, targetDists: Array<number>}}
 */
export function SpaseCollisionResolver(meshes, centers, threshold = 0.01, lr = 0.1, accuracy = 0.001, maxSteps = 1000) {
  // Compute target distances for all neighboring teeth
  // console.log(`centers from SpaseCollisionResolver:`);
  centers.forEach((c, i) => {
    // console.log(`Tooth ${i}: (${c.x.toFixed(2)}, ${c.y.toFixed(2)}, ${c.z.toFixed(2)})`);
  });
  const targetDists = [];
  for (let i = 0; i < meshes.length - 1; i++) {
    const id1 = meshes[i]?.userData?.toothId || i;
    const id2 = meshes[i+1]?.userData?.toothId || (i+1);
    const c1 = centers[i];
    const c2 = centers[i+1];
    const initialDist = c1.distanceTo(c2);
    const d = findTargetDistanceBinarySearch(
      meshes[i],
      meshes[i + 1],
      centers[i],
      centers[i + 1],
      threshold
    );
    console.log(`Target distance for ${id1}–${id2}: ${d.toFixed(2)} mm`);
    targetDists.push(d);
  }
  // Deep copy centers to avoid mutating input
  const centersCopy = centers.map(c => c.clone());
  console.log(`centers before optimization:`);
  centersCopy.forEach((c, i) => {
    console.log(`Tooth ${i}: (${c.x.toFixed(2)}, ${c.y.toFixed(2)}, ${c.z.toFixed(2)})`);
  });
  // Optimize centers using gradient descent
  const optimizedCenters = distResolve(centersCopy, targetDists, lr, accuracy, maxSteps);
  return { centers: optimizedCenters, targetDists };
}
