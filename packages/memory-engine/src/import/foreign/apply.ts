/**
 * import/foreign/apply.ts — write ported weights into a live model.
 *
 * Kept separate from the port itself so the mapping stays device-free and
 * testable: {@link portForeignMamba} produces plain `Float32Array`s, this is the
 * only piece that needs a `GPUDevice`.
 *
 * Matching is by parameter NAME, never by position, so a change to
 * `HybridMambaModel.parameters()` ordering can never silently mis-assign a
 * buffer — an unexpected or missing name throws instead.
 */

import type { HybridMambaModel } from "../../model/mamba_model.js";
import { uploadBuffer } from "../../utils/gpu_utils.js";
import type { PortedTensors } from "./plan.js";

/**
 * Upload every ported tensor into `model`'s parameter buffers.
 *
 * Throws when the model expects a parameter the port did not produce, when an
 * element count disagrees, or when the port produced a name the model does not
 * have — all three mean the model was built from a different config than
 * {@link PortedCheckpoint.modelConfig}.
 */
export function applyPortedWeights(model: HybridMambaModel, ported: PortedTensors): void {
  const params = model.parameters();
  const expected = new Set(params.map((p) => p.name));

  const extra = [...ported.weights.keys()].filter((name) => !expected.has(name));
  if (extra.length > 0) {
    throw new Error(
      `import/foreign: ported tensors ${extra.slice(0, 5).map((n) => `"${n}"`).join(", ")}` +
        `${extra.length > 5 ? ` (+${extra.length - 5} more)` : ""} have no parameter on this model — ` +
        `build it from the port's modelConfig`,
    );
  }

  for (const param of params) {
    const data = ported.weights.get(param.name);
    if (!data) {
      throw new Error(
        `import/foreign: no ported tensor for parameter "${param.name}" — ` +
          `the model was built from a different config than the port's modelConfig`,
      );
    }
    if (data.length !== param.numel) {
      throw new Error(
        `import/foreign: parameter "${param.name}" expects ${param.numel} elements, ` +
          `the port produced ${data.length}`,
      );
    }
    uploadBuffer(model.device, param.buf, data);
  }
}
