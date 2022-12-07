/* tslint:disable */
/* eslint-disable */
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @returns {Array<any>}
*/
export function compute_msm(point_vec: PointVectorInput, scalar_vec: ScalarVectorInput): Array<any>;
/**
*/
export class PointVectorInput {
  free(): void;
/**
* @param {number} size
*/
  constructor(size: number);
/**
* @returns {Array<any>}
*/
  toJsArray(): Array<any>;
/**
* @param {Array<any>} arr
* @returns {PointVectorInput}
*/
  static fromJsArray(arr: Array<any>): PointVectorInput;
}
/**
*/
export class ScalarVectorInput {
  free(): void;
/**
* @param {number} size
*/
  constructor(size: number);
/**
* @returns {Array<any>}
*/
  toJsArray(): Array<any>;
/**
* @param {Array<any>} arr
* @returns {ScalarVectorInput}
*/
  static fromJsArray(arr: Array<any>): ScalarVectorInput;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_pointvectorinput_free: (a: number) => void;
  readonly pointvectorinput_new: (a: number) => number;
  readonly pointvectorinput_toJsArray: (a: number) => number;
  readonly pointvectorinput_fromJsArray: (a: number) => number;
  readonly __wbg_scalarvectorinput_free: (a: number) => void;
  readonly scalarvectorinput_new: (a: number) => number;
  readonly scalarvectorinput_toJsArray: (a: number) => number;
  readonly scalarvectorinput_fromJsArray: (a: number) => number;
  readonly compute_msm: (a: number, b: number) => number;
}

/**
* Synchronously compiles the given `bytes` and instantiates the WebAssembly module.
*
* @param {BufferSource} bytes
*
* @returns {InitOutput}
*/
export function initSync(bytes: BufferSource): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
