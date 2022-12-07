/* tslint:disable */
/* eslint-disable */
/**
* @param {Uint8Array} data
* @returns {InstanceObjectVector}
*/
export function deserialize_msm_inputs(data: Uint8Array): InstanceObjectVector;
/**
* @param {number} size
* @returns {InstanceObject}
*/
export function generate_msm_inputs(size: number): InstanceObject;
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @returns {PointOutput}
*/
export function compute_msm_baseline(point_vec: PointVectorInput, scalar_vec: ScalarVectorInput): PointOutput;
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @returns {PointOutput}
*/
export function compute_msm(point_vec: PointVectorInput, scalar_vec: ScalarVectorInput): PointOutput;
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @param {number} c
* @returns {PointOutput}
*/
export function compute_msm_with_c(point_vec: PointVectorInput, scalar_vec: ScalarVectorInput, c: number): PointOutput;
/**
*/
export class InstanceObject {
  free(): void;
/**
* @returns {PointVectorInput}
*/
  points(): PointVectorInput;
/**
* @returns {ScalarVectorInput}
*/
  scalars(): ScalarVectorInput;
/**
*/
  readonly length: number;
}
/**
*/
export class InstanceObjectVector {
  free(): void;
/**
* @param {number} i
* @returns {InstanceObject}
*/
  at(i: number): InstanceObject;
/**
*/
  readonly length: number;
}
/**
*/
export class PointOutput {
  free(): void;
/**
* @returns {Array<any>}
*/
  toJsArray(): Array<any>;
}
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
