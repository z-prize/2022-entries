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
