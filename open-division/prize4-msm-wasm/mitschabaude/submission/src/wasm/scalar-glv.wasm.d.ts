export let memory: WebAssembly.Memory;
export let dataOffset: WebAssembly.Global;

export function decompose(x: number): void;
export function barrett(x: number): void;
export function multiplySchoolbook(xy: number, x: number, y: number): void;
// helpers
export function toPackedBytes(bytes: number, x: number): void;
export function fromPackedBytes(x: number, bytes: number): void;
export function fromPackedBytesDouble(x: number, bytes: number): void;
export function extractBitSlice(
  x: number,
  startBit: number,
  bitLength: number
): number;
