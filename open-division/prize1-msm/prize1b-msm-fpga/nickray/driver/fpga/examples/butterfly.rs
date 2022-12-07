use fpga::{Fpga as _, ReceiveBuffer, Result, SendBuffer, F1};

fn encode(a: u64, b: u64, twiddle: u64) -> SendBuffer {
    let mut buffer = SendBuffer::default();
    buffer[..8].copy_from_slice(&a.to_le_bytes());
    buffer[8..16].copy_from_slice(&b.to_le_bytes());
    buffer[16..24].copy_from_slice(&twiddle.to_le_bytes());
    buffer
}

fn decode(buffer: &ReceiveBuffer) -> (u64, u64) {
    let a = u64::from_le_bytes(buffer[..8].try_into().unwrap());
    let b = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
    (a, b)
}

fn main() -> Result<()> {
    let fpga = F1::new(0)?;
    println!("init");

    let index = 0;
    let send_buffer = encode(456, 123, 789);
    fpga.send(index, &send_buffer);
    println!("wrote");

    let recv_buffer = fpga.receive_alloc();
    println!("read");
    println!("response: {:?}", &recv_buffer);

    let (a, b) = decode(&recv_buffer);
    println!("a = {a}, b = {b}");

    Ok(())
}
