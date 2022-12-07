use fpga::SendBuffer64;
use msm_fpga::{random_points, timed};
use msm_fpga::{App, Instruction};
use our_bls12_377::{G1Affine, G1Projective, G1TEProjective};

fn main() {
    let size = 0;

    let f1 = fpga::F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size);
    #[allow(non_snake_case)]
    let P = timed("generating random points", || random_points(size))[0];
    println!("P: {}", P);

    app.set_points(&[P]);

    app.start();

    let mut cmds = SendBuffer64::default();
    cmds[0] = Instruction::new(-1);
    println!("{:016X}", cmds[0]);
    app.update(&cmds);

    println!("getting point...");
    let point: G1TEProjective = app.get_point();
    // println!("TE Projective point: {:?}", &point);
    let point: G1Projective = msm_fpga::twisted::into_weierstrass(&point);
    // println!("WE Projective point: {:?}", &point);
    println!("FPGA -P: {}", &point);

    println!("actual -P: {}", -P);

    assert_eq!(-P, point);

    println!("negative success!");
}
