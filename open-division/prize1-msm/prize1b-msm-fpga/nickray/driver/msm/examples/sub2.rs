use fpga::SendBuffer64;
use msm_fpga::{random_points, timed};
use msm_fpga::{App, Instruction};
use our_bls12_377::{G1Affine, G1Projective, G1TEProjective};

fn main() {
    let size = 1;

    let f1 = fpga::F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size);
    let points = timed("generating random points", || random_points(size));

    #[allow(non_snake_case)]
    let P: G1Projective = points[0].into();
    println!("P: {}", P);
    #[allow(non_snake_case)]
    let Q: G1Projective = points[1].into();
    println!("Q: {}", Q);

    app.set_points(&points);

    app.start();

    let mut cmds = SendBuffer64::default();
    cmds[0] = Instruction::new(1);
    cmds[1] = Instruction::new(-1);
    app.update(&cmds);

    println!("getting point...");
    let point: G1TEProjective = app.get_point();
    // println!("TE Projective point: {:?}", &point);
    let point: G1Projective = msm_fpga::twisted::into_weierstrass(&point);
    // println!("WE Projective point: {:?}", &point);
    let point: G1Affine = point.into();
    println!("WE Affine point: {}", &point);

    println!("P + Q: {}", P + Q);
    println!("P - Q: {}", P - Q);
    println!("-P - Q: {}", -P - Q);
    println!("-P + Q: {}", -P + Q);
    assert_eq!(P - Q, point);

    println!("sub2 success!");
}
