use fpga::SendBuffer64;
use msm_fpga::{always_timed, timed, App, Instruction};
use msm_fpga::{harness_digits, load_harness_points};

fn main() {
    let size = std::env::args()
        .nth(1)
        .expect("pass with SIZE argument")
        .parse()
        .expect("SIZE invalid as u8");

    let name = std::env::args().nth(2).expect("pass with NAME argument");

    let f1 = fpga::F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size);

    // let (beta, points) = harness_points(size as _);
    let (beta, points) = load_harness_points(size as _, &name);

    let (digits, sum) = harness_digits(&beta, size as _);
    // let (digits, sum) = msm_fpga::noconflict_harness_digits(&beta, size as _);

    // println!("{:?}", &digits[..10]);

    timed("setting points", || app.set_preprocessed_points(&points));

    let point = always_timed("column sum", || {
        // app.print_stats();
        app.start();

        let mut cmds = SendBuffer64::default();

        for chunk in digits.chunks(8) {
            for (digit, cmd) in chunk.iter().zip(cmds.iter_mut()) {
                // println!("{}", digit);
                *cmd = Instruction::new(*digit as i16);
                // schedule.push(*cmd);
            }

            app.update(&cmds);
        }

        // for digit in column.iter() {
        //     cmds.fill(0);
        //     cmds[0] = Instruction::new(*digit);
        //     app.update(&cmds);
        // }

        app.flush();

        // println!("getting point...");
        let point = app.get_point();
        msm_fpga::twisted::into_weierstrass(&point)
    });
    // println!("point: {}", point);
    // let recv: G1Affine = point.into();
    // println!("x: {}", recv.x);
    // println!("y: {}", recv.y);
    //
    // let recv_te = into_twisted(&recv);
    // println!("te point: {}", recv_te);
    // println!("x: {}", recv_te.x);
    // println!("y: {}", recv_te.y);
    //
    // app.print_stats();

    println!("verifying");

    if point != sum {
        println!("\n==> FAILURE <==");
        std::process::exit(1);
    } else {
        println!("\n==> SUCCESS <==");
    }
}
